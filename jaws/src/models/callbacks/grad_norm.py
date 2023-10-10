from typing import List

import torch
import torch.nn as nn


class GradNorm(nn.Module):
    """
    Callback adapting loss weights during the training for MTL.
    Implementation of https://arxiv.org/pdf/1711.02257.pdf.
    Note that: `pl_module` must have `loss_weights` and `task_losses`
    attributes and `model._get_shared_layer` method.

    Code adapted from: https://github.com/falkaer/artist-group-factors/
    """

    def __init__(self, num_tasks: int, alpha: float):
        super().__init__()
        self.num_tasks = num_tasks
        self.loss_weights = nn.Parameter(torch.ones(num_tasks, requires_grad=True))
        self.alpha = alpha
        self._batch_index = 0

    def fit(self, task_losses: torch.Tensor, shared_parameters: nn.Parameter):
        """Fit the loss weights according to the gradnorm."""
        # Zero the w_i(t) gradients to update the weights using gradnorm loss
        self.loss_weights.grad = 0.0 * self.loss_weights.grad
        W = list(shared_parameters)

        norms = []
        for task_index, (w_i, L_i) in enumerate(zip(self.loss_weights, task_losses)):
            # Retain the graph until the last pass
            retain_graph = True if task_index != self.num_tasks - 1 else False
            # Gradient of L_i(t) w.r.t. W
            gLgW = torch.autograd.grad(L_i, W, retain_graph=retain_graph)
            # G^{(i)}_W(t)
            norms.append(torch.norm(w_i * gLgW[0]))
        norms = torch.stack(norms)

        # Set L(0)
        if self._batch_index == 0:
            self.initial_losses = task_losses.detach()

        # Compute the constant term without accumulating gradients
        # as it should stay constant during back-propagation
        with torch.no_grad():
            # Loss ratios \curl{L}(t)
            loss_ratios = task_losses / self.initial_losses
            # Inverse training rate r(t)
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            constant_term = norms.mean() * (inverse_train_rates**self.alpha)

        # Write out the gradnorm loss L_grad and set the weight gradients
        grad_norm_loss = (norms - constant_term).abs().sum()
        self.loss_weights.grad = torch.autograd.grad(grad_norm_loss, self.loss_weights)[
            0
        ]

        self._batch_index += 1

    def normalize_weights(self) -> torch.Tensor:
        """Renormalize the gradient weights."""
        with torch.no_grad():
            normalize_coeff = len(self.loss_weights) / self.loss_weights.sum()
            self.loss_weights.data = self.loss_weights.data * normalize_coeff

    def _get_loss_weights(self, mask_weights: List[torch.Tensor]) -> torch.Tensor:
        """Return the loss weights for the current batch."""
        gradnorm_index, loss_weights = 0, []
        for task_weight in mask_weights:
            if task_weight:
                weight = self.loss_weights[gradnorm_index]
                loss_weights.append(weight)
                gradnorm_index += 1
            else:
                loss_weights.append(torch.tensor(0, device=self.loss_weights.device))
        loss_weights = torch.stack(loss_weights).clamp(min=0.05)

        return loss_weights
