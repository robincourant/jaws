import torch.nn as nn


class EarlyStopping(nn.Module):
    """Callback adapting stopping training if criteria are reached."""

    def __init__(self, num_patience: int, min_delta: float):
        super().__init__()
        self.num_patience = num_patience
        self.min_delta = min_delta
        self.loss_buffer = None
        self.wait_count = 0

    def run_early_stopping_check(self, current_loss: float) -> bool:
        if self.wait_count == 0:
            self.loss_buffer = current_loss
            self.wait_count += 1
            return False

        if current_loss < self.loss_buffer - self.min_delta:
            self.loss_buffer = current_loss
            self.wait_count = 0
            return False

        if self.wait_count > self.num_patience:
            return True

        self.wait_count += 1
        return False
