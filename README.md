# JAWS: Just a Wild Shot for Cinematic Transfer in Neural Radiance Fields

By Xi Wang*, Robin Courant*, Jinglei Shi, Eric Marchand and Marc Christie

CVPR 2023

### [Project Page](https://www.lix.polytechnique.fr/vista/projects/2023_cvpr_wang/) | [arXiv](https://arxiv.org/pdf/2303.15427.pdf) | [Paper + Supp](https://inria.hal.science/hal-04046701v1/file/main.pdf)

## Installation

1. Create working environment:
```
conda create --name jaws -y python=3.10
conda activate jaws
```

2. Install dependencies (adapt it according to your CUDA version):
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

3. Use the correct torch-ngp version:
```
mkdir ./lib
git clone git@github.com:ashawkey/torch-ngp.git
mv torch-ngp torch_ngp
cd torch_ngp
git checkout 3c14ad5d1a8a36f8d36604d1bbd91515fb4416fa
ln -s lib/torch_ngp dir_to/torch_ngp
```

4. Download `LitePose` [checkpoints](https://drive.google.com/drive/folders/1Jlh-bmS85RDWuspZUG-ncWYA7F8iXsa_?usp=drive_link) and puth them in `lib/LitePose/ckpt`

5. Download example dataset [flame_steak_frms_time](https://drive.google.com/file/d/15fO8J3G7k9X9cDb6LEorU60CdVnwMh1D/view?usp=drive_link) and put it in `./data`

# Usage

Train NeRF:
```
python jaws/run.py --config-name train_nerf data_dir=/path/to/dataset  xp_name=xp_name datamodule=jaws_dollyzoom.yaml
```

Launch JAWS
```
python jaws/run.py --config-name batch_jaws data_dir=path/to/data/dir/flame_steak_frms_time/ xp_name=xp_name jaws.target_dir=data/jaws_dolly_zoom_mask datamodule=jaws_dollyzoom.yaml
```

Render Images
```
python jaws/run.py --config-name render_jaws data_dir=path/to/data/dir/flame_steak_frms_time/ xp_name=xp_name jaws.target_dir=data/jaws_dolly_zoom_mask datamodule=jaws_dollyzoom.yaml render_target_dir=path/to/results/dir/final_res_n
```

# Citation:

```
@InProceedings{Wang_2023_CVPR,
    author    = {Wang, Xi and Courant, Robin and Shi, Jinglei and Marchand, Eric and Christie, Marc},
    title     = {JAWS: Just a Wild Shot for Cinematic Transfer in Neural Radiance Fields},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023},
}
```
