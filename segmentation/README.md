## Installation procedures

```
conda create --name mmseg python=3.8
conda activate mmseg
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install timm
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

## Train
```
# Tiny
bash tools/train.sh configs/f2hnet/f2hnet_tiny_inik_upernet_160k_ade20k_512x512.py 8 --cfg-options model.backbone.init_cfg.checkpoint='checkpoints/tiny.pth'
# Small
bash tools/train.sh configs/f2hnet/f2hnet_small_inik_upernet_160k_ade20k_512x512.py 8 --cfg-options model.backbone.init_cfg.checkpoint='checkpoints/small.pth'
# Base
bash tools/train.sh configs/f2hnet/f2hnet_base_inik_upernet_160k_ade20k_512x512.py 8 --cfg-options model.backbone.init_cfg.checkpoint='checkpoints/base.pth'
```
