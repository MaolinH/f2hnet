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
