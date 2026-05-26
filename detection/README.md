## Installation procedures

```
conda create --name mmdet python=3.8
conda activate mmdet
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install timm
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
git clone https://github.com/open-mmlab/mmdetection.git 
cd mmdetection
pip install -v -e .
```
