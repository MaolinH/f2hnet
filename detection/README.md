## Installation procedures

```
conda create --name mmdet python=3.8
conda activate mmdet
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install timm
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
git clone https://github.com/open-mmlab/mmdetection.git # 3.2.0
cd mmdetection
pip install -v -e .
```
## Train 
```
# F2HNet-T with mask rcnn 1x 
bash tools/train.sh configs/f2hnet/mask_rcnn_f2hnet_tiny_1x.py 8 --cfg-options model.backbone.init_cfg.checkpoint='checkpoints/tiny.pth' 
# F2HNet-T with mask rcnn 3x 
bash tools/train.sh configs/f2hnet/mask_rcnn_f2hnet_tiny_3x.py 8 --cfg-options model.backbone.init_cfg.checkpoint='checkpoints/tiny.pth'
# F2HNet-S with mask rcnn 3x 
bash tools/train.sh configs/f2hnet/mask_rcnn_f2hnet_small_3x.py 8 --cfg-options model.backbone.init_cfg.checkpoint='checkpoints/small.pth'
# F2HNet-B with mask rcnn 3x 
bash tools/train.sh configs/f2hnet/mask_rcnn_f2hnet_base_3x.py 8 --cfg-options model.backbone.init_cfg.checkpoint='checkpoints/base.pth'
```

## Test
```
# F2HNet-T with mask rcnn 1x 
bash tools/test.sh configs/f2hnet/mask_rcnn_f2hnet_tiny_1x.py ckpts/mask_rcnn_f2hnet_tiny_1x.pth 8 
# F2HNet-T with mask rcnn 3x 
bash tools/test.sh configs/f2hnet/mask_rcnn_f2hnet_tiny_3x.py ckpts/mask_rcnn_f2hnet_tiny_3x.pth 8 
# F2HNet-S with mask rcnn 3x 
bash tools/test.sh configs/f2hnet/mask_rcnn_f2hnet_small_3x.py ckpts/mask_rcnn_f2hnet_small_3x.pth 8 
# F2HNet-B with mask rcnn 3x 
bash tools/test.sh configs/f2hnet/mask_rcnn_f2hnet_base_3x.py ckpts/mask_rcnn_f2hnet_small_1x.pth 8 
```
