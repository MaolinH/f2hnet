

tools/train.sh \
    configs/f2hnet/f2hnet_small_inik_upernet_160k_ade20k_512x512.py \
    4 \
    --cfg-options model.backbone.init_cfg.checkpoint='checkpoints/small.pth'