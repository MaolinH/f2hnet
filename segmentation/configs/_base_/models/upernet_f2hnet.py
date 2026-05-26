# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='F2hNet',
        pretrained_img_size=224,
        patch_size=4,
        dilation=(8, 4, 2, 1),
        seq_size=(7, 7, 14, 7),
        depths=(2, 4, 12, 4),
        in_dim=3,
        dims=(64, 128, 320, 512),
        kernel_sizes=(3, 3, 3, 3),
        num_heads=(2, 4, 10, 16),
        mlp_ratio=(4, 4, 4, 4),
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        mlp_drop=0.,
        drop_path_rate=0.1,
        qkv_bias=True,
        layer_scale=(None,None,1.,1.),
        rpe=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint='')  # dict(type='Pretrained', checkpoint=pretrained)
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 320, 640],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))