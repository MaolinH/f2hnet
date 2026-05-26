_base_ = [
    '../_base_/models/upernet_f2hnet.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        in_dim=3,
        depths=[3, 12, 24, 4],
        dims=[96, 192, 384, 768],
        num_heads=(3, 6, 12, 24),
        dilation=(8, 4, 1, 1),
        seq_size=(7, 7, 14, 7),
        layer_scale=(None, None, 1., 1.),
        drop_path_rate=0.5,
        out_indices=[0, 1, 2, 3],
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))


param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
data_root = 'data/ADE20K/ADEChallengeData2016'

train_dataloader = dict(batch_size=4,num_workers=2,data_root=data_root)
val_dataloader = dict(batch_size=1,num_workers=2,data_root=data_root)
test_dataloader = val_dataloader