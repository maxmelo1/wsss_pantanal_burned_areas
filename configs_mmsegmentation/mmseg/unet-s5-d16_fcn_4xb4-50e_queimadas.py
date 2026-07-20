_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', 
    '../_base_/datasets/burned.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

crop_size = (256, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[0.041, 0.035, 0.023, 0.126],
    std=[0.019, 0.013, 0.01, 0.056],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg,
    ),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(128, 128))
)


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=32, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader