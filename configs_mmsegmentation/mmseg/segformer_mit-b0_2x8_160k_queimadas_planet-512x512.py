_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/queimadas_planet.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k_custom.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)

data_preprocessor = dict(
    size=crop_size, 
    mean=[0., 0., 0., 0.],
    std=[1., 1., 1., 1.],
    bgr_to_rgb=False
)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=4,
    ),
    decode_head=dict(
        # out_channels=1,
        num_classes=2,
        norm_cfg=norm_cfg,
        # sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, ),#class_weight=[.5, 5.]),
            # dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ]
    )
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
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1000,
        end=80000,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=32, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader
