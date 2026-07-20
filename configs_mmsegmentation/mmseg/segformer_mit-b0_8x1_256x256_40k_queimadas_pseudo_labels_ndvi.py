norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=4,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
        )),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='ImbalancedCrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            sigma=2)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'BurnedDataset'
data_root = '../queimadas/dataset2_bkp/'
img_norm_cfg = dict(
    mean=[0.041, 0.035, 0.023, 0.126],
    std=[0.019, 0.013, 0.01, 0.056],
    to_rgb=False)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='npy'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[0.041, 0.035, 0.023, 0.126],
        std=[0.019, 0.013, 0.01, 0.056],
        to_rgb=False),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='npy'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0.041, 0.035, 0.023, 0.126],
                std=[0.019, 0.013, 0.01, 0.056],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='BurnedDataset',
        data_root='../queimadas/dataset2_bkp/',
        img_dir='JPEGImages',
        ann_dir='pseudolabels_train_ndvi',
        split='data/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='npy'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[0.041, 0.035, 0.023, 0.126],
                std=[0.019, 0.013, 0.01, 0.056],
                to_rgb=False),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='BurnedDataset',
        data_root='../queimadas/dataset2_bkp/',
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='data/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='npy'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0.041, 0.035, 0.023, 0.126],
                        std=[0.019, 0.013, 0.01, 0.056],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='BurnedDataset',
        data_root='../queimadas/dataset2_bkp/',
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='data/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='npy'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0.041, 0.035, 0.023, 0.126],
                        std=[0.019, 0.013, 0.01, 0.056],
                        to_rgb=False),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(
    interval=4000, metric=['mIoU', 'mDice'], pre_eval=True, save_best='mIoU')
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'
work_dir = 'work_dirs/segformer_mit-b0_8x1_256x256_40k_queimadas_pseudo_labels_ndvi_imbalanced_loss_sigma_2_teste'
gpu_ids = [0]
auto_resume = False
