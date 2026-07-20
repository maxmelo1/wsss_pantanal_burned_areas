_base_ = [
    '../_base_/datasets/queimadas_planet.py',
    '../_base_/default_runtime_epoch.py',
    '../_base_/schedules/schedule_30e.py'
]

custom_imports = dict(
    imports=['mmseg.models.segmentors.transunet_segmentor'],
    allow_failed_imports=False)


# model settings
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[0., 0., 0., 0.],
    std=[1., 1., 1., 1.],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='TransUNetSegmentor',
    data_preprocessor=data_preprocessor,
    vit_name='R50-ViT-B_16',
    img_size=512,
    num_classes=2,
    in_channels=4,
    n_skip=3,
    pretrained_path='/mnt/58B0FAA2B0FA85B2/max/Queimadas/TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz',
    loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5, loss_name='loss_ce'),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5)
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# schedule
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)

# dataloader settings
train_dataloader = dict(
    batch_size=6,
    num_workers=2,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
)
test_dataloader = val_dataloader
