optimizer = dict(
    type='SGD', lr=0.30000000000000004, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CycleEMAHook', cycle_epoch=40, resume_from=None, priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/aiteam/tykim/scratch/lightweight/mmdet_nano/checkpoints/picodet_l_640_40.4.pth'
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='PicoDet',
    backbone=dict(
        type='ESNet',
        model_size='l',
        out_indices=[2, 9, 12],
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        act_cfg=dict(type='HSwish'),
        se_cfg=dict(
            conv_cfg=None,
            ratio=4,
            act_cfg=({
                'type': 'ReLU'
            }, {
                'type': 'HSigmoid',
                'bias': 3.0,
                'divisor': 6.0,
                'max_value': 6.0
            })),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/aiteam/tykim/scratch/lightweight/mmdet_nano/checkpoints/ESNet_x1_25_pretrained_mmdet_format.pth'
        )),
    neck=dict(
        type='CSPPAN',
        in_channels=[160, 320, 640],
        act_cfg=dict(type='HSwish'),
        norm_cfg=dict(type='BN', requires_grad=True),
        out_channels=160,
        num_features=4,
        expansion=1,
        num_csp_blocks=1),
    bbox_head=dict(
        type='PicoDetHead',
        num_classes=1,
        in_channels=160,
        feat_channels=160,
        stacked_convs=4,
        kernel_size=5,
        share_cls_reg=True,
        use_depthwise=True,
        reg_max=7,
        act_cfg=dict(type='HSwish'),
        strides=[8, 16, 32, 64],
        norm_cfg=dict(type='BN', requires_grad=True),
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(
            type='SimOTAAssigner', num_classes=1, use_vfl=True, iou_weight=6),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.025,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
data_root = '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/hold_smartphone'
dataset_type = 'COCODataset'
img_scales = [(576, 576), (608, 608), (640, 640), (672, 672), (704, 704)]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MinIoURandomCrop'),
    dict(
        type='Resize',
        img_scale=[(576, 576), (608, 608), (640, 640), (672, 672), (704, 704)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/hold_smartphone/annos/train.json',
        img_prefix=
        '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/hold_smartphone/imgs',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='MinIoURandomCrop'),
            dict(
                type='Resize',
                img_scale=[(576, 576), (608, 608), (640, 640), (672, 672),
                           (704, 704)],
                multiscale_mode='value',
                keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('smartphone', )),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/hold_smartphone/annos/val.json',
        img_prefix=
        '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/hold_smartphone/imgs',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('smartphone', )),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/hold_smartphone/annos/val.json',
        img_prefix=
        '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/hold_smartphone/imgs',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('smartphone', )))
evaluation = dict(interval=10, metric='bbox')
work_dir = '/home/aiteam/tykim/scratch/lightweight/mmdet_nano/pico_l_640_exp'
device = 'cuda'
seed = 0
gpu_ids = range(0, 8)
