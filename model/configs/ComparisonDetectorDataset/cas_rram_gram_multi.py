model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='BatchRoIHeadFPN',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractorFPN',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            fpn_level=0,
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='UnsharedConvFCRoIAttentionBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=11,
            attention_hidden_channels=128,
            attention_pool_size=2,
            attention_pool_size_gram=4,
            subsample='naive',
            combination='cas_rram_gram',
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=2.0))))
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=500,
        max_num=500,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='AutoAugment',
#         policies=[[{
#             'type':
#             'Resize',
#             'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                           (736, 1333), (768, 1333), (800, 1333)],
#             'multiscale_mode':
#             'value',
#             'keep_ratio':
#             True
#         }],
#                 [{
#                     'type': 'Resize',
#                     'img_scale': [(400, 4200), (500, 4200), (600, 4200)],
#                     'multiscale_mode': 'value',
#                     'keep_ratio': True
#                 }, {
#                     'type': 'RandomCrop',
#                 #   'crop_type': 'absolute_range',
#                     'crop_size': (384, 600),
#                     'allow_negative_crop': True
#                 }, {
#                     'type':'Resize',
#                     'img_scale': [(480, 1333), (512, 1333), (544, 1333),
#                                 (576, 1333), (608, 1333), (640, 1333),
#                                 (672, 1333), (704, 1333), (736, 1333),
#                                 (768, 1333), (800, 1333)],
#                     'multiscale_mode':'value',
#                     # 'override': True,
#                     # 'keep_ratio': True
#                 }]]),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
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
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
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
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='/root/commonfile/data/ComparisonDetectorDataset/train.json',
        img_prefix='/root/commonfile/data/ComparisonDetectorDataset/train/',
        pipeline=train_pipeline,
        classes=('ascus', 'asch', 'lsil', 'hsil', 'scc',
                 'agc', 'trichomonas', 'candida',
                 'flora', 'herps', 'actinomyces')),
    val=dict(
        type='CocoDataset',
        ann_file='/root/commonfile/data/ComparisonDetectorDataset/test.json',
        img_prefix='/root/commonfile/data/ComparisonDetectorDataset/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
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
        classes=('ascus', 'asch', 'lsil', 'hsil', 'scc',
                 'agc', 'trichomonas', 'candida',
                 'flora', 'herps', 'actinomyces')),
    test=dict(
        type='CocoDataset',
        ann_file='/root/commonfile/data/ComparisonDetectorDataset/test.json',
        img_prefix='/root/commonfile/data/ComparisonDetectorDataset/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
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
        classes=('ascus', 'asch', 'lsil', 'hsil', 'scc',
                 'agc', 'trichomonas', 'candida',
                 'flora', 'herps', 'actinomyces')))
evaluation = dict(interval=2, metric='bbox')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[32, 40, 45])
total_epochs = 48
checkpoint_config = dict(interval=2)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
classes = ('ascus', 'asch', 'lsil', 'hsil', 'scc',
                 'agc', 'trichomonas', 'candida',
                 'flora', 'herps', 'actinomyces')
work_dir = 'work_dir/roi_attention/cas_rram_gram/128_naive_unshared_loss_level0_multiscale_48ep_0520/'
gpu_ids = range(0, 1)
