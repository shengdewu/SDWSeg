# model settings
num_classes = 10
ignore_index = 255

model = dict(
    name='EncoderDecoder',
    ignore_index=ignore_index,
    encoder=dict(
        name='RegSegEncoder',
        stem_channels=32,
        stages=[
            # out_channels, dilations, group_width, stride, se_ratio
            [[48, [1], 16, 2, 4]],
            [[96, [1], 16, 2, 4], *[[96, [1], 16, 1, 4]] * 2],
            [
                [128, [1], 16, 2, 4],
                [128, [1], 16, 1, 4],
                [128, [1, 2], 16, 1, 4],
                *[[128, [1, 4], 16, 1, 4]] * 4,
                *[[128, [1, 14], 16, 1, 4]] * 6,
                [160, [1, 14], 16, 1, 4],
            ],
        ],
        out_indices=(0, 1, 2)
    ),
    decoder=dict(
        name='RegSegDecoder',
        in_channels=[48, 96, 160],
        projection_out_channels=[8, 128, 128],
        interpolation='bilinear',
        align_corners=False,
        head_channels=64,
        dropout=0.0,
        upsample_factor=4,
        num_classes=num_classes,
        ignore_index=ignore_index,
        loss_cfg=[
            dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=False, ignore_index=ignore_index)),
            dict(name='SSIMLoss', param=dict(lambda_weight=1.0, apply_sigmoid=False, ignore_index=ignore_index)),
            dict(name='IOULoss', param=dict(lambda_weight=1.0, apply_sigmoid=False, ignore_index=ignore_index)),
        ]),
)

trainer = dict(
    name='SegTrainer',
    weights='',
    enable_epoch_method=False,
    model=dict(
        name='SegModel',
        generator=model,
    )
)
