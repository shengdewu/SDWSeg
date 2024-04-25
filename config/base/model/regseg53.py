# model settings
num_classes = 1
ignore_index = 255

model = dict(
    name='EncoderDecoder',
    ignore_index=ignore_index,
    encoder=dict(
        name='RegSegEncoder',
        stem_channels=32,
        stages=[
            [[48, [1], 24, 2, 4], [48, [1], 24, 1, 4]],
            [[120, [1], 24, 2, 4], *[[120, [1], 24, 1, 4]] * 5],
            [
                [336, [1], 24, 2, 4],
                [336, [1], 24, 1, 4],
                [336, [1, 2], 24, 1, 4],
                *[[336, [1, 4], 24, 1, 4]] * 4,
                *[[336, [1, 14], 24, 1, 4]] * 6,
                [384, [1, 14], 24, 1, 4],
            ],
        ],
        out_indices=(0, 1, 2)
    ),
    decoder=dict(
        name='RegSegDecoder',
        in_channels=[48, 120, 384],
        projection_out_channels=[16, 256, 256],
        interpolation='bilinear',
        align_corners=False,
        head_channels=128,
        dropout=0.0,
        upsample_factor=4,
        num_classes=num_classes,
        ignore_index=ignore_index,
        loss_cfg=[
            dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=ignore_index)),
            dict(name='SSIMLoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=ignore_index)),
            dict(name='IOULoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=ignore_index)),
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
