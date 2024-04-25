# model settings
num_classes = 1
ignore_index = 255

norm_cfg = dict(type='BatchNorm2d')

model = dict(
    name='EncoderDecoder',
    ignore_index=ignore_index,
    encoder=dict(
        name='BiSeNetV2',
        in_channels=3,
        detail_channels=[64, 64, 128],
        semantic_channels=[16, 32, 64, 128],
        bga_channels=128,
        out_indices=[0, 1, 2, 3, 4]
    ),
    decoder=dict(
        name='FCNHead',
        in_channels=128,
        channels=1024,
        num_convs=1,
        in_index=0,
        concat_input=False,
        norm_cfg=norm_cfg,
        num_classes=num_classes,
        ignore_index=ignore_index,
        loss_cfg=[
            dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=False, ignore_index=ignore_index)),
        ]),
    auxiliary=[
        dict(
            name='FCNHead',
            in_channels=16,
            channels=16,
            num_convs=2,
            in_index=1,
            concat_input=False,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            ignore_index=ignore_index,
            loss_cfg=[
                dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=False, ignore_index=ignore_index)),
            ]),
        dict(
            name='FCNHead',
            in_channels=32,
            channels=64,
            num_convs=2,
            in_index=2,
            concat_input=False,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            ignore_index=ignore_index,
            loss_cfg=[
                dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=False, ignore_index=ignore_index)),
            ]),
        dict(
            name='FCNHead',
            in_channels=64,
            channels=256,
            num_convs=2,
            in_index=3,
            concat_input=False,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            ignore_index=ignore_index,
            loss_cfg=[
                dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=False, ignore_index=ignore_index)),
            ]),
        dict(
            name='FCNHead',
            in_channels=128,
            channels=1024,
            num_convs=2,
            in_index=4,
            concat_input=False,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            ignore_index=ignore_index,
            loss_cfg=[
                dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=False, ignore_index=ignore_index)),
            ]),
    ]
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
