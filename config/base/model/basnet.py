# model settings
num_classes = 1
ignore_index = 255

norm_cfg = dict(type='BatchNorm2d')


model = dict(
    name='EncoderDecoder',
    ignore_index=ignore_index,
    encoder=dict(
        name='ResNet',
        depth=34,
        in_channels=3,
        out_indices=[1, 2, 3, 4],
        stem_sharp=False
    ),
    decoder=dict(
        name='BASNet',
        in_channels=[64, 128, 256, 512],
        norm_cfg=norm_cfg,
        num_classes=num_classes,
        ignore_index=ignore_index,
        loss_cfg=[
            dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=ignore_index)),
        ]),
    auxiliary=[
        dict(
            name='FCNHead',
            in_channels=64,
            channels=64,
            num_convs=0,
            in_index=0,
            concat_input=False,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            ignore_index=ignore_index,
            loss_cfg=[
                dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=ignore_index)),
            ]),
        dict(
            name='FCNHead',
            in_channels=128,
            channels=128,
            num_convs=0,
            in_index=1,
            concat_input=False,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            ignore_index=ignore_index,
            loss_cfg=[
                dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=ignore_index)),
            ]),
        dict(
            name='FCNHead',
            in_channels=256,
            channels=256,
            num_convs=0,
            in_index=2,
            concat_input=False,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            ignore_index=ignore_index,
            loss_cfg=[
                dict(name='GeneralizedCELoss', param=dict(lambda_weight=0.1, apply_sigmoid=True, ignore_index=ignore_index)),
            ]),
        dict(
            name='FCNHead',
            in_channels=512,
            channels=512,
            num_convs=0,
            in_index=3,
            concat_input=False,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            ignore_index=ignore_index,
            loss_cfg=[
                dict(name='GeneralizedCELoss', param=dict(lambda_weight=0.01, apply_sigmoid=True, ignore_index=ignore_index)),
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
