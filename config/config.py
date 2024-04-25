_base_ = [
    './base/dataset/data.py',
    './base/model/regseg50.py',
    './base/schedule/sgd_cosine.py',
]

dataloader = dict(
    num_workers=8,
)

# trainer = dict(
#     weights='RegSegEncoder-RegSegDecoder_final.pth',
# )

model = dict(
    decoder=dict(
        num_classes=1,
        loss_cfg=[
            dict(name='GeneralizedCELoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=255)),
            dict(name='SSIMLoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=255)),
            dict(name='IOULoss', param=dict(lambda_weight=1.0, apply_sigmoid=True, ignore_index=255)),
        ]),
)

solver = dict(
    train_per_batch=16,
    test_per_batch=8,
    max_iter=350000,
    checkpoint_period=5000,
    generator=dict(
        lr_scheduler=dict(
            enabled=True,
            type='LRMultiplierScheduler',
            params=dict(
                lr_scheduler_param=dict(
                    name='WarmupCosineLR',
                    gamma=0.1,
                    steps=[50000, 150000, 250000, 320000],
                ),
                warmup_factor=0.01,
                warmup_iter=1000,
                max_iter=350000,
            )
        ),
        optimizer=dict(
            type='AdamW',
            params=dict(
                lr=0.001,
                weight_decay=5E-4,
            )
        )
    )
)

output_dir = 'regseg50-608'
