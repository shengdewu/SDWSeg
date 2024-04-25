max_iter = 80000

solver = dict(
    # ema=dict(enabled=False,
    #          decay_rate=0.99,
    #          num_updates=0),
    train_per_batch=8,
    test_per_batch=8,
    max_iter=max_iter,
    max_keep=20,
    checkpoint_period=10000,
    generator=dict(
        lr_scheduler=dict(
            enabled=True,
            type='WarmupPolynomialDecay',
            params=dict(
                max_iter=max_iter,
                warmup_factor=0.01,
                warmup_iter=2000,
                power=0.9,
                end_lr=0.00005,
                simple=True
            )
        ),
        optimizer=dict(
            type='SGD',
            params=dict(
                momentum=0.9,
                lr=0.01,
                weight_decay=5E-4,
            )
        ),
        clip_gradients=dict(
            enabled=False,
        ),
    ),
)
