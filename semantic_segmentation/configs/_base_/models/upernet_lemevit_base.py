# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MLeMeViT',
        depth=[2, 4, 4, 18, 4],
        embed_dim=[96, 96, 192, 384, 512], 
        head_dim=32,
        mlp_ratios=[4, 4, 4, 4, 4],
        attn_type=["C","D","D","S","S"],
        queries_len=16,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        qk_dims=None,
        cpe_ks=3,
        pre_norm=True,
        mlp_dwconv=False,
        representation_size=None,
        layer_scale_init_value=-1,
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, avg_non_ignore=True)
            ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, avg_non_ignore=True)),
    # model training and testing settings
    train_cfg=dict(),
    #test_cfg=dict(mode='whole')
    test_cfg=dict(mode='slide', stride=(384,384), crop_size=(512,512))
    )