_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained='outputs/classification/mixformer_small_224/exp6/last.pth.tar',
    backbone=dict(
        type='MixFormer',
        depth=[1, 2, 2, 6, 2],
        embed_dim=[96, 96, 192, 320, 384], 
        head_dim=32,
        mlp_ratios=[4, 4, 4, 4, 4],
        attn_type=["STEM","M","M","S","S"],
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
        use_checkpoint_stages=[],
        frozen_stages= [-1],
        ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 320, 384],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5)
)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0004, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)

runner = dict(type='EpochBasedRunner', max_epochs=12)
fp16 = dict(loss_scale=512.0)
find_unused_parameters=True
