_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    pretrained='outputs/classification/mixformer_small_224/exp6/last.pth.tar',
    backbone=dict(
        type='LeMeViT',
        depth=[1, 2, 2, 6, 2],
        embed_dim=[96, 96, 192, 320, 384], 
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
        use_checkpoint_stages=[],
        frozen_stages= [-1],
        ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 320, 384],
        out_channels=256,
        num_outs=5),
    )

###########################################################################################################
# https://github.com/Sense-X/UniFormer/blob/main/object_detection/exp/mask_rcnn_1x_hybrid_small/config.py
# We follow uniformer's optimizer and lr schedule
# but I do not like apex which requires extra MANUAL installation, hence we use pytorch's native amp instead

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,)
lr_config = dict(step=[8, 11])

# runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
# # do not use mmdet version fp16 -> WHY?
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

fp16 = dict()
find_unused_parameters=True