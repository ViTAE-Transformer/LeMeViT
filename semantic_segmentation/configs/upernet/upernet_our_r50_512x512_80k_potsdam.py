_base_ = [
    '../_base_/models/upernet_our_r50.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='../RS_CLS_finetune/output/resnet_50_224/epoch300/millionAID_224_None/0.0005_0.05_128/resnet/100/ckpt.pth',
    backbone=dict(),
    decode_head=dict(num_classes=5,ignore_index=5), 
    auxiliary_head=dict(num_classes=5,ignore_index=5)
    )
