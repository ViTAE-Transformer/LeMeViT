OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="0,1" \
  torchrun \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 2 \
    object_detection/tools/train.py \
    object_detection/configs/obb/oriented_rcnn/faster_rcnn_orpn_lemevit_small_rsp_fpn_1x_dota10.py \
    --work-dir outputs/object_detection/dota-small-exp8 \
    --launcher 'pytorch' \
    --options 'find_unused_parameters'=True