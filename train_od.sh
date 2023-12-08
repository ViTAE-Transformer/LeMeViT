OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="4,5,6,7" \
  torchrun \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 4 \
    object_detection/tools/train.py \
    object_detection/configs/obb/oriented_rcnn/faster_rcnn_orpn_mixformer_rsp_fpn_1x_dota10.py \
    --work-dir outputs/object_detection \
    --launcher 'pytorch' \
    --options 'find_unused_parameters'=True