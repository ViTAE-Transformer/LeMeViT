
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="0" \
  torchrun \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 1 \
    object_detection/tools/test.py \
    object_detection/configs/obb/oriented_rcnn/faster_rcnn_orpn_lemevit_tiny_rsp_fpn_1x_dota10.py \
    outputs/object_detection/tiny-exp1/epoch_12_.pth \
    --format-only --options save_dir=outputs/object_detection/tiny-exp1/results nproc=1 \
    # --show --show-dir outputs/object_detection/tiny-exp2/vis 