OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="7" \
  python \
    object_detection/tools/get_flops.py \
    object_detection/configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_dota10.py \
    --shape 1024

# OMP_NUM_THREADS=1 \
# CUDA_VISIBLE_DEVICES="7" \
#   python \
#     semantic_segmentation/tools/get_flops.py \
#     semantic_segmentation/configs/upernet/upernet_lemevit_512x512_80k_potsdam.py \
#     --shape 512