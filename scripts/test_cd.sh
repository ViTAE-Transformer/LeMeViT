# OMP_NUM_THREADS=1 \
# CUDA_VISIBLE_DEVICES="4,5,6,7" \
#   python \
#     change_detection/eval.py \
#     --backbone mixformer \
#     --dataset cdd \
#     --pretrained outputs/change_detection/cdd_mixformer/exp4/checkpoint_epoch_198.pth \
#     --exp exp4

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="1" \
  python \
    change_detection/eval.py \
    --backbone lemevit \
    --dataset cdd \
    --pretrained outputs/change_detection/cdd_mixformer/tiny-exp3/checkpoint_epoch_199_.pth \
    --exp test