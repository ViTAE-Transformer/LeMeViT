# OMP_NUM_THREADS=1 \
# CUDA_VISIBLE_DEVICES="5,6" \
#   python \
#     validate.py \
#     --data-dir ./datasets/millionaid \
#     --dataset torch/millionaid \
#     --model mixformer_small \
#     --checkpoint outputs/scene_recognition/mixformer_small_224/exp3/model_best.pth.tar \
#     --num-classes 51 \
#     --img-size 224 \
#     --batch-size 256

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="3" \
  python \
    validate.py \
    --data-dir ./datasets/millionaid \
    --dataset torch/millionaid \
    --model ViTAE_Window_NoShift_12_basic_stages4_14 \
    --num-classes 51 \
    --img-size 224 \
    --batch-size 64 \
    --amp
