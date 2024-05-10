OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="3" \
  python \
    validate.py \
    --data-dir ./datasets/IMNET1k \
    --dataset imagenet \
    --model lemevit_base \
    --num-classes 1000 \
    --checkpoint outputs/classification/lemevit_base_224/exp4/model_best_.pth.tar \
    --img-size 224 \
    --batch-size 256 \
    --amp \


# OMP_NUM_THREADS=1 \
# CUDA_VISIBLE_DEVICES="3" \
#   python \
#     validate.py \
#     --data-dir ./datasets/millionaid \
#     --dataset torch/millionaid \
#     --model ViTAE_Window_NoShift_12_basic_stages4_14 \
#     --num-classes 51 \
#     --img-size 224 \
#     --batch-size 64 \
#     --amp
