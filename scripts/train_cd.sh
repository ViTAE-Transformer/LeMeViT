OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="3" \
  python change_detection/train.py \
    --backbone lemevit \
    --dataset cdd \
    --pretrained outputs/classification/lemevit_small_224/exp6/model_best_.pth.tar \
    --exp small-1-0