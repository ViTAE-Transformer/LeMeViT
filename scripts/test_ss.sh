OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="0" \
  torchrun \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 1 \
    semantic_segmentation/tools/test.py \
    semantic_segmentation/configs/upernet/upernet_lemevit_512x512_80k_potsdam.py \
    outputs/semantic_segmentation/tiny-exp2/tiny_best.pth \
    --eval mIoU 'mFscore' \
    # --show-dir outputs/semantic_segmentation/exp5/vis \
