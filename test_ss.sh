OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="6" \
  torchrun \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 1 \
    semantic_segmentation/tools/test.py \
    semantic_segmentation/configs/upernet/upernet_mixformer_rsp_512x512_80k_potsdam.py \
    outputs/semantic_segmentation/exp1/iter_80000.pth \
    --eval mIoU 'mFscore' \
    # --show-dir outputs/semantic_segmentation/exp1/vis \
