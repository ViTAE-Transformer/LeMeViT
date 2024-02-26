OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="4,5,6,7" \
  torchrun \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 4 \
    semantic_segmentation/tools/train.py \
    semantic_segmentation/configs/upernet/upernet_mixformer_rsp_512x512_80k_potsdam.py \
    --work-dir outputs/semantic_segmentation/tiny-exp1 \
    --launcher 'pytorch' \
    --options 'find_unused_parameters'=True \
    # --resume outputs/semantic_segmentation/exp3/latest.pth
