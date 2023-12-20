OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" \
  torchrun \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 6 \
    main.py \
    --config configs/mixformer.yaml \
    --output outputs/classification \
    --experiment exp3 \
    --model mixformer_tiny \
    --resume outputs/classification/mixformer_tiny_224/exp3/last.pth.tar \
    --img-size 224 \
    --batch-size 300

# OMP_NUM_THREADS=1 \
# CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" \
#   torchrun \
#     --rdzv_backend c10d \
#     --rdzv-endpoint=localhost:0 \
#     --nnodes 1 \
#     --nproc_per_node 6 \
#     main.py \
#     --config configs/mixformer_scene_recognition.yaml \
#     --output outputs/scene_recognition \
#     --experiment exp_8 \
#     --model mixformer_tiny \
#     --img-size 224 \
#     --batch-size 256