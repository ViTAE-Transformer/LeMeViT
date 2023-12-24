# OMP_NUM_THREADS=1 \
# CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" \
#   torchrun \
#     --rdzv_backend c10d \
#     --rdzv-endpoint=localhost:0 \
#     --nnodes 1 \
#     --nproc_per_node 6 \
#     main.py \
#     --config configs/mixformer.yaml \
#     --output outputs/classification \
#     --experiment exp1 \
#     --model mixformer_tiny \
#     --img-size 224 \
#     --batch-size 400

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="4,5,6,7" \
  torchrun \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 4 \
    main.py \
    --config configs/mixformer_scene_recognition.yaml \
    --output outputs/scene_recognition \
    --experiment exp1 \
    --model mixformer_small \
    --img-size 224 \
    --batch-size 200