# OMP_NUM_THREADS=1 \
# CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" \
#   torchrun \
#     --rdzv_backend c10d \
#     --nnodes 1 \
#     --nproc_per_node 6 \
#     main.py \
#     --config configs/mixformer.yaml \
#     --output outputs/classification \
#     --experiment exp2 \
#     --model mixformer_tiny_v2 \
#     --img-size 224 \
#     --batch-size 200

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="4,5,6,7" \
  torchrun \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 4 \
    main.py \
    --config configs/mixformer_scene_recognition.yaml \
    --output outputs/visulization \
    --experiment exp1 \
    --model mixformer_tiny \
    --initial-checkpoint outputs/scene_recognition/mixformer_tiny_224/exp6/model_best.pth.tar \
    --img-size 224 \
    --batch-size 256