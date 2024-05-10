OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8" \
  torchrun \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes 1 \
    --nproc_per_node 8 \
    main.py \
    --config configs/lemevit_scene_recognition.yaml \
    --output outputs/scene_recognition \
    --experiment exp-0 \
    --model lemevit_tiny \
    --img-size 224 \
    --batch-size 256 \
    # --resume outputs/classification/lemevit_tiny_224/exp3/model_best_.pth.tar \
    # --no-resume-opt
    # --resume outputs/classification/lemevit_base_224/exp-1-0/checkpoint-188.pth.tar

# OMP_NUM_THREADS=1 \
# CUDA_VISIBLE_DEVICES="0" \
#   torchrun \
#     --rdzv_backend c10d \
#     --rdzv-endpoint=localhost:0 \
#     --nnodes 1 \
#     --nproc_per_node 1 \
#     main.py \
#     --config configs/lemevit_scene_recognition.yaml \
#     --output outputs/scene_recognition \
#     --experiment ablation_without_stem \
#     --model lemevit_small \
#     --img-size 224 \
#     --batch-size 256
