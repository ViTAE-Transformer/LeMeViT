OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" \
  torchrun \
    --rdzv_backend c10d \
    --nnodes 1 \
    --nproc_per_node 6 \
    main.py \
    --config configs/mixformer.yaml \
    --output outputs/classification \
    --experiment exp1 \
    --override \
    --model mixformer_tiny \
    --data-dir datasets/IMNET1k \
    --img-size 224 \
    --batch-size 400
