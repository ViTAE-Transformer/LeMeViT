OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="1" \
  python -m debugpy --listen localhost:5678 --wait-for-client \
    main.py \
    --config configs/mixformer.yaml \
    --output outputs/classification \
    --experiment debug \
    --override \
    --model mixformer_tiny \
    --data-dir datasets/IMNET1k \
    --img-size 224 \
    --batch-size 100
