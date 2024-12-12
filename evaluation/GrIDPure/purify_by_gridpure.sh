CUDA_VISIBLE_DEVICES=0 python gridpure_batch.py \
    --input_dir=$1 \
    --output_dir="$2" \
    --pure_steps=10 \
    --pure_iter_num=20 \
    --gamma=0.1 \
    --method CAAT