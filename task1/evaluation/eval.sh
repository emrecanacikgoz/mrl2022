#!/bin/bash
lang=$1
arch=$2
ckpt_dir=$3
model_path=$4
data_dir=$5
gpu_id=$6

export CUDA_VISIBLE_DEVICES=$gpu_id
echo "GPU: "$CUDA_VISIBLE_DEVICES

res=high
lr=0.001
scheduler=warmupinvsqr
max_steps=20000
warmup=4000
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=50
bs=400 # 256

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
# dropout=${2:-0.3}
dropout=0.3

python src/test.py \
    --dataset sigmorphon17task1 \
    --train $data_dir/$lang-train-$res \
    --dev $data_dir/$lang-dev \
    --test $data_dir/$lang-test \
    --model $ckpt_dir/$lang-$res-$decode \
    --load $model_path  \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --label_smooth $label_smooth --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --arch $arch --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc \
    --seed 0
    
 # --load checkpoints/transformer/transformer/hall-sigmorphon17-task1-dropout0.3/hall-1/deu-high-.nll_0.7268.acc_91.0.dist_0.429.epoch_494 \

    