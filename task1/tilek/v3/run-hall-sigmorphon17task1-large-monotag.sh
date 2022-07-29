#!/bin/bash
arch=$1
lang=$2
exp=$3
res=high
seed=2

# for lang in $(cat data/conll2017/lang.txt); do
python src/train.py \
    --dataset sigmorphon17task1 \
    --train /pscratch/sd/c/chubakov/code/morphology/inflection/data/$lang-train-$res \
    --dev /pscratch/sd/c/chubakov/code/morphology/inflection/data/$lang-dev \
    --test /pscratch/sd/c/chubakov/code/morphology/inflection/data/$lang-dev \
    --model model/sigmorphon17-task1/large/hall-monotag-$arch/$exp/$lang-$res \
    --init init/sigmorphon17-task1/large/seed-$seed/$lang-$res --seed $seed \
    --embed_dim 200 --src_hs 400 --trg_hs 400 --dropout 0.4 \
    --src_layer 2 --trg_layer 1 --max_norm 5 --nb_sample 4 \
    --arch $arch --gpuid 0 --estop 1e-8 --epochs 50 --bs 20 --indtag --mono
# done
