#!/bin/bash
lang=$1
arch=tagtransformer

res=high

data_dir=data
ckpt_dir=checkpoints/transformer

python src/test.py \
    --dataset sigmorphon17task1 \
    --train $data_dir/conll2017/all/task1/$lang-train-$res \
    --dev $data_dir/conll2017/all/task1/$lang-dev \
    --test $data_dir/conll2017/all/task1/$lang-dev \
    --model $ckpt_dir/$arch/sigmorphon17-task1-dropout$dropout/$lang-$res-$decode \
    --arch $arch --gpuid 0 \
    --load $ckpt_dir/tagtransformer/sigmorphon17-task1-dropout0.3/tur-high-.nll_0.6911.acc_97.3.dist_0.06.epoch_288
    
    