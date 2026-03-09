#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0 # ! single-gpu debug

workdir='.'
model_name='ours'
datasets=('sintel' 'bonn' 'kitti' 'nyu')
experiment=eval_monodepth
ckpt=./STream3R/model_alpha.ckpt

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}/${ckpt}"
    echo "$output_dir"

    python -m eval.monodepth.launch \
    +experiment=${experiment} \
    model.encoder.pretrained_weights=${ckpt} \
    output_dir="$output_dir" \
    eval_dataset="$data" \

done

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}/${ckpt}"
    python eval/monodepth/eval_metrics.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done