#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES=0 # ! single-gpu debug

workdir='.'
model_name='ours'

ckpt=./STream3R/model_alpha.ckpt
ckpt_name=${ckpt}

model_weights="${workdir}/src/${ckpt_name}.pth"
datasets=('scannet' 'tum' 'sintel')

experiment=eval_relpose

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/relpose/${data}/${ckpt}"
    echo "$output_dir"

    python -m eval.relpose.launch \
        +experiment=${experiment} \
        model.encoder.pretrained_weights=${ckpt} \
        output_dir="$output_dir" \
        eval_dataset="$data" \
        size=512 \
        model_name="$model_name"
done


