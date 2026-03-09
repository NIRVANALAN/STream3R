#!/bin/bash

set -e
export CUDA_VISIBLE_DEVICES=0 # ! single-gpu debug

workdir='.'
model_name='ours'
ckpt=./STream3R/model_alpha.ckpt

ckpt_name=${ckpt}
output_dir="${workdir}/eval_results/mv_recon/${model_name}/${ckpt_name}"
echo "$output_dir"

experiment=eval_mv_recon

python -m eval.mv_recon.launch \
    +experiment=${experiment} \
    model.encoder.pretrained_weights=${ckpt} \
    output_dir="$output_dir" \
    model_name="$model_name"
