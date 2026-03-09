#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0 # ! single-gpu debug

workdir='.'
model_name='ours'
datasets=('sintel' 'bonn' 'kitti')

# ! load configs
experiment=eval_videodepth
ckpt=./STream3R/model_alpha.ckpt


for data in "${datasets[@]}"; do
    # output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}"
    output_dir="${workdir}/eval_results/video_depth/${data}/${ckpt}"
    echo "$output_dir"

    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale&shift"

    # --align "metric"
    # --align "scale"
done
