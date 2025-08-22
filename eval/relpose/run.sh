#!/bin/bash
set -e

workdir='.'

datasets=('tum' 'sintel' 'scannet')
model_name='stream3r'

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/relpose/${model_name}/${data}"
    echo "$output_dir"
    accelerate launch --num_processes 1 --main_process_port 29558 eval/relpose/launch.py \
        --output_dir "$output_dir/" \
        --eval_dataset "$data"
done


