#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

config_file=$1
text_file=$2

source ${config_file}

params=${@:3}

#echo "Using CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES

transition_stats=$data_dir/transition_probs.tsv

python3 src/analyze_text.py \
--test_files $test_files \
--dev_files $dev_files \
--transition_stats $transition_stats \
--data_config $data_config \
--model_configs $model_configs \
--task_configs $task_configs \
--layer_configs $layer_configs \
--attention_configs "$attention_configs" \
$params $text_file

