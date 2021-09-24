#!/bin/bash

CUDA_VISIBLE_DEVICES=$1
master_port=$2
data_dir=$3
input_dir=$4
output_dir=$5
model_type=$6
model_name_or_path=$7
data_aug=$8
per_device_eval_batch_size=$9
generation_type=${10}
max_length=${11}
dedup_ranking_type=${12}
add_extra_entity=${13}
search_attention_head_type=${14}
search_ranking_type=${15}
sentence=${16}
dist_const=${17}
beam_size=${18}
beam_mode=${19}
IFS=', ' read -r -a cuda_arr <<< "$CUDA_VISIBLE_DEVICES"
nproc_per_node=${#cuda_arr[@]}
output_dir_ext=${output_dir}${model_name_or_path}.${generation_type}.${data_aug}.${dedup_ranking_type}.${add_extra_entity}.${search_attention_head_type}.${search_ranking_type}.${sentence}.${dist_const}.${beam_size} # output folder containing the results

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=true

sent_dedup_type="entity_pair"
doc_dedup_type="whole"

python scripts/post_processing.py --input_dir=${output_dir} \
    --filepath=${output_dir}result.json \
    --dedup_ranking_type=${dedup_ranking_type} \
    --sent_dedup_type=${sent_dedup_type} \
    --doc_dedup_type=${doc_dedup_type}
