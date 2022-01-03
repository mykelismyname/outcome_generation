#!/bin/bash
output_dir="/LOCAL/micheala/micheal2/mlm_models/No_trainer/ebm-comet/"
declare -a layers=("last" "average")
declare -a conditioning=(1 2)

for condition in "${conditioning[@]}"
do
  for layer in "${layers[@]}"
    do
      save_dir="bert"
      echo "${save_dir} ${condition} ${layer}"
      CUDA_AVAILABLE_DEVICES=0,1 \
      python train_back.py \
        --data data/ebm-comet/ \
        --output_dir "${output_dir}${save_dir}_${layer}_PC${condition}" \
        --pretrained_model 'bert-base-uncased' \
        --num_train_epochs 10 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --layers $layer \
        --prompt_conditioning $condition \
        --custom_mask --detection_loss --add_marker_tokens --do_train --do_eval
    done
    echo ""
done

for condition in "${conditioning[@]}"
do
  for layer in "${layers[@]}"
    do
      save_dir="biobert"
      echo "${save_dir} ${condition} ${layer}"
      CUDA_AVAILABLE_DEVICES=0,1 \
      python train_back.py \
        --data data/ebm-comet/ \
        --output_dir "${output_dir}${save_dir}_${layer}_PC${condition}" \
        --pretrained_model 'dmis-lab/biobert-v1.1' \
        --num_train_epochs 10 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --layers $layer \
        --prompt_conditioning $condition \
        --custom_mask --detection_loss --add_marker_tokens --do_train --do_eval
    done
  echo ""
done

for condition in "${conditioning[@]}"
do
  for layer in "${layers[@]}"
    do
      save_dir="umlsbert"
      echo "${save_dir} ${condition} ${layer}"
      CUDA_AVAILABLE_DEVICES=0,1 \
      python train_back.py \
        --data data/ebm-comet/ \
        --output_dir "${output_dir}${save_dir}_${layer}_PC${condition}" \
        --pretrained_model 'GanjinZero/UMLSBert_ENG' \
        --num_train_epochs 10 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --layers $layer \
        --prompt_conditioning $condition \
        --custom_mask --detection_loss --add_marker_tokens --do_train --do_eval
    done
  echo ""
done

for condition in "${conditioning[@]}"
do
  for layer in "${layers[@]}"
    do
      save_dir="roberta"
      echo "${save_dir} ${condition} ${layer}"
      CUDA_AVAILABLE_DEVICES=0,1 \
      python train_back.py \
        --data data/ebm-comet/ \
        --output_dir "${output_dir}${save_dir}_${layer}_PC${condition}" \
        --pretrained_model 'allenai/biomed_roberta_base' \
        --num_train_epochs 10 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --layers $layer \
        --prompt_conditioning $condition \
        --custom_mask --detection_loss --add_marker_tokens --do_train --do_eval
    done
  echo ""
done

for condition in "${conditioning[@]}"
do
  for layer in "${layers[@]}"
    do
      save_dir="scibert"
      echo "${save_dir} ${condition} ${layer}"
      CUDA_AVAILABLE_DEVICES=0,1 \
      python train_back.py \
        --data data/ebm-comet/ \
        --output_dir "${output_dir}${save_dir}_${layer}_PC${condition}" \
        --pretrained_model 'allenai/scibert_scivocab_uncased' \
        --num_train_epochs 10 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --layers $layer \
        --prompt_conditioning $condition \
        --custom_mask --detection_loss --add_marker_tokens --do_train --do_eval
    done
  echo ""
done