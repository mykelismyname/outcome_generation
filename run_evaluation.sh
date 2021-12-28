#!/bin/bash
source="/LOCAL/micheala/micheal2/mlm_models/No_trainer/ebm-comet/"
#data="data/ebm-comet/dev.txt"
outcomes="data/ebm-comet/outcome_occurrence.json"
declare -a metrics=("exact_match" "partial_match")

declare -a models=(
                  "BERT_MT_DL_14-12-2021_14-09-35"
                  "Biobert_MT_DL_14-12-2021_13-15-13"
                  "Bio_ClinicalBERT_MT_DL_14-12-2021_13-47-28"
                  "Biomed_roberta_MT_DL_14-12-2021_12-33-53"
                  "Scibert_MT_DL_14-12-2021_11-42-41"
                  "UmlsBert_MT_DL_14-12-2021_12-04-12"
                  )
train_data="data/ebm-comet/train.txt"
train_output_dir="/decoding_results/train/"
fewshot_data="data/ebm-comet/dev.txt"
fewshot_output_dir="/decoding_results/fewshot/"

for model in "${models[@]}"
do
  for m in "${metrics[@]}"
  do
    echo "${source} ${model} ${train_output_dir} ${m}"
    CUDA_AVAILABLE_DEVICES=0,1 \
    python train_back.py \
      --data $train_data \
      --output_dir "${source}${model}${train_output_dir}" \
      --pretrained_model "${source}${model}" \
      --fill_evaluation \
      --recall_metric $m \
      --mention_frequency $outcomes
  done
  echo ""
done

for model_f in "${models[@]}"
do
  for m_f in "${metrics[@]}"
  do
    echo "${source} ${model_f} ${fewshot_output_dir} ${m}"
    CUDA_AVAILABLE_DEVICES=0,1 \
    python train_back.py \
      --data $fewshot_data \
      --output_dir "${source}${model_f}${fewshot_output_dir}" \
      --pretrained_model "${source}${model_f}" \
      --fill_evaluation \
      --recall_metric $m_f \
      --mention_frequency $outcomes
  done
  echo ""
done