# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 28/06/2021 
# @Contact: michealabaho265@gmail.com

import logging
import math
import os
import numpy as np
import transformers
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datasets import load_dataset
import prepare_data
import torch
import transformers
import json
import outcome_mask
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    PreTrainedTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
)
from torch.utils.data import DataLoader
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from argparse import ArgumentParser
from tqdm import tqdm


class Outcome_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        # self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings.input_ids)

#loading a tokenizer and tokenizing the input
def tokenize(input_text, tokenizer):
    tokenized_encodings = tokenizer(input_text, max_length=512, truncation=True, padding=True)
    tokenized_encodings['labels'] = tokenized_encodings.input_ids.copy()
    return tokenized_encodings

def train(model, train_data, eval_data, train_args, tokenizer, trainer_API=False):
    #use the hugging face trainer API
    if trainer_API:
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            pad_to_multiple_of=None,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        train_result = trainer.train() if train_args.resume_from_checkpoint is None else trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_data)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if train_args.do_eval:
            eval_results = trainer.evaluate()
            eval_results["eval_samples"] = len(eval_data)
            try:
                perplexity = math.exp(eval_results["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            eval_results["perplexity"] = perplexity
            trainer.log_metrics("eval", eval_results)
            trainer.save_metrics("eval", eval_results)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle_train_data)
        optim = AdamW(model.parameters(), lr=args.learning_rate)
        for epoch in range(args.epoch):
            training_loop = tqdm(train_loader, leave=True)
            for batch in training_loop:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input)

def evaluate(data, labels, train_args, model, tokenizer):
    seqs = []
    labels_copy = list(set([x for y in labels.copy() for x in y.split()]))
    labels_copy.remove('O')
    with open(train_args.output_dir + '/fact_prediction.txt', 'w') as fact:
        for txt, lab in zip(data, labels):
            masked_sequence = f" ".join([tokenizer.mask_token if j in labels_copy else i for i,j in zip(txt.split(), lab.split())])
            seqs.append(masked_sequence)
            input = tokenizer.encode(masked_sequence, return_tensors="pt")
            mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
            logits = model(input)[0]
            if tokenizer.mask_token in masked_sequence:
                top_3_token_predictions = []
                # print(txt, '\n', masked_sequence, '\n')
                fact.write(txt+'\n')
                for mask in mask_token_index:
                    mask = torch.unsqueeze(mask, 0)
                    mask_token_logits = logits[0, mask, :]
                    top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()
                    top_3_token_predictions.append(top_3_tokens)
                fact.write(str(masked_sequence)+'\n')
                for j,token_ids in enumerate(top_3_token_predictions):
                    # print(PreTrainedTokenizerFast.convert_ids_to_tokens(tokenizer, ids=token_ids))
                    masked_sequence = masked_sequence.replace(tokenizer.mask_token, tokenizer.decode([token_ids[0]]), 1)
                    # print(masked_sequence)
                    fact.write(str(masked_sequence))
                    fact.write('\n')
                fact.write('\n')
        fact.close()


def main(args):
    outcomes = json.load(open(args.data + 'outcome_occurrence.json', 'r'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = HfArgumentParser(TrainingArguments)
    train_args, = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
    model.to(device)
    print(train_args)
    print('---------------------------------------Read and Tokenize the data-------------------------------------------')
    dataset, dataset_labels, train_samples_len, eval_samples_len = prepare_data.fetch_data(files=[args.data + '/train.txt', args.data + '/dev.txt'])
    tokenized_input = tokenize(dataset, tokenizer)
    train_dataset, eval_dataset = dataset[:train_samples_len], dataset[train_samples_len:]
    train_dataset_labels, eval_dataset_labels = dataset_labels[:train_samples_len], dataset_labels[train_samples_len:]
    train_tokenized_input, eval_tokenized_input = {}, {}
    for k, v in tokenized_input.items():
        train_tokenized_input[k] = v[:train_samples_len]
        eval_tokenized_input[k] = v[train_samples_len:]

    # prepare data
    if train_args.do_train:
        if args.custom_mask:
            train_tokenized_input = outcome_mask.custom_mask(tokenizer=tokenizer, tokenized_input=train_tokenized_input, dataset=train_dataset, dataset_labels=train_dataset_labels)
        train_data = Outcome_Dataset(train_tokenized_input)

    if train_args.do_eval:
        if args.custom_mask:
            eval_tokenized_input = outcome_mask.custom_mask(tokenizer=tokenizer, tokenized_input=eval_tokenized_input, dataset=eval_dataset, dataset_labels=eval_dataset_labels)
        eval_data = Outcome_Dataset(eval_tokenized_input)

    #training and evaluation
    if train_args.do_train:
        train(model=model, train_data=train_data, eval_data=eval_data, train_args=train_args, tokenizer=tokenizer, trainer_API=True)

    if args.do_fill:
        eval_model = AutoModelForMaskedLM.from_pretrained(train_args.output_dir)
        eval_data, eval_data_labels = prepare_data.read_outcome_data_to_sentences(args.data+'train.txt')
        evaluate(data=eval_data, labels=eval_data_labels, train_args=train_args, model=eval_model, tokenizer=tokenizer)

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='data/ebm-comet/', help='source of data')
    par.add_argument('--do_train', action='store_true', help='call if you want to fine-tune pretrained model on dataset')
    par.add_argument('--do_eval', action='store_true', help='call if you want to evaluate a fine-tuned pretrained model on validation dataset')
    par.add_argument('--do_fill', action='store_true', help='Get the model to fill in masked entities in the validation dataset')
    par.add_argument('--output_dir', default='output', help='indicate where you want model and results to be stored')
    par.add_argument('--overwrite_output_dir', action='store_true', help='overwrite existing output directory')
    par.add_argument('--pretrained_model', default='dmis-lab/biobert-v1.1', help='pre-trained model available via hugging face e.g. dmis-lab/biobert-v1.1')
    par.add_argument('--num_train_epochs', default=3, help='number of training epochs')
    par.add_argument('--per_device_train_batch_size', default=8, help='training batch size')
    par.add_argument('--per_device_eval_batch_size', default=8, help='eval batch size')
    par.add_argument('--save_steps', default=1500, help='eval batch size')
    par.add_argument('--resume_from_checkpoint', default=None, help='location of most recent model checkpoint')
    args = par.parse_args()
    main(args)
