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
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
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

def train(model, train_data, train_args, tokenizer, trainer_API=False):
    #use the hugging face trainer API
    if trainer_API:
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_data if train_args.do_train else None,
            # eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
        )
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_data)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

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

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = HfArgumentParser(TrainingArguments)
    train_args, = parser.parse_args_into_dataclasses()
    # print('\n\n',len(train_args),'\n\n')
    if train_args.do_train:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
        model.to(device)
        train_data, train_data_labels = prepare_data.read_outcome_data_to_sentences(args.data+'train.txt')
        train_data = train_data[:20]
        train_tokenized_input = tokenize(train_data, tokenizer)
        train_data = Outcome_Dataset(train_tokenized_input)
        print(train_data)
        # train_loader = DataLoader(train_data, batch_size=16)
        train(model=model, train_data=train_data, train_args=train_args, tokenizer=tokenizer, trainer_API=True)

    if args.do_eval:
        eval_data, eval_data_labels = prepare_data.read_outcome_data_to_sentences(args.data+'dev.txt')
        eval_tokenized_input = tokenize(eval_data)

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='data/ebm-comet/', help='source of data')
    par.add_argument('--do_train', action='store_true', help='call if you want to fine-tune pretrained model on dataset')
    par.add_argument('--output_dir', default='output', help='indicate where you want model and results to be stored')
    par.add_argument('--pretrained_model', default='dmis-lab/biobert-v1.1', help='pre-trained model available via hugging face')
    par.add_argument('--per_device_train_batch_size', default=8, help='training batch size')
    args = par.parse_args()
    main(args)
