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
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from argparse import ArgumentParser

class Outcome_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings.input_ids)

#loading a tokenizer and tokenizing the input
def tokenize(input_text):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    tokenized_encodings = tokenizer(input_text, max_length=512, truncation=True, return_special_tokens_mask=True)
    return tokenized_encodings

def main(args):
    if args.train:
        train_data, train_data_labels = prepare_data.read_outcome_data_to_sentences(args.data+'train.txt')
        train_tokenized_input = tokenize(train_data)
        print(train_tokenized_input.input_ids[0])
        train_data = Outcome_Dataset(train_tokenized_input, train_data_labels)
        print(train_data)
if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='data/ebm-comet/', help='source of data')
    par.add_argument('--train', action='store_true', help='call if you want to fine-tune pretrained model on dataset')
    par.add_argument('--pretrained_model', default='dmis-lab/biobert-v1.1', help='pre-trained model available via hugging face')
    args = par.parse_args()
    main(args)
