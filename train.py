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
import random
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from datasets import Dataset, load_dataset, ClassLabel
import prepare_data
import torch
import transformers
import json
import argparse
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
    default_data_collator,
    PreTrainedTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    GPT2Tokenizer, GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel, T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration,
    AdamW,
)
from torch.utils.data import DataLoader
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from argparse import ArgumentParser
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from time import sleep
import detection_model as detection
import prompt_model as prompt_model
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
logging.basicConfig(level=logging.INFO)
accelerator = Accelerator()

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

@dataclass
class ExtraArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    data: Optional[str] = field(
        default='data/ebm-comet/',
        metadata={"help": "'source of data'"},
    )
    pretrained_model: Optional[str] = field(
        default='bert-base-uncased',
        metadata={"help": "The model checkpoint for weights initialization." "Don't set if you want to train a model from scratch."
        },
    )
    custom_mask: bool = field(
        default=False,
        metadata={"help": "Whether to customize the masking before input is sent to the models."},
    )
    trainer_api: bool = field(
        default=False,
        metadata={"help": "use the trainer api"},
    )
    do_fill: bool = field(
        default=False,
        metadata={"help": "Call if you want to fill in masks of the eval/test datasets for evaluation purposes"},
    )
    fill_evaluation: bool = field(
        default=False,
        metadata={"help": "Evaluate how well model recalls outcomes"},
    )
    mention_frequency: Optional[str] = field(
        default='outcome_occurrence.json',
        metadata={"help": "File with the outcome mention frequency."},
    )
    recall_metric: Optional[str] = field(
        default='exact_match',
        metadata={"help": "exact_match or partial match of all tokens"},
    )
    max_seq_length: Optional[int] = field(
        default=256,
        metadata={"help": "Max length of sequence"},
    )
    label_all_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "opt to label all tokens or not after tokenization"},
    )
    detection_loss: Optional[bool] = field(
        default=False,
        metadata={
            "help": "include an auxiliary cross entropy loss for outcome detection"},
    )
    det_batch_size: Optional[int] = field(
        default=64,
        metadata={
            "help": "Batch size of the detection model"},
    )
    det_hidden_dim: Optional[int] = field(
        default=768,
        metadata={
            "help": "Hidden state dimension of the detection model"},
    )
    partial_contexts: Optional[bool] = field(
        default=False,
        metadata={"help":"train mlm with prompts having different contexts, where contexts refers to prompts of different freq occurence"},
    )
    auxilliary_task_decay: Optional[float] = field(
        default=0.001,
        metadata={"help":"Decay the auxilliary loss during training"}
    )
    add_marker_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "Trigger the f_prompt  function to insert marker tokens at start of prompt"}
    )
    alm: Optional[bool] = field(
        default=False,
        metadata={"help": "amalgamate auto-regressive model hidden states"}
    )
    mask_id: int = field(
        default=103,
        metadata={"help": "id for the special mask token"}
    )

def train(model, alm_args, detection_args, train_data, eval_data, train_args, extra_args, tokenizer):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=None
    )

    #use the hugging face trainer API
    if extra_args.trainer_api:
        for i, j in enumerate(train_data):
            if i < 1:
                print(j)
                for m, n in j.items():
                    print(m, n.size())
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_data if train_args.do_train else None,
            eval_dataset=eval_data if train_args.do_eval else None,
            data_collator=data_collator if not extra_args.custom_mask else None,
            tokenizer=tokenizer
        )
        train_result = trainer.train(resume_from_checkpoint=None) if train_args.resume_from_checkpoint is None else trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_data)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if train_args.do_eval:
            print("---------------------------------------Evaluate-------------------------------------------")
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
        if not os.path.exists(train_args.output_dir):
            os.makedirs(train_args.output_dir)
        #Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": train_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        print('\nType of Dataset\n', train_data)
        # Log a few random samples from the training set:
        for index in range(1):
            print(tokenizer.all_special_tokens, tokenizer.all_special_ids)
            print("Sample {} of the training set: {}. {}, {}".format(index, train_data[index], type(train_data), type(train_data[index])))
            print(PreTrainedTokenizerFast.convert_ids_to_tokens(tokenizer, ids=train_data[index]['input_ids']))
            for i, j in train_data[index].items():
                print(i, len(j), type(j))

        print("\n------Before dataloader-------\n---{}-------\n---------{}".format(train_data, type(train_data)))
        train_data = train_data.remove_columns(['tokens', 'ner_tags'])
        eval_data = eval_data.remove_columns(['tokens', 'ner_tags'])
        print("\n------Before dataloader 2-------\n---{}-------\n---------{}".format(train_data, type(train_data)))

        if extra_args.custom_mask:
            train_dataset_dict = transformers.BatchEncoding(Dataset.to_dict(train_data))
            train_data = Outcome_Dataset(train_dataset_dict)
            train_loader = DataLoader(train_data, batch_size=train_args.per_device_train_batch_size)
            eval_dataset_dict = transformers.BatchEncoding(Dataset.to_dict(eval_data))
            eval_data = Outcome_Dataset(eval_dataset_dict)
            eval_loader = DataLoader(eval_data, batch_size=train_args.per_device_eval_batch_size)
        else:
            train_loader = DataLoader(train_data, batch_size=train_args.per_device_train_batch_size, collate_fn=data_collator)
            eval_loader = DataLoader(eval_data, batch_size=train_args.per_device_eval_batch_size, collate_fn=data_collator)
        optim = AdamW(optimizer_grouped_parameters, lr=train_args.learning_rate)

        if extra_args.detection_loss:
            detection_model, detection_criterion, detection_params = detection_args
        if extra_args.alm:
            pretrained_alm, alm_model = alm_args

        # alm_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        # alm_model.to(device)
        with open(os.path.join(train_args.output_dir, 'losses.txt'), 'w') as l:
            loss_metrics = {'training':[],'val':[]}
            l.write('Train \t Val \t Perplexity\n')
            logging.info("\n--------------------TRAINING BEGINS--------------------\n")
            for epoch in range(int(train_args.num_train_epochs)):
                model.train()
                training_loop = tqdm(train_loader, leave=True)
                train_loss = []
                for step, batch in enumerate(training_loop):
                    optim.zero_grad()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    ner_labels = batch['ner_labels'].to(device)
                    batch = transformers.BatchEncoding({'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels})

                    outputs = model(**batch, output_hidden_states=True)
                    if extra_args.alm:
                        pretrained_alm_outputs = pretrained_alm(**batch, output_hidden_states=True)
                        preds = alm_model(outputs.hidden_states, pretrained_alm_outputs.hidden_states)
                    # loss
                    mlm_loss = outputs.loss
                    # alm_loss = alm_outputs.loss
                    if extra_args.detection_loss:
                        mlm_hidden_states = outputs.hidden_states
                        ner_batch_losses = detection_model(input_ids, mlm_hidden_states, ner_labels, mode='average')
                    # logging.info("Step {}: Training MLM loss: {}, {} and ALM loss: {}".format(step, mlm_loss, ner_batch_losses, torch.mean(torch.stack(ner_batch_losses))))
                    loss = mlm_loss + (extra_args.auxilliary_task_decay * torch.mean(torch.stack(ner_batch_losses))) if extra_args.detection_loss else mlm_loss
                    train_loss.append(float(loss))
                    accelerator.backward(loss)
                    # loss.backward()
                    optim.step()

                train_epoch_loss = np.mean(train_loss)
                logging.info("Epoch {}: Training loss: {}".format(epoch+1, np.round(train_epoch_loss, 4)))
                # output_logits = outputs.logits

                #Evaluation
                model.eval()
                eval_loss = []
                for step, batch in enumerate(eval_loader):
                    with torch.no_grad():
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        ner_labels = batch['ner_labels'].to(device)
                        batch = transformers.BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels})
                        outputs = model(**batch, output_hidden_states=True)
                        mlm_loss = outputs.loss
                        if extra_args.detection_loss:
                            mlm_hidden_states = outputs.hidden_states
                            ner_batch_losses = detection_model(input_ids, mlm_hidden_states, ner_labels, mode='average')
                        loss = mlm_loss + (extra_args.auxilliary_task_decay * torch.mean(torch.stack(ner_batch_losses))) if extra_args.detection_loss else mlm_loss
                    eval_loss.append(float(loss))

                eval_epoch_loss = np.mean(eval_loss)
                logging.info("Epoch {}: Evaluation loss: {}".format(epoch + 1, np.round(eval_epoch_loss, 4)))

                try:
                    perplexity = math.exp(eval_epoch_loss)
                except OverflowError:
                    perplexity = float("inf")

                logging.info("Epoch {}: perplexity: {}".format(epoch+1, perplexity))
                loss_metrics['training'].append(train_epoch_loss)
                loss_metrics['val'].append(eval_epoch_loss)
                l.write('{}\t{}\t{}\n'.format(train_epoch_loss, eval_epoch_loss, perplexity))
            loss_metrics['epochs'] = [i+1 for i in range(int(train_args.num_train_epochs))]
            log_metrics = pd.DataFrame(loss_metrics)
            # sns.lineplot(data=log_metrics, x='epochs', y='training')
            # sns.lineplot(data=log_metrics, x='epochs', y='val')
            # plt.savefig(args.output_dir+'/loss.png')
            l.close()

            if train_args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(train_args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(train_args.output_dir)

def pad_tensor(ts):
    pass

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
                fact.write(txt+'\n')
                for mask in mask_token_index:
                    mask = torch.unsqueeze(mask, 0)
                    mask_token_logits = logits[0, mask, :]
                    top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()
                    top_3_token_predictions.append(top_3_tokens)
                fact.write(str(masked_sequence)+'\n')
                for j,token_ids in enumerate(top_3_token_predictions):
                    masked_sequence = masked_sequence.replace(tokenizer.mask_token, tokenizer.decode([token_ids[0]]), 1)
                    fact.write(str(masked_sequence))
                    fact.write('\n')
                fact.write('\n')
        fact.close()


def fill_evaluation(data, labels, train_args, extra_args, model, tokenizer):
    """
        As the model to fill in the unknown e.g. After patients were given Sorafenib, they reported [BLANK].
        Model should fill in the blank e.g. FATIGUE
        metric: partial_match -  Given 4 outcomes of span length 3, if model doesn't recall all 3 tokens for each span e.g. (1/3 for outcome 1, 2/3 for outcome 2
        1/3 for outcome 4, and 3/3 for outcome 4. accuracy will be determined by an average accuracy computed as (1/3 + 2/3 + 1/3 + 3/3)/4 = 1/2
        metric: exact match - For the same example above, exact match accuracy would be 1/4, because only 1 outcome was fully recalled
    """
    outcomes = json.load(open(extra_args.mention_frequency, 'r'))
    outcomes = {k.split(' [SEP] ')[0].strip():v for k,v in outcomes.items()}
    mem_accuracy = {}
    facts = {}
    prompt_count = 1
    print('\n\n\n+++++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++++\n\n\n')
    for text, labels in zip(data, labels):
        prompt = {}
        exisiting_outcomes_labels = prepare_data.identify_outcome_using_label(seq=text, seq_labels=labels)
        existing_outcomes = [i[0] for i in exisiting_outcomes_labels]
        prompt['text'] = text
        correct_count = 0
        if existing_outcomes:
            for outcome in existing_outcomes:
                prompt['masked_outcome'] = outcome
                outcome_len = len(outcome.split())
                mask = " ".join(tokenizer.mask_token for i in range(len(outcome.split())))
                masked_text = text.replace(outcome, mask.rstrip())

                input = tokenizer.encode(masked_text, return_tensors="pt")
                mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
                logits = model(input).logits
                top_token_predictions = []
                for mask in mask_token_index:
                    mask = torch.unsqueeze(mask, 0)
                    mask_token_logits = logits[0, mask, :]
                    top_tokens = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
                    top_token_predictions.append(top_tokens[0])
                prediction = ' '.join([tokenizer.decode([id]) for id in top_token_predictions])

                for j,token_id in enumerate(top_token_predictions):
                    masked_text = masked_text.replace(tokenizer.mask_token, tokenizer.decode([token_id]), 1)

                outcome, prediction = outcome.lower().strip(), prediction.lower().strip()
                masked_text_len = len(masked_text.split())
                if outcomes[outcome] in mem_accuracy:
                    if extra_args.recall_metric == 'partial_match':
                        print(outcome, prediction)
                        T = [a == p for a, p in zip(outcome.split(), prediction.split())]
                        C = np.count_nonzero(T)
                        mem_accuracy[outcomes[outcome]].append(float(C/len(T)))
                    elif extra_args.recall_metric == 'exact_match':
                        if outcome == prediction or outcome in prediction:
                            mem_accuracy[outcomes[outcome]]['Correct'] +=1
                            prompt[str(outcome_len)+'_'+str(masked_text_len)+'_Correct'+'_'+str(correct_count+1)] = masked_text
                        mem_accuracy[outcomes[outcome]]['Total'] += 1
                else:
                    if extra_args.recall_metric == 'partial_match':
                        print(outcome, prediction)
                        T = [a == p for a, p in zip(outcome.split(), prediction.split())]
                        C = np.count_nonzero(T)
                        mem_accuracy[outcomes[outcome]] = [float(C/len(T))]
                    elif extra_args.recall_metric == 'exact_match':
                        mem_accuracy[outcomes[outcome]] = {'Total':1}
                        if outcome == prediction or outcome in prediction:
                            mem_accuracy[outcomes[outcome]]['Correct'] = 1
                            prompt[str(outcome_len)+'_'+str(masked_text_len)+'_Correct'+'_'+str(correct_count+1)] = masked_text
                        else:
                            mem_accuracy[outcomes[outcome]]['Correct'] = 0
                correct_count += 1
        facts[prompt_count] = prompt
        prompt_count += 1
    print(mem_accuracy)
    eval_dir = prepare_data.create_directory(train_args.output_dir+'/{}'.format(extra_args.recall_metric))
    #store_the memorization accuracy
    with open(eval_dir+'/mem_accuracy.json', 'w') as mem_acc, \
            open(eval_dir+'/fact_predictions.json', 'w') as fc:
        mem_accuracy_ = {}
        for freq in mem_accuracy:
            if extra_args.recall_metric == 'partial_match':
                mem_accuracy_[freq] = np.mean(mem_accuracy[freq])
            elif extra_args.recall_metric == 'exact_match':
                if mem_accuracy[freq]['Correct'] > 0 and mem_accuracy[freq]['Total'] > 0:
                    mem_accuracy_[freq] = float(mem_accuracy[freq]['Correct']/mem_accuracy[freq]['Total'])
        json.dump(mem_accuracy_, mem_acc, indent=2)
        json.dump(facts, fc, indent=2)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = HfArgumentParser((ExtraArguments, TrainingArguments))
    extra_args, train_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(extra_args.pretrained_model, add_prefix_space=True)
    model = AutoModelForMaskedLM.from_pretrained(extra_args.pretrained_model)
    model.to(device)
    if extra_args.alm:
        pretrained_alm = T5ForConditionalGeneration.from_pretrained('t5-small')
        pretrained_alm_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        pretrained_alm.to(device)

    cur_date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not extra_args.fill_evaluation:
        train_args.output_dir= train_args.output_dir + '_' + cur_date_time

    print(train_args, '\n', extra_args)
    print('---------------------------------------Read and Tokenize the data-------------------------------------------')

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    #convert labels of a split(train or dev or test) to ids
    def labels_to_ids(list_of_labels, label_to_id):
        lst = []
        for labels in list_of_labels:
            _labels_ = [label_to_id[i.strip()] for i in labels.split()]
            lst.append(_labels_)
        return lst

    # If passed along, set the training seed now.
    if train_args.seed is not None:
        set_seed(train_args.seed)

    accelerator.wait_for_everyone()
    # load data
    if train_args.do_train:
        tr_data = load_dataset('ebm-comet-data.py', data_files=[extra_args.data + '/train.txt'])
        features = tr_data['train'].features
        column_names = tr_data['train'].column_names
        text_column_name = "tokens" if "tokens" in column_names else column_names[0]
        label_column_name = "ner_tags" if "ner_tags" in column_names else column_names[1]
        labels = tr_data['train'].features[label_column_name].feature.names

    if train_args.do_eval:
        ev_data = load_dataset('ebm-comet-data.py', data_files=[extra_args.data + '/dev.txt'])
        features = ev_data['dev'].features

    # fetch labels in the loaded datasets
    if train_args.do_train or train_args.do_eval:
        if isinstance(features[label_column_name].feature, ClassLabel):
            label_list = features[label_column_name].feature.names
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            data = tr_data["train"] if train_args.do_train else ev_data['dev']
            label_list = get_label_list(data[label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}

    max_seq_length = min(extra_args.max_seq_length, tokenizer.model_max_length)

    # loading a tokenizer for tokenizing the input
    def tokenize(examples):
        print('\n------------------------------------------label\n', label_list, '\n------------------------------------------Label to id\n', label_to_id)
        tokenized_encodings = tokenizer(examples[text_column_name]) if extra_args.pretrained_model.lower() == 'gpt2' \
            else tokenizer(examples[text_column_name],  max_length=max_seq_length,  truncation=True, padding='max_length', is_split_into_words=True, return_special_tokens_mask=True)
        tokenized_encodings['labels'] = tokenized_encodings.input_ids.copy()
        labels = []
        print('Length of tokenized embeddings', len(tokenized_encodings))
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_encodings.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if extra_args.label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_encodings["ner_labels"] = labels
        return tokenized_encodings

    def re_format_tokenized_data(tokenized_input):
        reformulated_tokenized_input = {'input_ids':[],
                                        'token_type_ids': [],
                                        'attention_mask': [],
                                        'labels':[],
                                        'ner_labels':[],
                                        'ner_tags':[],
                                        'tokens':[]}
        for ins in tokenized_input:
            for k,v in ins.items():
                for m,n in reformulated_tokenized_input.items():
                    if k == m:
                        reformulated_tokenized_input[m].append(v)
        return reformulated_tokenized_input

    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def add_marker_tokens(instance):
        """
        Insert marker tokens to the start of a prompt sequence. These tokens indicate what type of prompt/prompt pattern, prefix/postfix/cloze/mixed and null prompt
        """
        k = []
        input_ids = instance['input_ids']
        special_tokens = tokenizer.additional_special_tokens
        special_tokens_ids = tokenizer.additional_special_tokens_ids
        instance_len = len(instance['ner_tags'])

        #check where outcomes are within the prompt
        x = 0
        for u, v in enumerate(instance['ner_tags']):
            if x == u:
                o = []
                if v > 0:
                    for m, n in enumerate(instance['ner_tags'][u:]):
                        if n > 0:
                            o.append((u + m, n))
                            x += 1
                        else:
                            break
                    if o:
                        k.append(list(zip(*o)))
                else:
                    x += 1
        if k:
            if len(k) == 1:
                if k[0][0][0] == 0:
                    #postfix_prompt because the outcome appears at the start of prompt and context follows
                    instance['tokens'].insert(0, str(special_tokens[1]))
                    instance['input_ids'].insert(1, special_tokens_ids[1])
                    instance['labels'].insert(1, special_tokens_ids[1])
                elif k[0][0][-1] == (instance_len - 1):
                    #prefix prompt because the outcome appears at the end of prompt and context before
                    instance['tokens'].insert(0, str(special_tokens[0]))
                    instance['input_ids'].insert(1, special_tokens_ids[0])
                    instance['labels'].insert(1, special_tokens_ids[0])
                else:
                    #cloze prompt because the outcome appears in the middle of prompt and context surrounds it
                    instance['tokens'].insert(0, str(special_tokens[2]))
                    instance['input_ids'].insert(1, special_tokens_ids[2])
                    instance['labels'].insert(1, special_tokens_ids[2])
            else:
                #mixed prompt because there is multiple outcomes to unmask within a prompt
                instance['tokens'].insert(0, str(special_tokens[3]))
                instance['input_ids'].insert(1, special_tokens_ids[3])
                instance['labels'].insert(1, special_tokens_ids[3])
        else:
            #null prompt
            instance['tokens'].insert(0, str(special_tokens[4]))
            instance['input_ids'].insert(1, special_tokens_ids[4])
            instance['labels'].insert(1, special_tokens_ids[4])

        instance['attention_mask'].insert(1, 1) #all tokens to be attended to should be 1 otherwise 0
        if 'token_type_ids' in instance:
            instance['token_type_ids'].insert(1, 0) #first sequence has all it's tokens represented with 0 and second sequence are 1's
        instance['special_tokens_mask'].insert(1, 0)
        instance['ner_labels'].insert(1, 0) #the detection label is O i.e. outside of tag hence id is 0
        return instance

    def prefixing(lst, items_and_ids, special_tokens, special_tokens_ids):
        for i,v in items_and_ids.items():
            if i == 'tokens':
                lst[i].insert(0, special_tokens[v])
            else:
                lst[i].insert(1, special_tokens_ids[v])
        return lst

    def optimization():
        weights = []
        for n, p in model.named_parameters():
            parameter_modified = False
            for q, r in detection_model.named_parameters():
                if q == 'output.weight':
                    for i in range(r.size(0)):
                        if str(n).__contains__(str(i)):
                            p = p + r
                            weights.append(p)
                            parameter_modified = True
            if not parameter_modified:
                weights.append(p)
            break

    #tokenise and prepare dataset for training
    if train_args.do_train:
        print('Here is the train data before tokenization', tr_data)
        with accelerator.main_process_first():
            tokenized_input = tr_data.map(tokenize, batched=True, desc="Running tokenizer on train data")
        # tokenized_input = tokenized_input.map(group_texts, batched=True, desc="Grouping texts in chunks of {}".format(max_seq_length))
        train_tokenized_data = tokenized_input['train']
        train_data = outcome_mask.customMask(train_tokenized_data,
                                             tokenizer=tokenizer,
                                             labels_list=label_list,
                                             mask_id=tokenizer.mask_token_id,
                                             mask=extra_args.custom_mask)

        # expand tokenizer vocabularly
        if extra_args.add_marker_tokens:
            special_tokens = {'additional_special_tokens': ['[prefix]', '[postfix]', '[cloze]', '[mixed]', '[null]']}
            num_added_toks = tokenizer.add_special_tokens(special_tokens)
            model.resize_token_embeddings(len(tokenizer))
            # prompt engineering
            train_data = train_data.map(add_marker_tokens)
            for i, j in enumerate(train_data):
                if i < 1:
                    print(j)
                    print('\n')

        if extra_args.trainer_api:
            print(train_data)
            train_data = train_data.remove_columns(['ner_labels', 'tokens', 'ner_tags', 'special_tokens_mask'])
            train_dataset_dict = transformers.BatchEncoding(Dataset.to_dict(train_data))
            train_data = Outcome_Dataset(train_dataset_dict)

        for id in tokenizer.all_special_ids:
            print(id, PreTrainedTokenizerFast.convert_ids_to_tokens(tokenizer, ids=id))
        # train_data = Outcome_Dataset(train_data)
        print('Here is the train data after tokenization', train_data)

    if train_args.do_eval:
        print('Here is the eval data before tokenization', ev_data)
        tokenized_input = ev_data.map(tokenize, batched=True, desc="Running tokenizer on train data")
        # tokenized_input = tokenized_input.map(group_texts, batched=True, desc="Grouping texts in chunks of {}".format(max_seq_length))
        ev_tokenized_data = tokenized_input['dev']
        print(extra_args.mask_id, type(extra_args.mask_id))
        eval_data = outcome_mask.customMask(ev_tokenized_data,
                                            tokenizer=tokenizer,
                                            labels_list=label_list,
                                            mask_id=tokenizer.mask_token_id,
                                            mask=extra_args.custom_mask)

        # expand tokenizer vocabularly
        if extra_args.add_marker_tokens:
            # prompt engineering
            eval_data = eval_data.map(add_marker_tokens)


        if extra_args.trainer_api:
            eval_data = eval_data.remove_columns(['ner_labels', 'tokens', 'ner_tags', 'special_tokens_mask'])
            eval_dataset_dict = transformers.BatchEncoding(Dataset.to_dict(eval_data))
            eval_data = Outcome_Dataset(eval_dataset_dict)

        print('Here is the eval data after tokenization', train_data)

    #training and evaluation
    if train_args.do_train:
        # defining a complementary Autoregressive language model
        if extra_args.alm:
            alm_model = prompt_model.prompt_model(hdim=extra_args.det_hidden_dim, vocab_size=tokenizer.vocab_size)
            alm_model.to(device)
            alm_args = (pretrained_alm, alm_model)
        else:
            alm_args = None

        # defining a detection model
        if extra_args.detection_loss:
            #add special token to labels based on the tokenization that incorporates expansion of the token ner_labels
            det_criterion = torch.nn.CrossEntropyLoss()
            label_list.append('special_token')
            label_ids = [x for x,y in label_to_id.items()]
            label_to_id[label_ids[-1]+1] = -100
            detection_model = detection.outcome_detection_model(batch_size=extra_args.det_batch_size,
                                                                hdim=extra_args.det_hidden_dim,
                                                                tokenizer=tokenizer,
                                                                det_criterion=det_criterion,
                                                                token_ids=label_to_id)
            detection_model.to(device)
            detection_args = (detection_model, det_criterion, detection_model.parameters())
        else:
            detection_args = None

        train(model=model,
              alm_args=alm_args,
              detection_args=detection_args,
              train_data=train_data,
              eval_data=eval_data if train_args.do_eval else None,
              train_args=train_args,
              extra_args=extra_args,
              tokenizer=tokenizer)

    #fill in masked tokens
    if extra_args.do_fill:
        #data should be a file in which we intend to fill in unknown of a prompt
        eval_model = AutoModelForMaskedLM.from_pretrained(extra_args.pretrained_model)
        eval_data, eval_data_labels = prepare_data.read_outcome_data_to_sentences(extra_args.data + '/dev.txt')
        # ev_data = load_dataset('ebm-comet-data.py', data_files=[extra_args.data + '/dev.txt'])
        evaluate(data=eval_data, labels=eval_data_labels, train_args=train_args, model=eval_model, tokenizer=tokenizer)

    #evlauate filling task
    if extra_args.fill_evaluation:
        model = AutoModelForMaskedLM.from_pretrained(extra_args.pretrained_model)
        eval_data, eval_data_labels = prepare_data.read_outcome_data_to_sentences(extra_args.data)
        data = load_dataset('ebm-comet-data.py', data_files=[extra_args.data])
        tokenizer = AutoTokenizer.from_pretrained(extra_args.pretrained_model)
        fill_evaluation(data=eval_data, labels=eval_data_labels, train_args=train_args, extra_args=extra_args, model=model, tokenizer=tokenizer)

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='data/ebm-comet/', help='source of data')
    par.add_argument('--do_train', action='store_true', help='call if you want to fine-tune pretrained model on dataset')
    par.add_argument('--do_eval', action='store_true', help='call if you want to evaluate a fine-tuned pretrained model on validation dataset')
    par.add_argument('--do_fill', action='store_true', help='Get the model to fill in masked entities in the validation dataset')
    par.add_argument('--fill_evaluation', action='store_true', help='Evaluate how well model recalls outcomes')
    par.add_argument('--output_dir', default='output', help='indicate where you want model and results to be stored')
    par.add_argument('--overwrite_output_dir', action='store_true', help='overwrite existing output directory')
    par.add_argument('--pretrained_model', default='bert-base-uncased', help='pre-trained model available via hugging face e.g. dmis-lab/biobert-v1.1')
    par.add_argument('--num_train_epochs', default=3, help='number of training epochs')
    par.add_argument('--max_seq_length', default=256, help='Maximum length of a sequence')
    par.add_argument('--per_device_train_batch_size', default=16, help='training batch size')
    par.add_argument('--per_device_eval_batch_size', default=16, help='eval batch size')
    par.add_argument('--save_steps', default=1500, help='eval batch size')
    par.add_argument('--mask_id', default=103, help='id for the special mask token 103-Bert, 50264-Roberta, 104-SciBERT')
    par.add_argument('--resume_from_checkpoint', default=None, help='location of most recent model checkpoint')
    par.add_argument('--custom_mask', action='store_true', help='specify tokens to mask and avoid using the data collator')
    par.add_argument('--mention_frequency', default='outcome_occurrence.json', help='File with the outcome mention frequency.')
    par.add_argument('--recall_metric', default='exact_match', help='exact_match or partial_matial')
    par.add_argument('--trainer_api', action='store_true', help='use the trainer api')
    par.add_argument('--detection_loss', action='store_true', help='include an auxiliary cross entropy loss for outcome detection')
    par.add_argument('--det_batch_size', default=64, help='Batch size of the detection model')
    par.add_argument('--det_hidden_dim', default=768, help='Hidden state dimension of the detection model')
    par.add_argument('--auxilliary_task_decay', default=0.001, help='Decay the auxilliary loss during training')
    par.add_argument('--alm', action='store_true', help='amalgamate auto-regressive model hidden states')
    par.add_argument('--add_marker_tokens', action='store_true', help='Trigger the f_prompt  function to insert marker tokens at start of prompt')
    par.add_argument('--evaluation_strategy', default='steps', help='The evaluation strategy to adopt during training')
    par.add_argument('--label_all_tokens', action='store_true', help='opt to label all tokens or not after tokenization')
    par.add_argument('--partial_contexts', action='store_true', help='train mlm with prompts having different contexts, where contexts refers to prompts of different freq occurence')
    args = par.parse_args()
    main(args)
