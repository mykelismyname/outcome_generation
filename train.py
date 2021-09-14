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
    PreTrainedTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    GPT2Tokenizer, GPT2TokenizerFast, GPT2Model,
    AdamW,
)
from torch.utils.data import DataLoader
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from argparse import ArgumentParser
from tqdm import tqdm
import detection_model as detection
logger = logging.getLogger(__name__)

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
    label_all_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "opt to label all tokens or not after tokenization"},
    )
    partial_contexts: Optional[bool] = field(
        default=False,
        metadata={"help":"train mlm with prompts having different contexts, where contexts refers to prompts of different freq occurence"},
    )

def train(model, train_data, eval_data, train_args, extra_args, tokenizer):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )
    #use the hugging face trainer API
    if extra_args.trainer_api:
        print('\nhereherehere\n')
        for i, j in enumerate(train_data):
            if i < 1:
                print(j)
                print(type(j))
                for m,n in j.items():
                    print(m,len(n))
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=data_collator if not extra_args.custom_mask else None,
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
        for i in train_data:
            print(i)
            print(type(i))
            print(train_args.per_device_train_batch_size)
            for m,n in i.items():
                print(m, type(n))
            break
        train_loader = DataLoader(train_data, batch_size=train_args.per_device_train_batch_size, collate_fn=data_collator if not extra_args.custom_mask else None)
        eval_loader = DataLoader(eval_data, batch_size=train_args.per_device_eval_batch_size, collate_fn=data_collator if not extra_args.custom_mask else None)
        optim = AdamW(optimizer_grouped_parameters, lr=train_args.learning_rate)

        for t in train_loader:
            print(t)
            break
        for epoch in range(int(train_args.num_train_epochs)):
            # training_loop = tqdm(train_loader, leave=True)
            model.train()
            for step, batch in enumerate(train_loader):
                optim.zero_grad()
                print(type(batch))
                for i in batch:
                    print(i)
                    print('herhererher', batch[i].shape, type(batch[i]))
                    print(batch[i][:4, :5])
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                batch = transformers.BatchEncoding({'input_ids':input_ids, 'labels':labels, 'attention_mask':attention_mask})
                print(input_ids.shape, attention_mask.shape, labels.shape)
                outputs = model(**batch)
                #loss
                loss = outputs.loss
                loss = loss / train_args.gradient_accumulation_steps
                logger.info(f"step {step}: loss: {loss}")
                print('loss:', loss)
                loss.backward()
                optim.step()
                #output
                output_logits = outputs.logits
                # print(len(input_ids))
                # print(output_logits.shape)
                # for i,j in zip(input_ids, output_logits):
                #     tokens = PreTrainedTokenizerFast.convert_ids_to_tokens(tokenizer, ids=i)
                #     print(tokens)
                #     print(tokenizer.convert_tokens_to_string(tokens))
                break

                # if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                #     optimizer.step()
                #     lr_scheduler.step()
                #     optimizer.zero_grad()
                #     progress_bar.update(1)
                #     completed_steps += 1
                #
                # if completed_steps >= args.max_train_steps:
                #     break
            model.eval()
            losses = []
            for step, batch in enumerate(eval_loader):
                with torch.no_grad():
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    token_type_ids = batch['token_type_ids'].to(device)
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    labels=labels)
                loss = outputs.loss
                losses.append(loss)
                # losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

            losses = torch.cat(losses)
            losses = losses[: len(eval_data)]
            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")
            print()
            logger.info(f"epoch {epoch}: perplexity: {perplexity}")

        # if args.output_dir is not None:
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


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
    tokenizer = GPT2TokenizerFast.from_pretrained(extra_args.pretrained_model) if extra_args.pretrained_model.lower() == 'gpt2' else \
                AutoTokenizer.from_pretrained(extra_args.pretrained_model)
    model = GPT2Model.from_pretrained(extra_args.pretrained_model) if extra_args.pretrained_model.lower() == 'gpt2' else \
                AutoModelForMaskedLM.from_pretrained(extra_args.pretrained_model)
    detection_model = detection.outcome_detection_model()
    model.to(device)

    print(train_args, '\n', extra_args)
    print('---------------------------------------Read and Tokenize the data-------------------------------------------')

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = get_label_list(extra_args.data)
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    #convert labels of a split(train or dev or test) to ids
    def labels_to_ids(list_of_labels, label_to_id):
        lst = []
        for labels in list_of_labels:
            _labels_ = [label_to_id[i.strip()] for i in labels.split()]
            lst.append(_labels_)
        return lst

    # load data
    if train_args.do_train:
        tr_data = load_dataset('ebm-comet-data.py', data_files=[extra_args.data + '/train.txt'])
        print(tr_data)
        text_column_name, label_column_name = ['tokens', 'ner_tags']
        print(text_column_name, label_column_name)
        features = tr_data['train'].features
    if train_args.do_eval:
        ev_data = load_dataset('ebm-comet-data.py', data_files=[extra_args.data + '/dev.txt'])
        print(ev_data)
        text_column_name, label_column_name = ['tokens', 'ner_tags']
        features = ev_data['dev'].features

    # fetch labels in the loaded datasets
    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(tr_data["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    # loading a tokenizer for tokenizing the input
    def tokenize(examples):
        tokenized_encodings = tokenizer(examples[text_column_name]) if extra_args.pretrained_model.lower() == 'gpt2' \
            else tokenizer(examples[text_column_name],  max_length=512,  truncation=True, padding=True, is_split_into_words=True)
        tokenized_encodings['labels'] = tokenized_encodings.input_ids.copy()
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_encodings.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
                if word_idx is None:
                    # print('if 1', word_idx)
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    # print('else 1', word_idx, label, label[word_idx])
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on the label_all_tokens flag.
                else:
                    # print('else 2', word_idx, label, label[word_idx])
                    label_ids.append(label_to_id[label[word_idx]] if extra_args.label_all_tokens else -100)
                previous_word_idx = word_idx
            if i < 4:
                print(i, len(examples[text_column_name][i]), examples[text_column_name][i])
                print(i, len(word_ids), word_ids)
                print(i, len(label), label)
                print(i, len(label_ids), label_ids)
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

    #tokenise and prepare dataset for training
    if train_args.do_train:
        tokenized_input = tr_data.map(tokenize, batched=True, desc="Running tokenizer on train data")
        train_tokenized_data = tokenized_input['train']
        train_data = outcome_mask.customMask(train_tokenized_data,
                                             tokenizer=tokenizer,
                                             labels_list=label_list,
                                             mask=extra_args.custom_mask)
        if extra_args.trainer_api:
            train_data = train_data.remove_columns(['ner_labels', 'tokens', 'ner_tags'])
        else:
            train_data = train_data.remove_columns(['tokens', 'ner_tags'])
        # train_data = Outcome_Dataset(train_data)
        print(train_data)
        print(type(train_data))

    if train_args.do_eval:
        eval_dataset, eval_dataset_labels = prepare_data.fetch_data(extra_args, files=[extra_args.data + '/dev.txt'])
        eval_tokenized_input = tokenize(eval_dataset, tokenizer, extra_args.pretrained_model)
        if extra_args.custom_mask:
            eval_tokenized_input = outcome_mask.custom_mask(tokenizer=tokenizer, tokenized_input=eval_tokenized_input, dataset=eval_dataset, dataset_labels=eval_dataset_labels, model=extra_args.pretrained_model)
        eval_data = Outcome_Dataset(eval_tokenized_input)

    #training and evaluation
    if train_args.do_train:
        train(model=model, train_data=train_data, eval_data=eval_data if train_args.do_eval else None, train_args=train_args, extra_args=extra_args, tokenizer=tokenizer)

    #fill in masked tokens
    if extra_args.do_fill:
        #data should be a file in which we intend to fill in unknown of a prompt
        eval_model = AutoModelForMaskedLM.from_pretrained(train_args.output_dir)
        eval_data, eval_data_labels = prepare_data.read_outcome_data_to_sentences(extra_args.data+'dev.txt')
        evaluate(data=eval_data, labels=eval_data_labels, train_args=train_args, model=eval_model, tokenizer=tokenizer)

    #evlauate filling task
    if extra_args.fill_evaluation:
        model = AutoModelForMaskedLM.from_pretrained(extra_args.pretrained_model)
        eval_data, eval_data_labels = prepare_data.read_outcome_data_to_sentences(extra_args.data)
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
    par.add_argument('--pretrained_model', default='dmis-lab/biobert-v1.1', help='pre-trained model available via hugging face e.g. dmis-lab/biobert-v1.1')
    par.add_argument('--num_train_epochs', default=3, help='number of training epochs')
    par.add_argument('--per_device_train_batch_size', default=16, help='training batch size')
    par.add_argument('--per_device_eval_batch_size', default=16, help='eval batch size')
    par.add_argument('--save_steps', default=1500, help='eval batch size')
    par.add_argument('--resume_from_checkpoint', default=None, help='location of most recent model checkpoint')
    par.add_argument('--custom_mask', action='store_true', help='specify tokens to mask and avoid using the data collator')
    par.add_argument('--mention_frequency', default='outcome_occurrence.json', help='File with the outcome mention frequency.')
    par.add_argument('--recall_metric', default='exact_match', help='exact_match or partial_matial')
    par.add_argument('--trainer_api', action='store_true', help='use the trainer api')
    par.add_argument('--label_all_tokens', action='store_true', help='opt to label all tokens or not after tokenization')
    par.add_argument('--partial_contexts', action='store_true', help='train mlm with prompts having different contexts, where contexts refers to prompts of different freq occurence')
    args = par.parse_args()
    main(args)