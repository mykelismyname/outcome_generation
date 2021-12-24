# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 16/12/2021 
# @Contact: michealabaho265@gmail.com
import json
import prepare_data
import torch
import numpy as np
import prompt_model
from dataclasses import field
from typing import Optional
from argparse import ArgumentParser
import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset, ClassLabel
import prepare_data
import outcome_mask
from train import Outcome_Dataset
from torch.utils.data import DataLoader

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def fill_evaluation(data, args, model, tokenizer):
    """
        As the model to fill in the unknown e.g. After patients were given Sorafenib, they reported [BLANK].
        Model should fill in the blank e.g. FATIGUE
        metric: partial_match -  Given 4 outcomes of span length 3, if model doesn't recall all 3 tokens for each span e.g. (1/3 for outcome 1, 2/3 for outcome 2
        1/3 for outcome 4, and 3/3 for outcome 4. accuracy will be determined by an average accuracy computed as (1/3 + 2/3 + 1/3 + 3/3)/4 = 1/2
        metric: exact match - For the same example above, exact match accuracy would be 1/4, because only 1 outcome was fully recalled
    """

    outcomes = json.load(open(args.mention_frequency, 'r'))
    outcomes = {k.split(' [SEP] ')[0].strip():v for k,v in outcomes.items()}
    mem_accuracy = {}
    facts = {}
    prompt_count = 1

    print('\n\n\n+++++++++++++++++++++++++++++++==============================================+++++++++++++++++++++++++++++++\n\n\n')

    for batch in data:
        print(batch)
    #     print(text)
    #     print(labels)
    #     prompt = {}
    #     exisiting_outcomes_labels = prepare_data.identify_outcome_using_label(seq=text, seq_labels=labels)
    #     existing_outcomes = [i[0] for i in exisiting_outcomes_labels]
    #     prompt['text'] = text
    #     correct_count = 0
    #     if existing_outcomes:
    #         for outcome in existing_outcomes:
    #             prompt['masked_outcome'] = outcome
    #             outcome_len = len(outcome.split())
    #             mask = " ".join(tokenizer.mask_token for i in range(len(outcome.split())))
    #             masked_text = text.replace(outcome, mask.rstrip())
    #
    #             input = tokenizer.encode(masked_text, return_tensors="pt")
    #             mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    #             logits = model(input).logits
    #             top_token_predictions = []
    #             for mask in mask_token_index:
    #                 mask = torch.unsqueeze(mask, 0)
    #                 mask_token_logits = logits[0, mask, :]
    #                 top_tokens = torch.topk(mask_token_logits, 1, dim=1).indices[0].tolist()
    #                 top_token_predictions.append(top_tokens[0])
    #             prediction = ' '.join([tokenizer.decode([id]) for id in top_token_predictions])
    #
    #             for j,token_id in enumerate(top_token_predictions):
    #                 masked_text = masked_text.replace(tokenizer.mask_token, tokenizer.decode([token_id]), 1)
    #
    #             outcome, prediction = outcome.lower().strip(), prediction.lower().strip()
    #             masked_text_len = len(masked_text.split())
    #             if outcomes[outcome] in mem_accuracy:
    #                 if extra_args.recall_metric == 'partial_match':
    #                     print(outcome, prediction)
    #                     T = [a == p for a, p in zip(outcome.split(), prediction.split())]
    #                     C = np.count_nonzero(T)
    #                     mem_accuracy[outcomes[outcome]].append(float(C/len(T)))
    #                 elif extra_args.recall_metric == 'exact_match':
    #                     if outcome == prediction or outcome in prediction:
    #                         mem_accuracy[outcomes[outcome]]['Correct'] +=1
    #                         prompt[str(outcome_len)+'_'+str(masked_text_len)+'_Correct'+'_'+str(correct_count+1)] = masked_text
    #                     mem_accuracy[outcomes[outcome]]['Total'] += 1
    #             else:
    #                 if extra_args.recall_metric == 'partial_match':
    #                     print(outcome, prediction)
    #                     T = [a == p for a, p in zip(outcome.split(), prediction.split())]
    #                     C = np.count_nonzero(T)
    #                     mem_accuracy[outcomes[outcome]] = [float(C/len(T))]
    #                 elif extra_args.recall_metric == 'exact_match':
    #                     mem_accuracy[outcomes[outcome]] = {'Total':1}
    #                     if outcome == prediction or outcome in prediction:
    #                         mem_accuracy[outcomes[outcome]]['Correct'] = 1
    #                         prompt[str(outcome_len)+'_'+str(masked_text_len)+'_Correct'+'_'+str(correct_count+1)] = masked_text
    #                     else:
    #                         mem_accuracy[outcomes[outcome]]['Correct'] = 0
    #             correct_count += 1
    #     facts[prompt_count] = prompt
    #     prompt_count += 1
    # print(mem_accuracy)
    # eval_dir = prepare_data.create_directory(train_args.output_dir+'/{}'.format(extra_args.recall_metric))
    # #store_the memorization accuracy
    # with open(eval_dir+'/mem_accuracy.json', 'w') as mem_acc, \
    #         open(eval_dir+'/fact_predictions.json', 'w') as fc:
    #     mem_accuracy_ = {}
    #     for freq in mem_accuracy:
    #         if extra_args.recall_metric == 'partial_match':
    #             mem_accuracy_[freq] = np.mean(mem_accuracy[freq])
    #         elif extra_args.recall_metric == 'exact_match':
    #             if mem_accuracy[freq]['Correct'] > 0 and mem_accuracy[freq]['Total'] > 0:
    #                 mem_accuracy_[freq] = float(mem_accuracy[freq]['Correct']/mem_accuracy[freq]['Total'])
    #     json.dump(mem_accuracy_, mem_acc, indent=2)
    #     json.dump(facts, fc, indent=2)

def main(args):
    p_model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.saved_model+'/tokenizer/')
    data = load_dataset('ebm-comet-data.py', data_files=[args.data])
    file = 'dev' if args.data.__contains__('dev.txt') else 'train' if args.data.__contains__('train.txt') else args.data
    column_names = data[file].column_names
    features = data[file].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = "ner_tags" if "ner_tags" in column_names else column_names[1]
    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
    label_list.append('special_token')
    label_ids = [x for x, y in label_to_id.items()]
    label_to_id[label_ids[-1] + 1] = -100

    # loading a tokenizer for tokenizing the input
    def tokenize(examples):
        print('\n------------------------------------------label\n', label_list,
              '\n------------------------------------------Label to id\n', label_to_id)
        tokenized_encodings = tokenizer(examples[text_column_name]) if args.pretrained_model.lower() == 'gpt2' \
            else tokenizer(examples[text_column_name], max_length=args.max_seq_length, truncation=True, padding='max_length',
                           is_split_into_words=True, return_special_tokens_mask=True)
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
                    label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_encodings["ner_labels"] = labels
        return tokenized_encodings

    def add_marker_tokens(instance):
        """
        Insert marker tokens to the start of a prompt sequence. These tokens indicate what type of prompt/prompt pattern, prefix/postfix/cloze/mixed and null prompt
        """
        k = []
        input_ids = instance['input_ids']
        # special_tokens = tokenizer.additional_special_tokens
        # special_tokens_ids = tokenizer.additional_special_tokens_ids
        special_tokens = ['[prefix]', '[postfix]', '[cloze]', '[mixed]', '[null]']
        special_tokens_ids = [tokenizer.vocab_size+i for i in range(1, len(special_tokens)+1)]
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
                    # instance['positions'] = [len(instance['input_ids'])]
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

    tokenized_input = data.map(tokenize, batched=True, desc="Running tokenizer on eval data")
    print(tokenized_input)
    tokenized_data = tokenized_input[file]
    if args.custom_mask:
        eval_data = outcome_mask.customMask(tokenized_data,
                                            tokenizer=tokenizer,
                                            labels_list=label_list,
                                            mask_id=tokenizer.mask_token_id,
                                            mask=args.custom_mask)
    for i in eval_data:
        print(len(i['input_ids']))
        break
    # # expand tokenizer vocabularly
    # if args.add_marker_tokens:
    #     special_tokens = ['[prefix]', '[postfix]', '[cloze]', '[mixed]', '[null]']
    #     special_ids = [tokenizer.vocab_size + i for i in range(1, len(special_tokens) + 1)]
    #     special_token_ids = dict(zip(special_tokens, special_ids))
    #     eval_data = eval_data.map(add_marker_tokens)
    #
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm_probability=0.15,
    #     pad_to_multiple_of=None
    # )
    #
    # eval_data = eval_data.remove_columns(['tokens', 'ner_tags'])
    # if args.custom_mask:
    #     eval_dataset_dict = transformers.BatchEncoding(Dataset.to_dict(eval_data))
    #     eval_data = Outcome_Dataset(eval_dataset_dict)
    #     eval_loader = DataLoader(eval_data, batch_size=args.per_device_eval_batch_size)
    # else:
    #     eval_loader = DataLoader(eval_data, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator)
    #
    # mlm_model = prompt_model.prompt_model(model=p_model,
    #                                       special_token_ids=special_token_ids if args.add_marker_tokens else None,
    #                                       hdim=args.hidden_dim,
    #                                       tokenizer=tokenizer,
    #                                       seq_length=args.max_seq_length,
    #                                       add_marker_tokens=args.add_marker_tokens,
    #                                       marker_token_emb_size=args.marker_token_emb_size,
    #                                       ner_label_ids=label_to_ids,
    #                                       detection=args.detection_loss,
    #                                       prompt_conditioning=args.prompt_conditioning).to(device)
    #
    # mlm_model.load(args.saved_model + '/checkpoint_epoch_1.pt')
    # # eval_data, eval_data_labels = prepare_data.extract_examples_and_labels(data[i])
    # fill_evaluation(data=eval_loader, args=args, model=mlm_model, tokenizer=tokenizer)

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='data/ebm-comet/', help='source of data')
    par.add_argument('--fill_evaluation', action='store_true', help='Evaluate how well model recalls outcomes')
    par.add_argument('--output_dir', default='output', help='indicate where you want model and results to be stored')
    par.add_argument('--pretrained_model', default='bert-base-uncased', help='pre-trained model available via hugging face e.g. dmis-lab/biobert-v1.1')
    par.add_argument('--max_seq_length', default=256, help='Maximum length of a sequence')
    par.add_argument('--per_device_eval_batch_size', default=16, help='eval batch size')
    par.add_argument('--saved_model', required=True, help='Model fine-tuned on the prompts')
    par.add_argument('--mask_id', default=103, help='id for the special mask token 103-Bert, 50264-Roberta, 104-SciBERT')
    par.add_argument('--custom_mask', action='store_true', help='specify tokens to mask and avoid using the data collator')
    par.add_argument('--mention_frequency', default='outcome_occurrence.json', help='File with the outcome mention frequency.')
    par.add_argument('--recall_metric', default='exact_match', help='exact_match or partial_matial')
    par.add_argument('--detection_loss', action='store_true', help='include an auxiliary cross entropy loss for outcome detection')
    par.add_argument('--layers', default='average', help='either select a average of the hidden states across all model layers or select last layer hidden states')
    par.add_argument('--det_batch_size', default=64, help='Batch size of the detection model')
    par.add_argument('--hidden_dim', default=768, help='Hidden state dimension of the mlm model')
    par.add_argument('--add_marker_tokens', action='store_true', help='Trigger the f_prompt  function to insert marker tokens at start of prompt')
    par.add_argument('--marker_token_emb_size', default=50, help='Batch size of the detection model')
    par.add_argument('--prompt_conditioning', default=0, type=int, help='Trigger an order based attention to create a distributin')
    par.add_argument('--evaluation_strategy', default='steps', help='The evaluation strategy to adopt during training')
    par.add_argument('--label_all_tokens', action='store_true', help='opt to label all tokens or not after tokenization')
    par.add_argument('--partial_contexts', action='store_true', help='train mlm with prompts having different contexts, where contexts refers to prompts of different freq occurence')
    args = par.parse_args()
    main(args)
