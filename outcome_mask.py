# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 05/07/2021 
# @Contact: michealabaho265@gmail.com
import prepare_data
import re
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from datasets import Dataset

'''
    Masking specific tokens of the input dataset
'''
def oldcustomMask(tokenizer, tokenized_input, dataset, dataset_labels, model):
    v = 0
    print(tokenizer.all_special_tokens, tokenizer.all_special_ids)
    #get a list of unique labels in the dataset
    unique_labels = list(set([i for j in dataset_labels for i in j.split()]))
    unique_labels.remove('O')

    inpt_ids = []
    for i, j in tokenized_input.items():
        print('here', i)
        if i == 'input_ids':
            for n in range(len(j)):
                seq_ids, indices_to_mask, masked_seq_ids, token_labels = j[n], [], [], dataset_labels[n]
                tokens = GPT2TokenizerFast.convert_ids_to_tokens(tokenizer, ids=seq_ids) if model.lower() == 'gpt2' else \
                    PreTrainedTokenizerFast.convert_ids_to_tokens(tokenizer, ids=seq_ids)

                # extract outcomes from sequence given they have been labelled
                seq_outcomes_labels = prepare_data.identify_outcome_using_label(seq=dataset[n], seq_labels=dataset_labels[n])
                seq_outcomes = [i[0] for i in seq_outcomes_labels]
                if len(seq_outcomes) >= 1:
                    try:
                        p = [i for i in tokenizer.all_special_ids if i != 103]
                        for outcome in seq_outcomes:
                            outcome_tok_ids = [tok for tok in tokenizer.encode(outcome) if
                                               tok not in tokenizer.all_special_ids]
                            indices_to_mask.append(outcome_tok_ids)
                            seq_ids = replace_id_with_mask_id(outcome_tok_ids, 103, seq_ids, expand=True)
                    except:
                        raise ValueError('Outcome exists but not identified')
                elif len(seq_outcomes) == 0 and any(k in unique_labels for k in dataset_labels[n].split()):
                    print(dataset[n], dataset_labels[n])
                    raise ValueError('Outcome exists but not identified')
                # print(len(seq_ids), len(token_labels))
                inpt_ids.append(seq_ids)
    tokenized_input['input_ids'] = inpt_ids
    return tokenized_input

def customMask(tokenized_input, tokenizer, labels_list, mask_id, mask=False):
    custom_mask = mask
    if custom_mask:
        print('\n\n---------------CUSTOM MASKING-------------------------\n\n')
        unique_labels_list = labels_list.copy()
        unique_labels_list.remove('O')
        input_ids = []
        for i, j in enumerate(tokenized_input):
            seq_ids, indices_to_mask, masked_seq_ids, = j['input_ids'], [], []
            tokens, token_labels = j['tokens'], [labels_list[i] for i in j['ner_tags']]
            initial_seq_ids_lens = len(seq_ids)
            assert len(tokens) == len(token_labels), "How comes length of tokens doesn't match langth of ner tags"
            seq_outcomes_labels = prepare_data.identify_outcome_using_label(seq=' '.join(tokens), seq_labels=' '.join(token_labels))
            seq_outcomes = [i[0] for i in seq_outcomes_labels]
            if len(seq_outcomes) >= 1:
                try:
                    for outcome in seq_outcomes:
                        outcome_tok_ids = [tok for tok in tokenizer.encode(outcome) if
                                           tok not in tokenizer.all_special_ids]
                        indices_to_mask.append(outcome_tok_ids)
                        seq_ids = replace_id_with_mask_id(outcome_tok_ids, mask_id, seq_ids, expand=True)
                except:
                    raise ValueError('Outcome exists but not identified')
            elif len(seq_outcomes) == 0 and any(k in unique_labels_list for k in token_labels):
                print(tokens, '\n', token_labels, '\n', seq_outcomes)
                raise ValueError('Outcome exists but not identified')
            final_seq_ids_lens = len(seq_ids)
            assert initial_seq_ids_lens == final_seq_ids_lens == len(j['ner_labels']) \
                   == len(j['attention_mask']) == len(j['labels']), \
                   "\n-----------------SOMETHING IS WRONG, CHECK OUT THE LENGTH OF THE TENSORS---------------\n"
            input_ids.append(seq_ids)
        tokenized_input_ = tokenized_input.remove_columns('input_ids')
        tokenized_input_ = tokenized_input_.add_column(name='input_ids', column=input_ids)

        n = 0
        print(type(tokenized_input_))
        for i, j in zip(tokenized_input, tokenized_input_):
            if n < 1:
                print([PreTrainedTokenizerFast.convert_ids_to_tokens(tokenizer, ids=[i for i in j['input_ids'] if i != tokenizer.pad_token_id])], len(j['input_ids']))
            n += 1
        return tokenized_input_
    else:
        for n,j in enumerate(tokenized_input):
            if n < 1:
                print([PreTrainedTokenizerFast.convert_ids_to_tokens(tokenizer, ids=[i for i in j['input_ids'] if i != tokenizer.pad_token_id])], len(j['input_ids']))
            n += 1
        print(type(tokenized_input))
        return tokenized_input

'''
    replace a sequence of id's representing an outcome with a mask id or a replacement value within id's of an input sequence 
'''
def replace_id_with_mask_id(sequence, replacement, lst, expand=False):
    new_list = lst.copy()
    for i, e in enumerate(lst):
        if e == sequence[0]:
            end = i
            f = 1
            for e1, e2 in zip(sequence, lst[i:]):
                if e1 != e2:
                    f = 0
                    break
                end += 1
            if f == 1:
                del new_list[i:end]
                if expand:
                    for _ in range(len(sequence)):
                        new_list.insert(i, replacement)
                else:
                    new_list.insert(i, replacement)
    return new_list