# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 05/07/2021 
# @Contact: michealabaho265@gmail.com
import prepare_data
import re
from transformers import PreTrainedTokenizerFast

'''
    Masking specific tokens of the input dataset
'''
def custom_mask(tokenizer, tokenized_input, dataset, dataset_labels):
    v = 0
    print(tokenizer.all_special_tokens, tokenizer.all_special_ids)
    #get a list of unique labels in the dataset
    unique_labels = list(set([i for j in dataset_labels for i in j.split()]))
    unique_labels.remove('O')

    inpt_ids = []
    for i, j in tokenized_input.items():
        if i == 'input_ids':
            for n in range(len(j)):
                seq_ids, indices_to_mask, masked_seq_ids = j[n], [], []
                tokens = PreTrainedTokenizerFast.convert_ids_to_tokens(tokenizer, ids=seq_ids)
                # extract outcomes from sequence given they have been labelled
                seq_outcomes = prepare_data.identify_outcome_using_label(seq=dataset[n], seq_labels=dataset_labels[n])
                if len(seq_outcomes) >= 1:
                    try:
                        for outcome in seq_outcomes:
                            outcome_tok_ids = [tok for tok in tokenizer.encode(outcome) if
                                               tok not in tokenizer.all_special_ids]
                            indices_to_mask.append(outcome_tok_ids)
                            seq_ids = replace_id_with_mask_id(outcome_tok_ids, 103, seq_ids, expand=True)
                    except:
                        raise ValueError('Outcome exists but not identified')
                elif len(seq_outcomes) == 0 and any(k in unique_labels for k in dataset_labels[n].split()):
                    raise ValueError('Outcome exists but not identified')
                inpt_ids.append(seq_ids)
    tokenized_input['input_ids'] = inpt_ids

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