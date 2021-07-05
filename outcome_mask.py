# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 05/07/2021 
# @Contact: michealabaho265@gmail.com
import prepare_data
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
                        # print([tok for tok in tokens if tok not in tokenizer.all_special_tokens])
                        # print(seq_outcomes)
                        # print([id for id in seq_ids if id not in tokenizer.all_special_ids])
                        for outcome in seq_outcomes:
                            outcome_tok_ids = [tok for tok in tokenizer.encode(outcome) if
                                               tok not in tokenizer.all_special_ids]
                            for tok_id in outcome_tok_ids:
                                indices_to_mask.append(tok_id)
                        seq_ids = [103 if id in indices_to_mask else id for id in seq_ids]
                        # print(indices_to_mask)
                        # print([id for id in seq_ids if id not in [100, 102, 0, 101]])
                        # print('\n')
                    except:
                        raise ValueError('Outcome exists but not identified')
                elif len(seq_outcomes) == 0 and any(k in unique_labels for k in dataset_labels[n].split()):
                    raise ValueError('Outcome exists but not identified')
                inpt_ids.append(seq_ids)
    tokenized_input['input_ids'] = inpt_ids

    return tokenized_input