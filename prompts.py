import os
import prepare_data
import json
import re
from argparse import ArgumentParser
import sys
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

def prompts(data, outcome_occurrence, count):
    prompts, prompts_labels = prepare_data.read_outcome_data_to_sentences(data)
    outcomes_occurrence = json.load(open(outcome_occurrence))
    outcome_occurrence = {}
    for k,v in outcomes_occurrence.items():
        outcome, outcome_type = k.split('[SEP]')
        outcome_occurrence[outcome.strip()] = {'frequency':int(v), 'type':outcome_type.strip()}

    popular = False
    for text, labels in zip(prompts, prompts_labels):
        text_outcomes = [i.strip() for i in prepare_data.identify_outcome_using_label(seq=text, seq_labels=labels)]
        if text_outcomes:
            for i in outcome_occurrence.keys():
                if str(i) in text_outcomes:
                    popular = True
                    break
            if popular:
                if len(text.split()) < 15:
                    print(text, text_outcomes)

def prompt(args):
    model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    while True:
        prompt = input("Submit a propmt leaving out a blank:\n")
        if prompt == "":
            break
        try:
            prompt_tokens = re.split('\s+', prompt)
            mask_to_replace = [i for i in prompt_tokens if all(j=='#' for j in i)]
            print('Masked tokens:', mask_to_replace)
            for m in mask_to_replace:
                mask = tokenizer.mask_token
                if len(m) > 1:
                    mask = " ".join(tokenizer.mask_token for i in range(len(m)))
                masked_text = prompt.replace(m, mask.rstrip(), 1)
                print('Masked text:-', masked_text)
                _input_ = tokenizer.encode(masked_text, return_tensors="pt")
                mask_token_index = torch.where(_input_ == tokenizer.mask_token_id)[1]
                logits = model(_input_).logits
                top_token_predictions, all_predictions = [], []
                for msk in mask_token_index:
                    msk = torch.unsqueeze(msk, 0)
                    mask_token_logits = logits[0, msk, :]
                    # print(torch.topk(mask_token_logits, 5, dim=1))
                    top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
                    top_token_predictions.append(top_tokens[0])
                    all_predictions.append(top_tokens)
                # prediction = ' '.join([tokenizer.decode([id]) for id in top_token_predictions])

                for i,p in enumerate(list(zip(*all_predictions))):
                    # prediction = '\033[1m' + ' '.join([tokenizer.decode([id]) for id in p]) + '\033[0m'
                    prediction = ' '.join([tokenizer.decode([id]) for id in p])
                    if i == 0:
                        un_masked_text = masked_text.replace(mask, prediction, 1)
                        print(un_masked_text, '\nOther Predictions')
                    else:
                        print('Prediction {}:- {}'.format(i+1, prediction))

                prompt = un_masked_text
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_obj, exc_tb.tb_lineno)
            raise ValueError('Something is wrong {}'.format(e))



if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--pretrained_model', default='bert-base-cased', help='source of pretrained model')
    par.add_argument('--data',  help='source of data')
    par.add_argument('--mention_frequency', default='outcome_occurrence.json', help='File with the outcome mention frequency.')
    args = par.parse_args()
    # prompts(args.data, args.mention_frequency, 50)
    prompt(args)