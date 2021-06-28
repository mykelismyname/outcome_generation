# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 25/06/2021 
# @Contact: michealabaho265@gmail.com
import numpy as np
import os
import re
from argparse import ArgumentParser

#Pushing trailing punctuation characters at the end of tokens to next line
def correct_dataset(file):
    dummy = os.path.dirname(file)+'/'+os.path.basename(file).split('.')[0]
    with open(file, 'r') as f, open(dummy+'.bak', 'w') as d:
        for i in f:
            if i != '\n':
                i = i.strip().split()
                if len(i[0]) > 1 and i[0][-1] in ['.', ',', '%', ':', ';', '#', '-']:
                    new_entries = [i[0][:-1], i[0][-1]]
                    for n,m in enumerate(new_entries):
                        if n == 1:
                            if re.search('B\-', i[1]):
                                print(i)
                                d.write(new_entries[n]+' '+'I-'+i[1][2:])
                            else:
                                d.write(new_entries[n] + ' ' + i[1])
                        else:
                            d.write(new_entries[n]+' '+i[1])
                        d.write('\n')
                else:
                    d.write(i[0]+' '+i[1])
                    d.write('\n')
            else:
                d.write('\n')

#creating sentences from each of the files train, test and validation
def read_outcome_data_to_sentences(path):
    text, labels = [], []
    with open(path) as f:
        sentence, token_labels = [], []
        for i in f.readlines():
            if i != '\n':
                token, label = i.strip().split()
                sentence.append(token)
                token_labels.append(label)
            else:
                text.append(' '.join([i for i in sentence]))
                labels.append((' '.join([i for i in token_labels])))
                sentence.clear()
                token_labels.clear()
    print(list(zip(text[:5], labels[:5])))
    return (text, labels)

def main(args):
    if args.correct_dataset:
        correct_dataset(args.data)
    if args.read_outcome_data:
        read_outcome_data_to_sentences(args.data)

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='data/ebm_comet_multilabels.txt', help='source of data')
    par.add_argument('--correct_dataset', action='store_true', help='reading train, test and validation files independently')
    par.add_argument('--read_outcome_data', action='store_true', help='reading train, test and validation files independently')
    args = par.parse_args()
    main(args)
