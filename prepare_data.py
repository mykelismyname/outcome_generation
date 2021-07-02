# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 25/06/2021 
# @Contact: michealabaho265@gmail.com
import numpy as np
import os
import re
from argparse import ArgumentParser
from collections import Counter
import json

#fetch the data
def fetch_data(files):
    dataset = []
    dataset_labels = []
    train_samples_len, eval_samples_len = 0, 0
    for file in files:
        data, data_labels = read_outcome_data_to_sentences(args.data+'/'+file)
        if file.__contains__('train'):
            train_samples_len = len(data)
        if file.__contains__('dev'):
            eval_samples_len = len(data)
        dataset = dataset + data
        dataset_labels = dataset_labels + data_labels
    return dataset, dataset_labels, train_samples_len, eval_samples_len

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
    # print(list(zip(text[:5], labels[:5])))
    return (text, labels)

def study_context_patterns(data, labels):
    outcomes, types = [], []
    unique_labels = list(set([i for j in labels for i in j.split()]))
    unique_labels.remove('O')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for x,y in zip(data, labels):
        if any(i in unique_labels for i in y.split()):
            # print(x,y)
            x_split, y_split = x.split(), y.split()
            if len(x_split) == len(y_split):
                e = 0
                sentence_outcomes = []
                for m in range(len(x_split)):
                    if m == e:
                        outcome = ''
                        if y_split[m].strip().startswith('B-'):
                            outcome += x_split[m].strip()
                            e += 1
                            for w in range(m+1, len(x_split)):
                                if y_split[w].startswith('I-'):
                                    outcome += ' '+x_split[w].strip()
                                    e += 1
                                else:
                                    break
                            sentence_outcomes.append(outcome)
                        else:
                            e += 1
                if sentence_outcomes:
                    for o in sentence_outcomes:
                        outcomes.append(o)
            else:
                raise ValueError('There is either more or fewer labels than the tokens present in sentence')
        else:
            pass
    outcome_occurence = Counter(outcomes)
    outcome_occurence = dict(sorted(outcome_occurence.items(), key = lambda v:v[1], reverse=True))
    with open(args.data+'/outcome_occurrence.json', 'w') as oc:
        json.dump(outcome_occurence, oc, indent=2)
    print(outcome_occurence)


def main(args):
    if args.correct_dataset:
        correct_dataset(args.data)
    if args.read_outcome_data:
        read_outcome_data_to_sentences(args.data)
    if args.study_context_patterns:
        #specify the path to the directory with train.txt and dev.txt files i.e. via args.data
        dataset, dataset_labels, train_samples_len, eval_samples_len = fetch_data(['train.txt', 'dev.txt'])
        study_context_patterns(data=dataset, labels=dataset_labels)

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='data/ebm_comet_multilabels.txt', help='source of data')
    par.add_argument('--correct_dataset', action='store_true', help='move trailing punctuation characters to a new line in the dataset')
    par.add_argument('--read_outcome_data', action='store_true', help='reading train, test and validation files independently')
    par.add_argument('--study_context_patterns', action='store_true', help='check most frequent outcome and the context around it')
    args = par.parse_args()
    main(args)
