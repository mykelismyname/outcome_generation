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
from glob import glob

#fetch the data
def fetch_data(args, files):
    """
           Fecth data and create lists of train sentences and the list of sentence labels (label per token in sentence)
           Partial_contexts - reduces the size of training data by picking only sentences whose outcomes have a particular frequnecu,
           The top 25% most frequent, top half most (50%) frequent and least half (25%) most frequent
    """
    dataset, dataset_labels = [], []
    train_samples_len, eval_samples_len = 0, 0
    for file in files:
        data, data_labels = read_outcome_data_to_sentences(file)
        if args.partial_contexts:
            data_copy, data_labels_copy, = data.copy(), data_labels.copy()
            data, data_labels = [], []
            files_dir = os.path.dirname(file)
            outcomes_occurrence = json.load(open(files_dir+'/outcome_occurrence.json'))
            outcomes_occurrence = {k.split(' [SEP] ')[0].strip().lower():v for k,v in outcomes_occurrence.items()}
            outcomes_occurrence_list = set(list(outcomes_occurrence.values()))
            for text, labels in zip(data_copy, data_labels_copy):
                found_outcomes_labels = identify_outcome_using_label(seq=text, seq_labels=labels)
                found_outcomes = [i[0] for i in found_outcomes_labels]
                if not found_outcomes:
                    data.append(text)
                    data_labels.append(labels)
                else:
                    for o_come in found_outcomes:
                        if outcomes_occurrence[o_come.lower()] in range(0, int(np.percentile(list(outcomes_occurrence_list), 75))):
                            data.append(text)
                            data_labels.append(labels)
                            break
        if file.__contains__('train'):
            train_samples_len = len(data)
        if file.__contains__('dev'):
            eval_samples_len = len(data)
        if file.__contains__('test'):
            test_samples_len = len(data)
        dataset = dataset + data
        dataset_labels = dataset_labels + data_labels

    if len(files) > 1:
        print(len(dataset), train_samples_len, eval_samples_len)
        return dataset, dataset_labels, train_samples_len, eval_samples_len, test_samples_len
    else:
        print(len(dataset), train_samples_len, eval_samples_len)
        return dataset, dataset_labels

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
                sentence = []
                token_labels = []
    # print(list(zip(text[:5], labels[:5])))
    return (text, labels)

#count how frequent each outcome occurs in the dataset
def outcome_frequency(data, labels):
    outcomes, types = [], []
    unique_labels = list(set([i for j in labels for i in j.split()]))
    unique_labels.remove('O')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for x,y in zip(data, labels):
        if any(i in unique_labels for i in y.split()):
            x_split, y_split = x.split(), y.split()
            if len(x_split) == len(y_split):
                e = 0
                sentence_outcomes = []
                for m in range(len(x_split)):
                    if m == e:
                        outcome = ''
                        if y_split[m].strip().startswith('B-'):
                            outcome += x_split[m].lower().strip()
                            e += 1
                            outcome_type = y_split[m].strip().split('-', 1)[-1].strip()
                            for w in range(m+1, len(x_split)):
                                if y_split[w].startswith('I-'):
                                    outcome += ' '+x_split[w].lower().strip()
                                    e += 1
                                else:
                                    break
                            sentence_outcomes.append((outcome.strip(), outcome_type))
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
    outcome_occurence = dict([[' [SEP] '.join(i for i in k),v] for k,v in outcome_occurence.items()])
    # print(outcome_occurence)
    with open(args.output_dir+'/outcome_occurrence.json', 'w') as oc:
        json.dump(outcome_occurence, oc, indent=2)
    # print(outcome_occurence)

#extract an outcome based on the label
def identify_outcome_using_label(seq, seq_labels):
    x_split, y_split = seq.split(), seq_labels.split()
    sentence_outcomes = []
    if len(x_split) == len(y_split):
        e = 0
        for m in range(len(x_split)):
            if m == e:
                outcome, outcome_label = '', ''
                if y_split[m].strip().startswith('B-'):
                    outcome += x_split[m].strip()
                    outcome_label += y_split[m].strip()
                    e += 1
                    for w in range(m + 1, len(x_split)):
                        if y_split[w].startswith('I-'):
                            outcome += ' ' + x_split[w].strip()
                            outcome_label += ' ' + y_split[w].strip()
                            e += 1
                        else:
                            break
                    sentence_outcomes.append((outcome, outcome_label))
                else:
                    e += 1
    return sentence_outcomes

#creating an empty directory
def create_directory(name=''):
    _dir = os.path.abspath(os.path.join(os.path.curdir, name))
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir

#
def check_quartiles(args):
    outcomes_occurrence = json.load(open(args.data+'/outcome_occurrence.json'))
    outcomes_occurrence = {k.split(' [SEP] ')[0].strip().lower(): v for k, v in outcomes_occurrence.items()}
    outcomes_occurrence_list = list(set(list(outcomes_occurrence.values())))
    print('{}\nUpper percentile:-{}\nMiddle percentile:-{}\nLower percentile:-{}'.format(
        outcomes_occurrence_list,
        int(np.percentile(list(outcomes_occurrence_list), 75)),
        int(np.percentile(list(outcomes_occurrence_list), 50)),
        int(np.percentile(list(outcomes_occurrence_list), 25))
    ))

def main(args):
    if args.correct_dataset:
        correct_dataset(args.data)
    if args.read_outcome_data:
        read_outcome_data_to_sentences(args.data)
    if args.outcome_frequency:
        #specify the path to the directory with train.txt and dev.txt files i.e. via args.data
        files = [i for i in glob(args.data+'/*.txt') if os.path.basename(i) in ['train.txt', 'dev.txt', 'test.txt']]
        if len(files) > 1:
            dataset, dataset_labels, train_samples_len, eval_samples_len, test_samples_len = fetch_data(args, files=files)
        else:
            dataset, dataset_labels = fetch_data(args, files=[args.data])
        outcome_frequency(data=dataset, labels=dataset_labels)
    if args.create_json:
        data_dir = os.path.abspath(args.data)
        json_data_dir = os.path.join(data_dir, 'json_files')
        if not os.path.exists(json_data_dir):
            os.makedirs(json_data_dir)
        for file in os.listdir(args.data):
            # print(file)
            if file in ['train.txt', 'dev.txt', 'test.txt']:
                json_file_name = file.split('.')
                print(os.path.join(json_data_dir, json_file_name[0]+'.json'))
                with open(data_dir+'/'+file) as f, open(os.path.join(json_data_dir, json_file_name[0]+'.json'),'w') as w:
                    data, example, text, tags = [], {}, [], []
                    for i,row in enumerate(f):
                        if row == '\n':
                            if text:
                                example['text'] = text.copy()
                                example['tags'] = tags.copy()
                                data.append(example.copy())
                            text.clear()
                            tags.clear()
                            example.clear()
                        else:
                            row = row.rstrip()
                            token, label = row.split(' ')
                            text.append(str(token.strip()))
                            tags.append(str(label.strip()))
                    json.dump(data, w, indent=4)


if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='data/ebm_comet_multilabels.txt', help='source of data')
    par.add_argument('--output_dir', help='location to store the date')
    par.add_argument('--correct_dataset', action='store_true', help='move trailing punctuation characters to a new line in the dataset')
    par.add_argument('--partial_contexts', action='store_true', help='reduces the size of training data by picking only sentences whose outcomes have a particular frequnecu,')
    par.add_argument('--read_outcome_data', action='store_true', help='reading train, test and validation files independently')
    par.add_argument('--outcome_frequency', action='store_true', help='check most frequent outcome and the context around it')
    par.add_argument('--create_json', action='store_true', help='preparing json train, test and dev files')
    args = par.parse_args()
    main(args)
    # check_quartiles(args)
