import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pprint import pprint
import json
import argparse
import pandas as pd
import seaborn as sns
import math
from glob import glob
import os

def mem_fre_len(args):
    x,y,z,t = [], [], [], []
    with open(args.outcome_occurence, 'r') as oc, open(args.mem_accuracy, 'r') as ma:
        outcomes = json.load(oc)
        # outcomes = {k.split(' [SEP] ')[0].strip(): v for k, v in outcomes.items()}
        mem_acc = json.load(ma)
        mem_acc = dict([[int(k),v] for k,v in mem_acc.items()])
        for outcome in outcomes:
            if outcomes[outcome] in mem_acc:
                out_come, outcome_type = outcome.split(' [SEP] ')
                outcome_length = len(out_come.strip().split())
                #save details of an outcome in (x-length, y-frequency, z-accuracy, t-outcompe type)
                x.append(outcome_length)
                y.append(outcomes[outcome])
                z.append(mem_acc[outcomes[outcome]])
                t.append(outcome_type)

        # save details of an outcome in (length, frequency, accuracy, outvcome_type)
        data = [[m, n, o, p] for m, n, o, p in zip(x, y, z, t)]
        data = sorted(data, key=lambda x:x[0])
        splitter_length = int(data[-1][0]/3)
        data_expanded = []
        for i in data:
            if i[0] < splitter_length:
                i.append('Short span length')
                data_expanded.append(i)
            if i[0] >= splitter_length and i[0] < splitter_length*2:
                i.append('Medium span length')
                data_expanded.append(i)
            if i[0] > splitter_length*2:
                i.append('Long spans')
                data_expanded.append(i)

        data_expanded_df = pd.DataFrame(data_expanded, columns=['Length','Frequency','Accuracy','Outcome_type','Length_type'])
        d = sns.FacetGrid(data_expanded_df, col='Length_type', height=4, aspect=0.8, hue='Outcome_type', sharey=True)
        d.map(sns.scatterplot, 'Frequency', 'Accuracy', alpha=.8)
        d.fig.subplots_adjust(top=0.8)  # adjust the Figure in rp
        d.fig.suptitle('Model Recalling Rate')
        d.add_legend()
        data_expanded_df.to_csv(args.output_dir+'/data.csv')
        # # Creating figure
        # fig = plt.figure(figsize=(10, 7))
        # ax = plt.axes(projection="3d")
        # # Creating color map
        # my_cmap = plt.get_cmap('hsv')
        # p = {y:x for x,y in enumerate(list(set(t)))}
        # t = [p[i] for i in t]
        #
        # # Creating plot
        # sctt = ax.scatter3D(x, y, z,
        #                     alpha=0.8,
        #                     c=t,
        #                     cmap=my_cmap,
        #                     marker='o')
        #
        # plt.title("Model Recalling Rate", fontweight='bold')
        # ax.set_xlabel('Outcome span length', fontweight='bold')
        # ax.set_ylabel('Mention frequency', fontweight='bold')
        # ax.set_zlabel('Memorization Accuracy', fontweight='bold')
        # # fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)
        # plt.title('Model Recalling Rate')
        plt.savefig(args.output_dir+'/plot.png')
        # ax.legend()
        # # show plot
        plt.show()

def compute_average(f):
    a = json.load(open(f, 'r'))
    a = np.mean(list(a.values()))
    print(a)

#visualise the comparison between random masking results vs custom masking results
def random_masking_custom_masking(f1, f2):
    def return_perplexities(read_in_lines):
        perplexisties = []
        for line in read_in_lines[1:]:
            # print(line)
            tr_loss, val_loss, val_perp = line.split('\t')
            perplexisties.append(math.exp(float(tr_loss)))
            # print(tr_loss, math.exp(float(tr_loss)))
        return perplexisties

    def sort_list(l):
        l = [(i-1.0,l.index(i)) for i in l]
        l = sorted(l, key=lambda x:x[0], reverse=True)
        return l

    fig, ax = plt.subplots()
    with open(f1, 'r') as a, open(f2, 'r') as b:
        f1_perplexisties = return_perplexities(a.readlines())
        print(f1_perplexisties)
        f2_perplexisties = return_perplexities(b.readlines())[:20]
        f1_target_perp = sort_list(f1_perplexisties)
        f2_target_perp = sort_list(f2_perplexisties)
        print(f1_target_perp)
        f1_epochs = [i+1 for i in range(len(f1_perplexisties))]
        f2_epochs  = [i+1 for i in range(len(f2_perplexisties))]

        ax.plot(f1_epochs, f1_perplexisties, marker="d", markevery=[2], label='Custom Mask', color='chocolate')
        ax.plot(f2_epochs, f2_perplexisties, label='Random Mask (Generic BERT)', color='seagreen')
        ax.legend(loc='upper right', fontsize='medium')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel('epochs')
        plt.ylabel('Perplexity')
        plt.title('Comparison of the two masking approaches')
        plt.savefig('plot.png')
        plt.show()


def fewshot_visualization(args):
    dirs = glob(args.data+'/test*')
    accuracy = {}
    for d in dirs:
        results_file = d+'/'+args.recall_metric+'/mem_accuracy.json'
        with open(results_file, 'r') as r_f:
            results = json.load(r_f)
            results = {'frequency': [int(i) for i in list(results.keys())], 'Accuracy': list(results.values())}
            if os.path.basename(d).__contains__('pre'):
                accuracy['Prefix'] = results
            if os.path.basename(d).__contains__('pos'):
                accuracy['Postfix'] = results
            if os.path.basename(d).__contains__('clo'):
                accuracy['Cloze'] = results
            if os.path.basename(d).__contains__('mix'):
                accuracy['Mixed'] = results
    print(accuracy)
    frequencies = [j['frequency'] for i,j in accuracy.items()]
    frequencies = [i for j in frequencies for i in j]
    max_frequency = max(frequencies)

    if max_frequency % 10 == 0:
        pass
    else:
        for i in range(1, 10):
            max_frequency += 1
            if max_frequency % 10 == 0:
                break
    frequency_ranges = []
    print(max_frequency)
    m = 1
    for n in np.arange(10, 71, 10):
        n = n+1
        frequency_ranges.append(range(m,n))
        m = n

    z = np.zeros((4,len(frequency_ranges)))
    for k,prompt_type in enumerate(accuracy):
        pt = []
        print(prompt_type,'\n')
        for v,rge in enumerate(frequency_ranges):
            h,g = accuracy[prompt_type]['frequency'], accuracy[prompt_type]['Accuracy']
            s = [(i,j) for i,j in zip(h,g) if i in rge]
            print(accuracy[prompt_type]['frequency'], list(rge))
            print(s)
            if s:
                s_mean = np.mean([[i[1]] for i in s])
                z[k][v] = np.round(s_mean, 4)
    z = z*100
    # print(z)
    # print(z.astype(int))
    z_frame = pd.DataFrame(z)
    z_frame.columns = ['{}-{}'.format(i[0],i[-1]) for i in frequency_ranges]
    z_frame.index = [i for i in accuracy]
    print(z_frame)
    ax = sns.heatmap(z_frame, annot=True, cmap="Greens", fmt='.2f', cbar_kws={'label': args.recall_metric})
    plt.xlabel('Target outcome occurrence frequency')
    plt.ylabel('{} accuracy'.format(args.recall_metric))
    plt.title('Few shot setting')
    plt.savefig(args.output_dir+'/few_shot_{}.png'.format(args.recall_metric))
    plt.show()

    sub_plots = False
    if sub_plots:
        fig, axs = plt.subplots(2, 2)
        n = 0
        accuracy_access = dict(list(enumerate(accuracy)))
        for i in range(2):
            for j in range(2):
                prompt_type_index = dict((n,m) for m,n in accuracy_access.items())
                print(accuracy_access[n])
                x = [int(k) for k in accuracy[accuracy_access[n]]['frequency']]
                x_sorted = sorted(x)
                x_sorted_indices = [x.index(i) for i in x_sorted]
                y = [int(k) for k in accuracy[accuracy_access[n]]['Accuracy']]
                y_sorted = [y[i] for i in x_sorted_indices]
                axs[i, j].plot(x_sorted, y_sorted)
                axs[i, j].set_title('Axis [{} {}]'.format(i,j))
                n += 1
        for ax in axs.flat:
            ax.set(xlabel='x-label', ylabel='y-label')
        for ax in axs.flat:
            ax.label_outer()
        plt.show()

if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--outcome_occurence', default='data/ebm-comet/outcome_occurrence.json', help='source file with outcome frequence details')
    par.add_argument('--mem_accuracy', default='data/ebm-comet/mem_accuracy.json', help='source file with memorization accuracy')
    par.add_argument('--output_dir', default='output/', help='directory to send evaluation results')
    par.add_argument('--data', default='data', help='directory emorization accuracy')
    par.add_argument('--recall_metric', default='exact_match', help='exact_match or partial_matial')
    par.add_argument('--function', default='compute_average', help='function to execute')
    args = par.parse_args()
    if args.function == 'compute_average':
        compute_average(f=args.mem_accuracy)
    elif args.function == 'accuracy_per_freq_len':
        mem_fre_len(args)
    elif args.function == 'masking_visualisation':
        sys_args = sys.argv
        print(sys_args)
        random_masking_custom_masking(f1=sys_args[1], f2=sys_args[2])
    elif args.function == 'fewshot_visualisation':
        fewshot_visualization(args)




