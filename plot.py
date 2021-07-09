import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pprint import pprint
import json
import argparse
import pandas as pd
import seaborn as sns

def mem_fre_len(args):
    x,y,z,t = [], [], [], []
    with open(args.outcome_occurrence, 'r') as oc, open(args.mem_accuracy, 'r') as ma:
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
        data_expanded_df.to_csv('data/ebm-comet/data.csv')
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
        plt.savefig('data/ebm-comet/plot2.png')
        # ax.legend()
        # # show plot
        plt.show()

if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--outcome_occurrence', default='data/ebm-comet/outcome_occurrence.json', help='source file with outcome frequence details')
    par.add_argument('--mem_accuracy', default='data/ebm-comet/mem_accuracy.json', help='source file with memorization accuracy')
    args = par.parse_args()
    mem_fre_len(args)




