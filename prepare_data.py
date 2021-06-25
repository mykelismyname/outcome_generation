# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 25/06/2021 
# @Contact: michealabaho265@gmail.com
import numpy as np
import os
import re
from argparse import ArgumentParser

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

def main(args):
    correct_dataset(args.data)

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--data', default='data/ebm_comet_multilabels.txt', help='source of data')
    args = par.parse_args()
    main(args)
