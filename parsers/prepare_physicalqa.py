import numpy as np
import os
import json
import pickle
import sys

sys.path.append("../")
import classes
import config

def combine_answers(item):
    return [item['sol1'], item['sol2']]

def offset_answer(answer, offset):
    if answer:
        return answer-offset
    else:
        return answer

config_data=config.physicaliqa_config

# Load dataset examples
dataname=config_data['dataname']
binfile=f'{config.bin_dir}/{dataname}.bin'

dataset=classes.Dataset(dataname)

parts=['train', 'dev']

for split in parts:
    input_file='%s/%s' % (config_data['input_data_loc'], config_data[f'{split}_input_file'])
    labels_file='%s/%s' % (config_data['input_data_loc'], config_data[f'{split}_labels_file'])

    with open(input_file, 'r') as f:
        for index, l in enumerate(f):
            item = json.loads(l)
            split_data=getattr(dataset, split)
            print(l)
            an_entry=classes.Entry(
                split=split,
                id='{}-{}'.format(split, item['id']),
                intro="",
                question=item['goal'],
                answers=combine_answers(item),
                correct_answer=None, # if split == 'test' else item['label']
            )
            split_data.append(an_entry)

with open(binfile, 'wb') as w:
    pickle.dump(dataset, w)
