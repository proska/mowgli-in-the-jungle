import numpy as np
import os
import json
import pickle
import sys

sys.path.append("../")

import classes
import config
import utils

################## Utility functions #######################

def _part_a(item):
    if 'ctx_a' not in item:
        return item['ctx']
    if 'ctx' not in item:
        return item['ctx_a']
    if len(item['ctx']) == len(item['ctx_a']):
        return item['ctx']
    return item['ctx_a']

def _part_bs(item):
    if ('ctx_b' not in item) or len(item['ctx_b']) == 0:
        return ''
    else:
        return item['ctx_b']

def compose_hs_question(item):
    p1=_part_a(item)
    p2=_part_bs(item)
    p3=item['activity_label']

    return [p1, p2, p3]

def combine_siqa_answers(item, offset):
    return ['']*offset + [item['answerA'], item['answerB'], item['answerC']]

def combine_piqa_answers(item, offset):
    return ['']*offset + [item['sol1'], item['sol2']]

def combine_anli_answers(item, offset):
    return ['']*offset + [item['hyp1'], item['hyp2']]

#################### PARSERS ###########################

def prepare_anli_dataset():
    config_data=config.cfg['anli']

    # Load dataset examples
    dataname=config_data['dataname']

    inputdir=config_data['input_data_loc']

    dataset=classes.Dataset(dataname)

    parts=['train', 'dev']

    offset=config_data['answer_offset']

    for split in parts:
        input_file='%s/%s' % (inputdir, config_data[f'{split}_input_file'])
        labels_file='%s/%s' % (inputdir, config_data[f'{split}_labels_file'])
        labels=utils.load_predictions(labels_file)

        with open(input_file, 'r') as f:
            for index, l in enumerate(f):
                item = json.loads(l)
                split_data=getattr(dataset, split)
                print(l)
                an_entry=classes.Entry(
                    split=split,
                    id='{}-{}'.format(split, item["story_id"]),
                    question=[item['obs1'], item['obs2']],
                    answers=combine_anli_answers(item, offset),
                    correct_answer=None if split == 'test' else labels[index]
                )
                split_data.append(an_entry)
    return dataset

def prepare_hellaswag_dataset():
    config_data=config.cfg['hellaswag']

    # Load dataset examples
    dataname=config_data['dataname']

    inputdir=config_data['input_data_loc']

    dataset=classes.Dataset(dataname)

    parts=['train', 'dev']

    offset=config_data['answer_offset']

    for split in parts:
        input_file='%s/%s' % (inputdir, config_data[f'{split}_input_file'])
        labels_file='%s/%s' % (inputdir, config_data[f'{split}_labels_file'])
        labels=utils.load_predictions(labels_file)

        with open(input_file, 'r') as f:
            for index, l in enumerate(f):
                item = json.loads(l)
                split_data=getattr(dataset, split)
                print(l)
                an_entry=classes.Entry(
                    split=split,
                    id='{}-{}'.format(split, item['ind']),
                    question=compose_hs_question(item),
                    answers=['']*offset + item['ending_options'],
                    correct_answer=None if split == 'test' else labels[index],
                    metadata={'activity_label': item['activity_label'], 'dataset': item['dataset'], 'split_type': item['split_type']}
                )
                split_data.append(an_entry)
    return dataset

def prepare_socialiqa_dataset():
    config_data=config.cfg['socialiqa']

    # Load dataset examples
    dataname=config_data['dataname']

    inputdir=config_data['input_data_loc']

    dataset=classes.Dataset(dataname)

    parts=['train', 'dev']

    offset=config_data['answer_offset']

    for split in parts:
        input_file='%s/%s' % (inputdir, config_data[f'{split}_input_file'])
        labels_file='%s/%s' % (inputdir, config_data[f'{split}_labels_file'])
        labels=utils.load_predictions(labels_file)

        with open(input_file, 'r') as f:
            for index, l in enumerate(f):
                item = json.loads(l)
                split_data=getattr(dataset, split)
                #print(l)
                an_entry=classes.Entry(
                    split=split,
                    id='{}-{}'.format(split, index),
                    question=[item['context'], item['question']],
                    answers=combine_siqa_answers(item, offset),
                    correct_answer=None if split == 'test' else labels[index]
                )
                split_data.append(an_entry)
    return dataset

def parse_physicaliqa():
    config_data=config.cfg['physicaliqa']

    # Load dataset examples
    dataname=config_data['dataname']

    dataset=classes.Dataset(dataname)

    inputdir=config_data['input_data_loc']

    parts=['train', 'dev']

    offset=config_data['answer_offset']

    for split in parts:
        input_file='%s/%s' % (inputdir, config_data[f'{split}_input_file'])
        labels_file='%s/%s' % (inputdir, config_data[f'{split}_labels_file'])
        labels=utils.load_predictions(labels_file)

        with open(input_file, 'r') as f:
            for index, l in enumerate(f):
                item = json.loads(l)
                split_data=getattr(dataset, split)
                print(l)
                an_entry=classes.Entry(
                    split=split,
                    id='{}-{}'.format(split, item['id']),
                    question=[item['goal']],
                    answers=combine_piqa_answers(item, offset),
                    correct_answer=None if split == 'test' else labels[index]
                )
                split_data.append(an_entry)
    return dataset
