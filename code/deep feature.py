#!/usr/bin/env python
# coding: utf-8

get_ipython().system('nvidia-smi  ')

# In[ ]:

import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False) 
model = AutoModel.from_pretrained("Rostlab/prot_bert")
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0) 

# In[ ]:

import pandas as pd
import numpy as np

test_path_0 = './new_stroma.fasta'
test_path_1 = './new_envelope.fasta' 
test_path_2 = './new_thylakoid_lumen.fasta'
test_path_3 = './new_thylakoid_membrane.fasta' 

with open(test_path_0) as Data_file:  
    file_str = Data_file.read()
    all_sequence = file_str.split('\n')
    len_sequence = len(all_sequence)
    sequence_dict_0 = {}
    for i in range(0, len_sequence-1, 2):
        sequence_dict_neg[all_sequence[i]] = [all_sequence[i+1], 0, -1] 

with open(test_path_1) as Data_file:
    file_str = Data_file.read()
    all_sequence = file_str.split('\n')
    len_sequence = len(all_sequence)
    sequence_dict_1 = {}
    for i in range(0, len_sequence-1, 2):
        sequence_dict_pos[all_sequence[i]] = [all_sequence[i+1], 1, -1]
        
with open(test_path_2) as Data_file:
    file_str = Data_file.read()
    all_sequence = file_str.split('\n')
    len_sequence = len(all_sequence)
    sequence_dict_2 = {}
    for i in range(0, len_sequence-1, 2):
        sequence_dict_neg[all_sequence[i]] = [all_sequence[i+1], 2, -1] 

with open(test_path_3) as Data_file: 
    file_str = Data_file.read()
    all_sequence = file_str.split('\n')
    len_sequence = len(all_sequence)
    sequence_dict_3 = {}
    for i in range(0, len_sequence-1, 2):
        sequence_dict_neg[all_sequence[i]] = [all_sequence[i+1], 3, -1] 

seq_data_0 = pd.DataFrame(sequence_dict_0).T  
seq_data_1 = pd.DataFrame(sequence_dict_1).T
seq_data_2 = pd.DataFrame(sequence_dict_2).T 
seq_data_3 = pd.DataFrame(sequence_dict_3).T

test_data = pd.concat([seq_data_0, seq_data_1, seq_data_2, seq_data_3], axis=0)

test_data.columns = ['sequences', 'label', 'unlabel']


seq_data_0

test_seq = []
for seq in test_data.sequences.values:  
    test_seq.append(seq.replace('', ' ')) 
test_seq


embedding = fe(all_train_seq)
embedding = np.array(embedding)
embedding.shape 


features_mean = []  
features_token = []  
for seq_num in range(len(embedding)): 
    seq_len = len(data.sequences.values[seq_num]) 
    start_Idx = 1
    end_Idx = seq_len+1
    features_token.append(np.array(embedding[seq_num][0])) 
    seq_emd = embedding[seq_num][start_Idx:end_Idx]
    features_mean.append(np.mean(seq_emd, axis=0))


data_mean = pd.concat([data, pd.DataFrame(features_mean).set_index(data.index)], axis=1)
data_token = pd.concat([data, pd.DataFrame(features_token).set_index(data.index)], axis=1)


data_mean

data_mean.to_csv('data_mean.csv')

data_token

data_token.to_csv('data_token.csv')





