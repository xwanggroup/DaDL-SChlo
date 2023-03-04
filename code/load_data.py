#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

data_path = '../data/MSchlo578.txt'
str1 = '>'
str2 = '>'
with open(data_path) as Data_file:
    file_str = Data_file.read()
    all_sequence = file_str.split('\n')
    len_sequence = len(all_sequence)

def READ_FASTA():

    sequence_list = []
        
    for i in range(0, len_sequence-1):
        if str1 in all_sequence[i]:
            sequence_list.append(all_sequence[i][1:].split(' ')[0])
    return sequence_list

sequence_list = READ_FASTA()


def read_sequence():
    sequence_list1 = []
    for i in range(0, len_sequence - 1):
        if str1 in all_sequence[i]:
            sequence_list2 = []
            for j in range(i+1,len_sequence):
                if str2 in all_sequence[j]:
                    i = j
                    sequence_list1.append(str3)
                    break
                else:
                    sequence_list2.append(all_sequence[j])
                str3 = ''.join(sequence_list2)
    return sequence_list1

protein_list = read_sequence()

# len(sequence_list)
# type(sequence_list)
df = pd.DataFrame(sequence_list)


df['pro_name'] = df[0]
df = df.drop(0,axis=1)


def tidy_split(df,column,sep=','):
    indexes = []
    new_values = []
    for i,presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        for value in values:
            value  = value.strip() # 去除首尾空格
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes,:].copy()
    new_df[column] = new_values
    new_df.index = [i for i in range(new_df.shape[0])]
    return new_df


df1 = tidy_split(df,'pro_name',sep='|')


df2=[]

for index,row in df1.iterrows():
    if index%2!=0:
        # print(row['pro_name'])
        df2.append(row['pro_name'])
        #df1.drop(row['pro_name'])

df2 = pd.DataFrame(df2)


df3=[]

for index,row in df1.iterrows():
    if index%2==0:
        # print(row['pro_name'])
        df3.append(row['pro_name'])
        #df1.drop(row['pro_name'])
df3 = pd.DataFrame(df3)

df4 = pd.DataFrame(protein_list)

result = pd.concat([df3,df2,df4],axis = 1)
result.to_csv('MS_dataset.csv')


df = pd.read_csv('MS_dataset.csv',
            header=0, 
            names=['pro_ID', 'label', 'sequences'])

label_to_num = {
                    'stroma': 0,
                    'envelope': 1,
                    'lumen': 2,
                    'thylakoid_membrane': 3,
                    'plastoglobule': 4
}
df['pos_num'] = df['label'].map(label_to_num)
df['label'] = df['pos_num']
df = df.drop(['pos_num'],axis=1)
df['label'].fillna(5, inplace = True)
df['label'] = df['label'].astype(int)
df = df.drop(['pro_ID'],axis=1)
df.to_csv('MS_dataset.csv',index=False, header=None)




