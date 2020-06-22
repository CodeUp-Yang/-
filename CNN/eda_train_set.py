
import pandas as pd
import os
import re

total = sum(1 for line in open('CNN/data/train_set_0520.csv'))
train_num=int(total*3/4)

def clear(obj):
    text= re.sub(r'[^A-Za-z0-9]+',' ',obj)
    return re.sub(r'^(\s+)|(\s+)$', '', text.lower())

train_data = pd.read_csv('CNN/data/train_set_0520.csv', encoding='utf-8',nrows=train_num)
with open('CNN/data/yes_label.txt', 'w', encoding='utf-8') as yes_f:
    for line in train_data.values:
        line[5]=str(line[5]).replace('\n','')
        if str(line[6])=="yes":
            yes_f.write('yes'+'\t'+str(line[5])+'\n')
yes_f.close()

with open('CNN/data/no_label.txt', 'w', encoding='utf-8') as no_f:
    for line in train_data.values:
        line[5]=str(line[5]).replace('\n','')
        if str(line[6])=="no":
            no_f.write('no'+'\t'+str(line[5])+'\n')
no_f.close()


