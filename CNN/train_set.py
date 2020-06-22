
import pandas as pd
import os
import re

total = sum(1 for line in open('CNN/data/train_set_0520.csv'))
train_num=int(total*3/4)

def clear(obj):
    text= re.sub(r'[^A-Za-z0-9]+',' ',obj)
    return re.sub(r'^(\s+)|(\s+)$', '', text.lower())

train_data = pd.read_csv('CNN/data/train_set_0520.csv', encoding='utf-8',nrows=train_num)
with open('CNN/data/yes.txt', 'w', encoding='utf-8') as yes_f:
    for line in train_data.values:
        line[5]=str(line[5]).replace('\n','')
        if str(line[6])=="yes":
            yes_f.write(clear(str(line[5]))+'\n')
yes_f.close()

with open('CNN/data/no.txt', 'w', encoding='utf-8') as no_f:
    for line in train_data.values:
        line[5]=str(line[5]).replace('\n','')
        if str(line[6])=="no":
            no_f.write(clear(str(line[5]))+'\n')
no_f.close()



test_data = pd.read_csv('CNN/data/train_set_0520.csv', encoding='utf-8',skiprows=train_num+1)
with open('CNN/data/yes_test.txt', 'w', encoding='utf-8') as yes_test_f:
    for line in test_data.values:
        line[5]=str(line[5]).replace('\n','')
        if str(line[6])=="yes":
            yes_test_f.write(clear(str(line[5]))+'\n')
yes_test_f.close()

with open('CNN/data/no_test.txt', 'w', encoding='utf-8') as no_test_f:
    for line in test_data.values:
        line[5]=str(line[5]).replace('\n','')
        if str(line[6])=="no":
            no_test_f.write(clear(str(line[5]))+'\n')
no_test_f.close()

# /尝试预测
# /换另一种方式尝试