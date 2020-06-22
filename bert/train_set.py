
import pandas as pd
import os
import re

total = sum(1 for line in open('data/train_set_0520.csv'))
train_num=int(total*1/2)
dev_num=int(total*1/4)

def clear(obj):
    text= re.sub(r'[^A-Za-z0-9]+',' ',obj)
    return re.sub(r'^(\s+)|(\s+)$', '', text)

# 切分数据——训练集
train_data = pd.read_csv('data/train_set_0520.csv', encoding='utf-8',nrows=train_num)
with open('data/train.tsv', 'w', encoding='utf-8') as train_f:
    for line in train_data.values:
        line[5]=str(line[5]).replace('\n','')
        train_f.write((str(line[6])+'\t'+clear(str(line[5]))+'\n'))
train_f.close()

# 切分数据——验证集
dev_data = pd.read_csv('data/train_set_0520.csv', encoding='utf-8',header=train_num+1,nrows=dev_num)
with open('data/dev.tsv', 'w', encoding='utf-8') as dev_f:
    for line in dev_data.values:
        line[5]=str(line[5]).replace('\n','')
        dev_f.write((str(line[6])+'\t'+clear(str(line[5]))+'\n'))
dev_f.close()

# 切分数据——测试集
test_data = pd.read_csv('data/train_set_0520.csv', encoding='utf-8',header=train_num+dev_num+1)
with open('data/test.tsv', 'w', encoding='utf-8') as test_f:
    for line in test_data.values:
        line[5]=str(line[5]).replace('\n','')
        test_f.write((str(line[6])+'\t'+clear(str(line[5]))+'\n'))
test_f.close()


