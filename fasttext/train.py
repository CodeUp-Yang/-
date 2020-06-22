import fasttext
import pandas as pd
import os
import re
import numpy as np

total = sum(1 for line in open('train_set_0520.csv'))
train_num=int(total*3/4)



def clear(obj):
    text=re.sub(r'[^A-Za-z0-9]+',' ',obj)
    return re.sub(r'^(\s+)|(\s+)$', '', text)
# 切分数据——训练集
train_data = pd.read_csv('train_set_0520.csv', encoding='utf-8',nrows=train_num)
with open('train.csv', 'w', encoding='utf-8') as train_f:
    for line in train_data.values:
        line[5]=str(line[5]).replace('\n','')
        train_f.write(('__label__'+str(line[6])+'\t'+clear(str(line[5]))+'\n'))
train_f.close()

# 切分数据——测试集
test_data = pd.read_csv('train_set_0520.csv', encoding='utf-8',header=train_num+1)
with open('test.csv', 'w', encoding='utf-8') as test_f:
    for line in test_data.values:
        line[5]=str(line[5]).replace('\n','')
        test_f.write(('__label__'+str(line[6])+'\t'+clear(str(line[5]))+'\n'))
test_f.close()


def train_model(ipt=None, opt=None, model='', dim=100, epoch=20, lr=0.1, loss='softmax'):
    np.set_printoptions(suppress=True)
    if os.path.isfile(model):
    #     classifier = fasttext.load_model(model)
        classifier = fasttext.train_supervised(ipt, label='__label__', dim=dim, epoch=epoch,
                                         lr=lr, wordNgrams=2, loss=loss)
        classifier.save_model(opt)
    else:
        classifier = fasttext.train_supervised(ipt, label='__label__', dim=dim, epoch=epoch,
                                         lr=lr, wordNgrams=2, loss=loss)
        classifier.save_model(opt)
    return classifier


dim = 100
lr = 0.0001
epoch = 50
model = f'train.model'



classifier = train_model(ipt='train.csv',
                         opt=model,
                         model=model,
                         dim=dim, epoch=epoch, lr=lr
                         )

result = classifier.test('train.csv')
print(result)


