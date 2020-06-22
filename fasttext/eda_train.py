import fasttext
import pandas as pd
import os
import re
import numpy as np





# 切分数据——训练集
train_data = pd.read_table('./eda_trainset.txt', encoding='utf-8')
with open('fasttext/eda_train.csv', 'w', encoding='utf-8') as train_f:
    for line in train_data.values:
        train_f.write(('__label__'+str(line[0])+'\t'+str(line[1]))+'\n')
train_f.close()


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
model = f'fasttext/eda_train.model'




classifier = train_model(ipt='fasttext/eda_train.csv',
                         opt=model,
                         model=model,
                         dim=dim, epoch=epoch, lr=lr
                         )

result = classifier.test('fasttext/test.csv')
print(result)


