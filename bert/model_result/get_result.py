import os
import pandas as pd
from sklearn.metrics import classification_report




path = "model_result/testset"
pd_all = pd.read_csv(os.path.join(path, "test_results.tsv") ,sep='\t',header=None) 

texts=[]
real_labels=[]
predict_labels=[]

for index in pd_all.index:
    yes_score = pd_all.loc[index].values[0]
    no_score = pd_all.loc[index].values[1]

    if max(yes_score, no_score) == yes_score:
        # data.append(pd.DataFrame([index, "yes"],columns=['id','polarity']),ignore_index=True)
        predict_labels.append('yes')
    else:
        #data.append(pd.DataFrame([index, "no"],columns=['id','polarity']),ignore_index=True)
        predict_labels.append('no')
    #print(no_score, positive_score, no_score)


with open("./data/test.tsv") as fr:
    for line in fr:
        text=line.split("\t")[1].replace("\n","")
        texts.append(text) 
        real_labels.append(line.split("\t")[0])
        
# target_names = ['class no', 'class yes']
print(classification_report(real_labels, predict_labels))





df = pd.DataFrame({'text':texts, "real_label": real_labels , "predict_label": predict_labels})
df.to_csv('data/predict_result_Bert_test.csv', index=False)