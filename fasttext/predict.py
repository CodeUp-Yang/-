import fasttext
import pandas as pd
from sklearn.metrics import classification_report

classifier = fasttext.load_model("fasttext/eda_train.model")

# print(classifier.predict (["Synchronize changes of the underlying date value with the temporalAccessorValue","each entry type"]))

texts=[]
real_labels=[]
predict_labels=[]
with open("fasttext/test.csv") as fr:
    for line in fr:
        text=line.split("\t")[1].replace("\n","")
        texts.append(text) 
        real_labels.append(line.split("\t")[0].replace("__label__",""))
        predict_labels.append(classifier.predict(text)[0].replace("__label__",""))


# print(predict_labels)
# print(real_labels)


print(classification_report(real_labels, predict_labels))




# 将结果保存为xlsx文件
df = pd.DataFrame({'text':texts, "real_label": real_labels , "predict_label": predict_labels})
df.to_csv('fasttext/eda_predict_result_fasttext_test.csv', index=False)