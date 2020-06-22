
import pandas as pd
from sklearn.metrics import classification_report

result=pd.read_csv('CNN/data/eda_predict_result_CNN_test.csv')
texts=result['text']
real_labels=result['real_label']
predict_labels=result['predict_label']
print(predict_labels)

target_names = ['class no', 'class yes']
print(classification_report(real_labels, predict_labels))



