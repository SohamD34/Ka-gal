import numpy as np 
import pandas as pd
from statistics import mode
import os

%cd E:/Kaggle Competitions/Titanic

all_submissions = [ 'dtc.csv',
 'output.csv',
 'output_RFC.csv',
 'output_svm_poly.csv',
 'output_svm_rb.csv',
 'output_svm_rbf.csv',
 'output_svm_sig.csv']

all_predictions = []

for file in all_submissions:
    df = pd.read_csv('../Titanic/'+file)
    predictions = list(df['Survived'])
    all_predictions.append(predictions)

all_predictions = np.array(all_predictions).T
print(all_predictions.shape)

final_pred = []

for i in range(len(all_predictions)):
    row = all_predictions[i]
    pred = mode(row)
    final_pred.append(pred)

print(len(final_pred))

sub = pd.DataFrame()
sub['PassengerId'] = df['PassengerId']
sub['Survived'] = final_pred

sub.to_csv('combined.csv',index=False)