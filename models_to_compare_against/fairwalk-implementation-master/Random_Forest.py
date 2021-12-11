# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 08:48:17 2021

@author: aares
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:13:54 2021

@author: aares
"""

import numpy as np
from sklearn.metrics import f1_score
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# Instantiate model with 1000 decision trees

df = pd.read_csv('20_5_f.csv')  # 'Sliding_Window_16_2_2.csv'
df.describe()
print(df.shape)

# df=df.drop(['FixSeqDSST','FixSeqOTDS'],axis=1)
df = df.dropna(axis=0)
labels = np.array(1 - df['target'])  # interchange 0's to 1's
df = df.drop('target', axis=1)
df_title = df.columns
print(df.shape)
features = np.array(df)
features = zscore(features, axis=0, nan_policy='omit')

i = 0
Train_acc = list()
Test_acc = list()
F1_list = list()
kf = KFold(n_splits=5, shuffle=False)
kf = StratifiedKFold(n_splits=5, shuffle=False)
for train_index, test_index in kf.split(df, labels):
    # for train_index, test_index in kf.split(df):
    X_train = df.iloc[train_index].loc[:]
    X_test = df.iloc[test_index].loc[:]
    Y_train = labels[train_index]
    Y_test = labels[test_index]

    i += 1
    print('Fold ', i)
    # train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42, shuffle = False)

    # baseline_preds = X_test[:, 0]
    # baseline_errors = abs(baseline_preds - Y_test)
    # print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, Y_train);
    predictions = rf.predict(X_test)
    errors = abs(predictions - Y_test)

    result_trial = predictions > 0.5
    result_trial = result_trial * 1

    Train_acc.append(rf.score(X_train, Y_train))
    Test_acc.append(rf.score(X_test, Y_test))
    F1_list.append(f1_score(Y_test, result_trial, labels=None, pos_label=1, average='binary', sample_weight=None,
                            zero_division='warn'))

    print('Mean Absolute Error:', round(np.mean(errors), 2))

    print('Number of erros', sum(abs(Y_test - result_trial)))
    # print('Accuracy',1-sum(abs(test_labels-result_trial))/len(test_labels)*100,'%')

    print('Accuracy')
    # print(1-sum(abs(test_labels-result_trial))/len(test_labels))
    print(sklearn.metrics.accuracy_score(Y_test, result_trial, normalize=True, sample_weight=None))

    print('F1', f1_score(Y_test, result_trial, labels=None, pos_label=1, average='binary', sample_weight=None,
                         zero_division='warn'))
    print('Confusion Matrix',
          sklearn.metrics.confusion_matrix(Y_test, result_trial, labels=None, sample_weight=None, normalize=None))
