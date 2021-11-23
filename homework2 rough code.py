# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:13:52 2020

@author: farah
"""

import pandas as pd 
import numpy as np
import scipy as sp 
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv('C:/Users/farah/Desktop/FALL 2020-21/OMIS 3020/creditscores_dataset.csv')
print(df.info())

credit_amount = pd.get_dummies(df['Requested Credit Amount'],drop_first=True)
income = pd.get_dummies(df['Monthly Income'],drop_first=True)
expense = pd.get_dummies(df['Monthly Expense'],drop_first=True)
status = pd.get_dummies(df['Marital Status'],drop_first=True)


df = pd.get_dummies(df, drop_first=True)

pca = PCA().fit(x)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


true_values = y 
predicted_values = model.predict(x)
f1_score(true_values, predicted_values, average='weighted')