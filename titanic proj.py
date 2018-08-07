# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 23:41:30 2018

@author: DELL
"""

import numpy as np 
import pandas as pd 
import seaborn as sb 
import matplotlib.pyplot as plt 
import sklearn 
from pandas import Series, DataFrame 
from pylab import rcParams 
from sklearn import preprocessing 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report

Url= "https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv "
titanic_df = pd.read_csv(Url)

def data_preprocessing(df):
    df = df.loc[:,['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]
    df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True) 
    
    return df

def handle_non_numeric_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            
            x = 0            
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1                    
            df[column] = list(map(convert_to_int, df[column]))
            
    return df



titanic_df1 = data_preprocessing(titanic_df)
print(titanic_df1.tail())


titanic_df1 = handle_non_numeric_data(titanic_df1)
print(titanic_df1.tail())



X = np.array(titanic_df1.drop(['Survived'], 1).astype(float))
Y = np.array(titanic_df1['Survived'])



X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
d_tree = DecisionTreeClassifier(min_samples_split=20, random_state=99)
d_tree.fit(X_train, Y_train)


Y_pred = d_tree.predict(X_test)
print("Accuracy is ", accuracy_score(Y_test,Y_pred)*100)




# this produces a 2x2 numpy array (matrix)
confusion = metrics.confusion_matrix(Y_test, Y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print(f"Accuracy calculation using confusion metrics : {((TP + TN) / float(TP + TN + FP + FN))}")





print(f"classification_error using accuracy_score is : {1 - metrics.accuracy_score(Y_test, Y_pred)}")
print(f"classification_error using confusion metrics is : {(FP + FN) / float(TP + TN + FP + FN)}")
















