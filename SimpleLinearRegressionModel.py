# -*- coding: utf-8 -*-
"""
Created on Sun May 31 03:07:05 2020

@author: Saeed


#Import Libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading Data with Pandas
df = pd.read_csv("Salaries.csv")
X = df.iloc[:,0].values #Independent Variable or Metrics of Feature
y = df.iloc[:,1].values # Dependent Variables 

#Spliting Data into test and train 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,
                                                 random_state=0)

#Model Selection
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
regressor.fit(X_train,y_train)

#predicitng for Y based on the values of X
X_test = X_test.reshape(-1,1)
y_pred = regressor.predict(X_test)
