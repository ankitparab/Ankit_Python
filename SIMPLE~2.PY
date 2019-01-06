# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 22:45:22 2018

@author: Ankit
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression

Regressor = LinearRegression()
Regressor.fit(x_train,y_train)


#Predicting the Test Result

y_pred=Regressor.predict(x_test)

#Visulaizing the training set result

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,Regressor.predict(x_train),color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



#Visulaizing the test set result

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,Regressor.predict(x_train),color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()