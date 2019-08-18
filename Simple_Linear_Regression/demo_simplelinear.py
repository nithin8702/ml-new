# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 12:43:39 2018

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

import sklearn.metrics as sm
#sm.accuracy_score(y_test, y_pred) # accuracy is for classification, not for regression
round(sm.mean_absolute_error(y_test, y_pred),2)

#sm.mean_absolute_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])
#sm.mean_absolute_error([3000, -0.5, 2000, 7000], [2.5, 0.0, 2000, 8000])

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('salary vs exp(training set)')
plt.xlabel('yr of exp')
plt.ylabel('sal')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue') #best fit line is same for both train and test
plt.title('salary vs exp(test set)')
plt.xlabel('yr of exp')
plt.ylabel('sal')
plt.show()

regressor.coef_ #slope
regressor.intercept_ #intercept
(4 * regressor.intercept_) + regressor.coef_