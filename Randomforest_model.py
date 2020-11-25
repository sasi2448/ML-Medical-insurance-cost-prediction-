# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:40:44 2020

@author: Sasi
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('C:/Users/Sasi/Desktop/data science/Deployment/Insurance price/Data/data.csv')

df.drop_duplicates(inplace = True) #Droping the Duplicates

df.drop(['sex','region'],axis =1,inplace = True)

df = pd.get_dummies(df, columns=['smoker'],drop_first=True)

X = df.drop(['charges'], axis = 1)
y = df['charges']



regressor = RandomForestRegressor(n_estimators = 1100,min_samples_split=2,min_samples_leaf=4,max_features='auto',max_depth = 5,bootstrap=True)

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[45, 55, 2,1]]))