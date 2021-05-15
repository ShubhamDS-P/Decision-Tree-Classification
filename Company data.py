# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 21:00:53 2021

@author: Shubham
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("D:\\Data Science study\\Assignment of Data Science\\Sent\\13 Decision Tree\\Company_Data.csv")
data.head()
data.describe
data[data.isnull().any(axis=1)]   #There are no null values in the data.
print(data.info())
data.columns
colnames = list(data.columns)
predictors = colnames[1:]
target = colnames[0]


import numpy as np
import seaborn as sb

#let us see the boxplot for the given data 
sb.boxplot(data = data)

#There are few plots with the outliers present in them so we will create a function  for finding out these outliers

outliers=[]
def detect_outlier(data_1):         # Creating a function for finding the outliers
    outliers.clear()
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

# Outliers present in the sales
outlier_sales = detect_outlier(data.Sales)
print(outlier_sales)
len(outlier_sales)

# Outliers present in the CompPrice
outlier_compprice = detect_outlier(data.CompPrice)
print(outlier_compprice)
len(outlier_compprice)

# Outliers present in the Price 
outlier_price = detect_outlier(data.Price)
print(outlier_price)
len(outlier_price)

#lets create the function which will take an index number as input and gives out 
#the countplot for that particular column

def count_plot(x):
    plot = sb.countplot(x = data.columns[x], data = data, palette = 'hls')
    return plot


#let us see the plot of all the columns in the data
count_plot(0)
count_plot(1)
count_plot(2)
count_plot(3)
count_plot(4)
count_plot(5)
count_plot(6)
count_plot(7)
count_plot(8)
count_plot(9)
count_plot(10)

#Lets see how a plot against the sales look like
pd.crosstab(data.Sales,data.CompPrice).plot(kind = "bar")
pd.crosstab(data.Sales,data.Advertising).plot(kind = "bar")
pd.crosstab(data.Sales,data.Price).plot(kind = "bar")


#we will create new dataframe with the encoded data since the decision tree classifier can't handle the categorical string data
#first we will create a dictionary for the values which needs to be replaced.

data.dtypes

data["ShelveLoc"].value_counts()

data["Urban"].value_counts()

data["US"].value_counts()

val_replace = {"ShelveLoc": {"Medium":2, "Bad":3, "Good":1}, "Urban": {"Yes":1, "No":0}, "US": {"Yes":1, "No":0}}

x = data.replace(val_replace).copy()

#x["Sales"].astype(int)
#x.dtypes


#Also we will type cast the sales column values into the int because the classifier doesn't accepts floating values which it considers as countinuous values

model_data = x.copy()

model_data.dtypes

# Let's split the data into train and test data so we can have different datas for model building and model testing.
from sklearn.model_selection import train_test_split
train,test = train_test_split(model_data,test_size = 0.2)


from sklearn.tree import DecisionTreeClassifier
help(DecisionTreeClassifier)
model = DecisionTreeClassifier(criterion = 'entropy')

#fitting the model with data

model.fit(train[predictors],train[target].astype(int))

fitted_model = model.fit(train[predictors],train[target].astype(int))
# we used type casting for target to change it from float to int because float because float type creats an error of continuous label type.

#we will predict the test data based on the model which we have just created

model.predict(test[predictors])

pred = pd.Series(model.predict(test[predictors]))
#Let's plot the decision tree model which we have fitted in the above code line.

from sklearn import tree

tree.plot_tree(fitted_model)


type(pred)
print(pred)
pd.Series(pred).value_counts()
pd.crosstab(test[target],pred)

print(test[target])

#Accurracy  ---Train & Test

acc = np.mean(model.predict(train[predictors]) == train[target])
acc

accuracy = np.mean(pd.Series(train[target]).reset_index(drop = True) == pd.Series(model.predict(train[predictors])))
accuracy
np.mean(test[target].reset_index(drop =True, inplace =True) == model.predict(test[predictors]))
original = (test[target])
predicted = pd.Series(model.predict(test[predictors]))



#for both the programs related to the decision tree my accuracy is not getting what I expect and I don't know where the problem is 
# so I hope you will point out the issue for both codes and help me improve it any way you can.