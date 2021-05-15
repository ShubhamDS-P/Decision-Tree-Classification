# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:24:10 2021

@author: Shubham
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

fraud_data = pd.read_csv("D:\\Data Science study\\Assignment of Data Science\\Sent\\13 Decision Tree\\Fraud_check.csv")
fraud_data
fraud_data.head()
fraud_data.info()
# Checking the null values in the data
null = fraud_data.isnull().any(axis = 1)
null.describe()
# This shows there are no null values present in the dataframe
fraud_data.columns

# Changing the column names, since the dots in them creates errors while running the code.

fraud_data.rename(columns={'Marital.Status':'Marital_Status', 'Taxable.Income':'Taxable_Income', 'City.Population':'City_Population', 'Work.Experience':'Work_Experience'},inplace= True)

# Now we will insert a column at the 0th position as an out put column where we will divide our taxable income into 'good' and 'risky'.

fraud_data.insert(0,"output",str) # Since we are going to insert the strings in the column hence we set it as 'str'

#y = fraud_data.iloc[:,2]
#type(y)
fraud_data.loc[fraud_data.Taxable_Income<=30000,"output"] = "Risky"
fraud_data.loc[fraud_data.Taxable_Income>30000,"output"] = "Good"
fraud_data['output'].describe()

type(fraud_data.output)

# Checking the null values once again in the data.

#null = fraud_data.isnull().any(axis = 1)
#null.describe()

# Another method to find the null values in the data.

fraud_data[fraud_data.isnull().any(axis = 1)]

# There are no null values in the data.

print(fraud_data.info())
#fraud_data.drop("Taxable_Income",axis = 1, inplace = True) # We will remove the taxable income column which is of no more use to us in the further process.
print(fraud_data.info())

#Now we will create the predictors and target for our further model building.

#colnames = list(fraud_data.columns)
#predictors = colnames[1:]
#target = colnames[0]

sb.boxplot(fraud_data.City_Population)

sb.boxplot(fraud_data.Work_Experience)

# Basically there are no outliers in the numerical data 
# Now we will plot some graphs for the remaining data and see how it looks.
fraud_data.columns

graph1 = sb.countplot(x = 'Undergrad', data = fraud_data,palette = 'hls')

graph2 = sb.countplot(x = 'Marital_Status', data = fraud_data, palette = 'hls')

graph2 = sb.countplot(x = 'Urban', data = fraud_data, palette = 'hls')
# we can also have a bar plot against the output

pd.crosstab(fraud_data.Marital_Status,fraud_data.output).plot(kind = 'bar')

pd.crosstab(fraud_data.output,fraud_data.Undergrad).plot(kind = 'bar')

pd.crosstab(fraud_data.Urban,fraud_data.output).plot(kind = 'bar')

# These are some of the visuals we got from the graph and we will build our model next based on the decision tree classifier.
# Before the model building we first have to make all the data in to the proper format for the algorithm 
# Since, as per my knowledge the algorithm can't handle the categorical data so we will assign the numerical values to the categories in each column.
# We can also create dummy variables, but in this case I prefer to do it manually since it is much more easy for me.

fraud_data["Undergrad"].value_counts()

fraud_data["Marital_Status"].value_counts()

fraud_data["Urban"].value_counts()

fraud_data["output"].value_counts()

val_replace = {"Undergrad" : {"YES":1,"NO":0}, "Marital_Status" : {"Single":1, "Married":2, "Divorced":3},
               "Urban" : {"YES":1,"NO":0}, "output" : {"Good": 1, "Risky": 0}}

x = fraud_data.replace(val_replace).copy()

x.drop("Taxable_Income",axis = 1, inplace = True) # We will remove the taxable income column which is of no more use to us in the further process.

#since we have changed the main dataset for our model we need to redefine the predictors and target variables.

colnames = list(x.columns)
predictors = colnames[1:]
target = colnames[0]

train,test = train_test_split(x,test_size = 0.2)

model = DecisionTreeClassifier(criterion = 'entropy')

#fitting the model with data

model.fit(train[predictors],train[target].astype(int))

fitted_model = model.fit(train[predictors],train[target].astype(int))

# We will predict the data based on the model which we have built
# We will be passing the test data for the predictions.

model.predict(test[predictors])

# Let's store the predictions in the variable pred
pred = pd.Series(model.predict(test[predictors]))

#lets try to plot the decision tree model for the predicted data.
tree.plot_tree(fitted_model)

type(pred)
pred.value_counts()  # NOTE that '1' stands for 'Good' and '0' stands for 'Risky'

pd.crosstab(test[target],pred)

print('Accuracy', (14+1)/(1+3+2+14)*100)

# Accuracy ---Test and Train

train[target].dtype
type(train[target])


acc = np.mean(pred == train[target].reset_index(drop = True, inplace = True))
acc


#for both the programs related to the decision tree my accuracy is not getting what I expect and I don't know where the problem is 
# so I hope you will point out the issue for both codes and help me improve it any way you can.