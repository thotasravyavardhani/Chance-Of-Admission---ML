# -*- coding: utf-8 -*-
"""Chance Of Admission -ML project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1w8C8sVwAoeomHHWVTdFNDPYuWKKB4_KN

Chance Of Admission Prediction - Machine learning Project

Chance of Admission for Higher Studies
Predict the chances of admission of a student to a Graduate program based on:

1. GRE Scores (290 to 340)
2. TOEFL Scores (92 to 120)
3. University Rating (1 to 5)
4. Statement of Purpose (1 to 5)
5. Letter of Recommendation Strength (1 to 5)
6. Undergraduate CGPA (6.8 to 9.92)
7. Research Experience (0 or 1)
8. Chance of Admit (0.34 to 0.97)

Steps to create a ML model:

1. import library
2. import data
3. Select X and Y (Dependent or o/p variable(Y) and Independent or i/p
   variable(X)).
4. Train Test Split: Splitting the data into Train data and Test data Train data is used to Train the ML model and test data is used to by taking random samples from dataset.
5. Select model : based on Y- o/p variable type
   if y = Continuous data - Regression models
   if y = Categorical data (0 / 1,T/ F)- Classification models
  
  Regression models:

    a. if y depends only on one of the other Variable (x)
      then eq-> y=mx+c
    b. if y depends on multiple then eq-> y=b0+b1x1+b2x2+....
    use LINEAR REGRESSION MODEL Not only these their can be many other chance

  Classification models:
     
     Clustering, Naive Buyes....
6. Training or fitting the model
7. Testing or predicting the model
8. Accuracy check
"""

#step1 : Import library
import pandas as pd

Admission = pd.read_csv('Admission Chance.csv')

Admission # Here Admission is Dataframe of 400 rows/ Students data with 9 columns/ details

Admission.head()#Top 5 elements

Admission.tail()#Last 5 elements

Admission.head(7) # For first n rows --> Dataset.head(n)

Admission.tail(7) # For last n rows --> Dataset.tail(n)

Admission.info()# info() - Explains about dataset no of rows colns, datatypes of columns and other...

Admission.describe() #D.describe()-- gives the Aggregate statistics for each coln, like...count,mean,std....

Admission.columns #This row gives column names in the dataset

Admission.shape # Dimensions--(rows,columns) of dataframe or series will be given.
#Note: if y=1, and dataset= Dataframe = (400,1) if y=1, and dataset= Series =(400,)

#splitting into X and Y
X=Admission.drop(['Serial No','Chance of Admit '],axis=1)
Y=Admission['Chance of Admit ']

#Splitting the data into Train data(X_Train,Y_Train) and Test data(X_test,Y_test)
#Always Y_Test and y_Train are series only single coln o/p data
from sklearn.model_selection import train_test_split

X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,random_state=2529)# default ---> train_size=0.75 or 75%
#default ---> test_size= 0.25 /25% (25% data taken from entire data set to test the model that data is given to model)
#random_state ---> to take random sets

#To test size of these X_Train,X_Test,Y_Train,Y_Test
X_Train.shape,X_Test.shape,Y_Train.shape,Y_Test.shape #Y_train and Y_Test are Series

#Select the model - Here y= Chance Of Admit = label / y data is 'Continuous' - so Regression problem
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#To train model
model.fit(X_Train,Y_Train)

model.intercept_# To fint intercept value for eq: y=b0+b1x1+b2x2+.... --> b1,b2,b3...=slope ,b0= intercept

model.coef_#To find slopes/b1,b2,b3... from y=b0+b1x1+b2x2+...

#To test or predict the model
#To test we only give X_Test data to test and store the Y values from testing stored in Y_pred so that y matching those y_test values based onthat accuracy is calculated
Y_Pred=model.predict(X_Test)

#To find accuracy
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error
mean_absolute_error(Y_Test,Y_Pred)# Accuracy for error is 0.4% so,To become Correct Accuracy is 100-0.4=99.6%

mean_absolute_percentage_error(Y_Test,Y_Pred) #Accuracy for error is 8% , To become Correct Accuracy is 100-8=92%

mean_squared_error(Y_Test,Y_Pred)#Same as mean_absolute_error

"""Conclusion : Chance Of Admit = 94% so that their is 94% chance to get admission in the collage"""