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

import pandas as pd

Admission = pd.read_csv('Admission Chance.csv')

Admission # Here Admission is Dataframe of 400 rows/ Students data with 9 columns/ details

Admission.head()

Admission.tail()

Admission.head(7) 

Admission.tail(7) 

Admission.info()

Admission.describe() 

Admission.columns 

Admission.shape


X=Admission.drop(['Serial No','Chance of Admit '],axis=1)
Y=Admission['Chance of Admit ']


from sklearn.model_selection import train_test_split

X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,random_state=2529)



X_Train.shape,X_Test.shape,Y_Train.shape,Y_Test.shape #Y_train and Y_Test are Series


from sklearn.linear_model import LinearRegression
model=LinearRegression()


model.fit(X_Train,Y_Train)

model.intercept_

model.coef_

Y_Pred=model.predict(X_Test)


from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error
mean_absolute_error(Y_Test,Y_Pred)# Accuracy for error is 0.4% so,To become Correct Accuracy is 100-0.4=99.6%

mean_absolute_percentage_error(Y_Test,Y_Pred) #Accuracy for error is 8% , To become Correct Accuracy is 100-8=92%

mean_squared_error(Y_Test,Y_Pred)#Same as mean_absolute_error

"""Conclusion : Chance Of Admit = 94% so that their is 94% chance to get admission in the collage"""
