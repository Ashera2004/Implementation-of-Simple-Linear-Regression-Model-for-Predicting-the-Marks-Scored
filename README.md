# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare Data

   • Import necessary libraries (pandas, numpy, matplotlib, sklearn).
   
    • Load student_scores.csv and extract values of independent variable X and dependent variable Y.
   
3. Split Data into Training and Testing Sets

    • Use train_test_split() to divide data (2/3 training, 1/3 testing).
   
5. Train Linear Regression Model

    • Initialize LinearRegression().

   • Fit the model using X_train and Y_train.

7. Make Predictions and Evaluate Model

   • Predict scores using X_test.

   • compute MSE, MAE, and RMSE to assess performance.

9. Visualize Results

   • Plot training data with a scatter plot.

    • Draw the best-fit regression line.

## Program:

```
##Program to implement the simple linear regression model for predicting the marks scored.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')

#displaying the content in datafile
df.head()
df.tail()

#Segregating data to variables
X = df.iloc[:,:-1].values
X
Y=df.iloc[:,-1].values
Y

#splitting your data and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred

mse=mean_squared_error(Y_test,Y_pred)
print("MSE=",mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE=",mae)

rmse=np.sqrt(mse)
print("RMSE=",rmse)

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

Developed by: A S Siddarth
RegisterNumber: 212224040316

```

## Output:
![image](https://github.com/Ashera2004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/fe058a77738d4e74fe00bef77cbc4f3644e5e65a/graph_ex2.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
