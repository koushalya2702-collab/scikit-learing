#Mean Absolute Error

from sklearn.metrics import mean_absolute_error

actual=[50,60,70]
prediction=[45,65,68]

print(mean_absolute_error(actual,prediction))


#confusion_matrix

from sklearn.metrics import confusion_matrix

y_true = [1,1,0,0]
y_pred = [1,0,0,0]

print(confusion_matrix(y_true,y_pred))

#Classification Report

from sklearn.metrics import classification_report

y_true = [1,1,0,0]
y_pred = [1,0,0,0]

print(classification_report(y_true,y_pred))

#Data Visualization

import matplotlib.pyplot as plt

X = [1,2,3,4,5]
y = [10,20,30,40,50]

plt.scatter(X,y)

plt.xlabel("Hours")
plt.ylabel("Marks")

plt.show()

#Real CSV Dataset

import pandas as pd

df = pd.read_csv("student.csv")

print(df.head())

