#Mini Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df=pd.read_csv("student.csv")

X = df[["Hours"]]
y = df["Marks"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LinearRegression()

model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("Predictions:",prediction)

print("Score:",model.score(X_test,y_test))

print("MAE:",mean_absolute_error(y_test,prediction))