#Logistic Regression
from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4]]

y = [0, 0, 1, 1]
model=LogisticRegression()
model.fit(X,y)
prediction=model.predict([[6]])
print(prediction)


#Example
from sklearn.metrics import accuracy_score

y_true=[1,5,1,1]
y_pred=[1,1,1,1]
accuracy=accuracy_score(y_true, y_pred)
print(accuracy)

#Example
from sklearn.metrics import accuracy_score
y_true=[1, 1, 1, 1]
y_pred=[0,1,0,1]
accuracy=accuracy_score(y_true,y_pred)
print(accuracy)