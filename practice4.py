#Project 1 — Student Marks Prediction (Linear Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [20, 30, 40, 50, 60, 70, 80, 90]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model=LinearRegression()
model.fit(X_train, y_train)

prediction=model.predict([[9]])
print("Predicted Marks:", prediction)

print("Model Score:",model.score(X_test,y_test))