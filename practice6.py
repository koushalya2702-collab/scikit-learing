#Project 3 — Salary Prediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5], [6]]
y = [25000, 30000, 40000, 50000, 60000, 70000]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model=LinearRegression()

model.fit(X_train,y_train)

prediction=model.predict([[7]])

print("Prediction Salary:",prediction)

print("Score:",model.score(X_test,y_test))