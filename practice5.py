#Project 2 — Pass/Fail Prediction (Logistic Regression)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 1, 1, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model=LogisticRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)
print(X_test)

print("Predictions:", prediction)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", accuracy)