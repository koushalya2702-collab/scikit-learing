#Example
from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5]]

y = [10, 20, 30, 40, 50]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train)
print(X_test)


print(y_train)
print(y_test)



#Example
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]

y = [10, 20, 30, 40, 50]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

model=LinearRegression()
model.fit(X_train,y_train)

prediction = model.predict([[6]])
print("Prediction:", prediction)

score=model.score(X_test,y_test)
print("Score:",score)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
