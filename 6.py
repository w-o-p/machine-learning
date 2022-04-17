import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import random
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import classification_report as clf_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, plot_tree

def f(x):
    return (x - 5) ** 2 + random.random() * x

def func(x, a, b, c):
    return (x - 5) ** 2 + random.random() + a + b - c


def func_derivative(x):
    return 2 * (x - 5)


x = np.arange(-5, 100, 0.1)
a = np.random.random(len(x)) * 10
b = np.random.random(len(x)) * 10
c = np.random.random(len(x)) * 10

# x = [num for num in range(-90, 100)]
y = [func(x[n], a[n], b[n], c[n]) for n in range(len(x))]
# y = func(x)
plt.plot(x, y)

steps = 10
step_size = 0.2
# current_x = previous_x - func_derivative(previous_x)*step_size
previous_x = 0
steps_x = []
for i in range(steps):
    current_x = previous_x - func_derivative(previous_x) * step_size
    steps_x.append(current_x)
    previous_x = current_x
steps_y = [f(i) for i in steps_x]
# steps_y = [func(i) for i in steps_x]
plt.plot(steps_x, steps_y, c='red')
plt.scatter(steps_x, steps_y, c='black')

model = SGDRegressor(max_iter=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# print(X_train, X_test, y_train, y_test)
model.fit(X_train.reshape(-1, 1), y_train)
y_pred = model.predict(X_test.reshape(-1, 1))
print(MSE(y_test, y_pred))
# plt.scatter(x,y, c='green')
plt.scatter(X_test, y_pred, c='red')

plt.show()
