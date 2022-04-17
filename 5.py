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


def func(x):
    return (x - 5) ** 2 + random.random() * x


def func_derivative(x):
    return 2 * (x - 5)


x = np.arange(-5, 10, 0.1)
# x = [num for num in range(-90, 100)]
y = [func(n) for n in x]
plt.plot(x, y)

steps = 10
step_size = 0.2
# current_x = previous_x - func_derivative(previous_x)*step_size
previous_x = -10
steps_x = []
for i in range(steps):
    current_x = previous_x - func_derivative(previous_x) * step_size
    steps_x.append(current_x)
    previous_x = current_x
steps_y = [func(i) for i in steps_x]
# plt.plot(steps_x, steps_y, c='red')
# plt.scatter(steps_x, steps_y, c='black')
print(previous_x)
print(func(previous_x))
plt.show()
