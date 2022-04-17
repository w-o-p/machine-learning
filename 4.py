
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
    return 2 * x + 1 + random.random() * x


x = np.random.random(100) * 1000
y = [f(n) for n in x]

model = LinearRegression()
print(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#print(X_train, X_test, y_train, y_test)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(MSE(y_test, y_pred))
plt.scatter([i[0] for i in x], y, c='purple')
plt.scatter([i[0] for i in X_test],y_pred, c='white', s=15)
#plt.plot(X_test, y_pred, c='black')
plt.show()