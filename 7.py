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

df = pd.read_csv('Iris-Dataset/iris.data',
                 names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'], index_col=False)
df.head()

a = ['sepal length', 'sepal width', 'petal length', 'petal width']
model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(df[a], df['class'], test_size=0.2, random_state=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(clf_report(y_test, y_pred))
plot_tree(model)
sklearn.metrics.plot_confusion_matrix(model, X_test, y_test)
model.feature_importances_
plt.show()
