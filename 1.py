import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import classification_report as clf_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv(r'titanic.csv')
df.head()


def sex_bin(s):
    if s == 'male':
        return 1
    else:
        return 0

def app(a):
    if a == 'S':
        return 1
    elif a == 'Q':
        return 2
    return 3

df['Sex_bin'] = df['Sex'].apply(sex_bin)
df['Embarked'] = df['Embarked'].apply(app)
df = df.dropna(axis=0, subset=['Age', 'Embarked'])
#df['Age'] = df['Age'].fillna(-1)
df.head()

x = ['Sex_bin', 'Age', 'Pclass']
model = DecisionTreeClassifier(min_samples_leaf=3)
X_train, X_test, y_train, y_test = train_test_split(df[x], df['Survived'], test_size=0.2, random_state=10)
model.max_depth = 100
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(clf_report(y_test, y_pred))
#plot_tree(model)
sklearn.metrics.plot_confusion_matrix(model, X_test, y_test)
plt.show()
print(model.feature_importances_)