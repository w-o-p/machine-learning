import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import string

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import classification_report as clf_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('text/questions_classification.csv')
df.head()


def edit(a):
    a = a.lower()
    t = str.maketrans('', '', string.punctuation)
    a = a.translate(t)
    return a


# text = edit('oprert rr tt yy oprert tt'.lower())
vectoriser = CountVectorizer()
# text_vect = vectoriser.fit_transform([text])
# print(text_vect)

# texts = df['text'].apply(edit)
# print(texts)
clf = RandomForestClassifier()
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.03, train_size=0.07,
                                                    random_state=0)
X_train = vectoriser.fit_transform(X_train)
clf.fit(X_train, y_train)
X_test = vectoriser.transform(X_test)
y_pred = clf.predict(X_test)
print(clf_report(y_test, y_pred))
sklearn.metrics.plot_confusion_matrix(clf, X_test, y_test)
plt.show()
