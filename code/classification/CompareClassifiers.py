''' a piece of code to iterate over classifiers '''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pprint import pprint

from yapf.yapflib.yapf_api import FormatFile
FormatFile(
    "/Users/ivysandberg/mycode/PEconPy/code/classification/CompareClassifiers.py",
    in_place=True)

# list all the names of the Classifiers
names = [
    'K Nearest Neighbors', 'Gaussian Naive Bayes', 'Bernoulli Naive Bayes',
    'Decision Tree', 'Random Forest', 'AdaBoost', 'Support Vector Machine'
]

classifiers = [
    KNeighborsClassifier(3),
    GaussianNB(),
    BernoulliNB(),
    DecisionTreeClassifier(max_depth=3),
    RandomForestClassifier(max_depth=4, random_state=0),
    AdaBoostClassifier(),
    svm.SVC(kernel='rbf', C=1, gamma='auto')
]

# import data
path = '/Users/ivysandberg/mycode/PEconPy/data/default of credit card clients.csv'
data = pd.read_csv(path, header=1)
data = pd.DataFrame(data)

# assign the design array - attributes excluding ID & target
X = np.array(data.iloc[:, 1:23])
# assign target variable as a numpy array
y = np.array(data.iloc[:, 24])  # target = default payment next month(0/1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

score = []

for name, model in zip(names, classifiers):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy_scores = accuracy_score(y_test, pred) * 100
    score.append(accuracy_scores)
    #print (score)

results = dict(zip(names, score))
#print (results)

pprint(results)
