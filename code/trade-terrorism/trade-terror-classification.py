''' CLASSIFICATION ALGORITHMS -- PREDICT LIKLIHOOD OF FUTURE TERRORIST ACTIVITY '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from pprint import pprint
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


df = pd.read_csv('/Users/ivysandberg/mycode/PEconPy/data/trade-terrorism_classification.csv')
df = pd.DataFrame(df)
print (df.head())

# create new target column with ranges for terrorist activity
terrorism = []

for i in df['trans_terrorism_cj']:
    if i == 0:
        i = 'No events'
        terrorism.append(i)
    elif i > 0 and i <= 10:
        i = 'Few events'
        terrorism.append(i)
    else:
        i = 'Many events'
        terrorism.append(i)

df['terrorism'] = terrorism
print(df['terrorism'])

print(df.head())
print(df.info())

print(df.groupby('terrorism').count())

# examine distribution of target variable (in continuous form)
plt.figure()
sns.countplot(x='trans_terrorism_cj', data=df)
plt.xticks(range(0,10))
plt.show()

# Create arrays for the features and target variable
y = df['terrorism'].values
X = df.drop(['terrorism', 'country2', 'country1'], axis=1).values

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.25, random_state=21, stratify=y)

# K Nearest Neighbors
# Create a for loop to test different values of k
neighbors = np.arange(1,9)
# make an array to story test accuracies
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    test_accuracy[i] = model.score(X_test, y_test)

# plot accuracy scores
plt.title('KNN Vary Value of K (n_neighbors)')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
# neighbors=7 showed the highest accuracy

y_pred = model.predict(X_test)
print (model.score(X_test, y_test))

# Evaluate level of confidence in the model
print (confusion_matrix(y_test, y_pred))
print (classification_report(y_test, y_pred))


# Compare classifiers
# list all the names of the Classifiers
names = [
    'K Nearest Neighbors', 'Gaussian Naive Bayes', 'Multinomial Naive Bayes', 'Bernoulli Naive Bayes',
    'Decision Tree', 'Random Forest', 'AdaBoost', 'Support Vector Machine'
]

classifiers = [
    KNeighborsClassifier(7),
    GaussianNB(),
    MultinomialNB(),
    BernoulliNB(),
    DecisionTreeClassifier(max_depth=3),
    RandomForestClassifier(max_depth=4, random_state=0),
    AdaBoostClassifier(),
    svm.SVC(kernel='rbf', C=1, gamma='auto')
]

score = []

for name, model in zip(names, classifiers):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy_scores = accuracy_score(y_test, pred) * 100
    score.append(accuracy_scores)
    #print (score)

results = dict(zip(names, score))

pprint(results)

# Random Forest Classifier (bootstrap aggreagted decision trees)
RFmodel = RandomForestClassifier(max_depth=4, random_state=0)
RFmodel.fit(X_train, y_train)

# print feature importances, the higher the number the more important
#print(RFmodel.feature_importances_)

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(df.columns, RFmodel.feature_importances_):
    feats[feature] = importance #add the name/value pair

pprint (feats)

plt.hist(y_test)
plt.show()


preds = RFmodel.predict(X_test)

cv_scores = cross_val_score(RFmodel, X, y, cv=5)
print (cv_scores)

X_test_df = pd.DataFrame(X_test)
X_test_df['preds'] = preds

print (X_test_df.head())
print (X_test_df.groupby("preds").count())

print (X_test_df.loc[X_test_df['preds'] == "Many events"])

plt.scatter(df['exp_index'], df['trans_terrorism_cj'])
plt.show()
