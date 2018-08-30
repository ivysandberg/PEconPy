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
from scipy.stats.stats import pearsonr

from yapf.yapflib.yapf_api import FormatFile
FormatFile(
    "/Users/ivysandberg/mycode/PEconPy/code/trade-terrorism/trade-terror-classification.py",
    in_place=True)

df = pd.read_csv(
    '/Users/ivysandberg/mycode/PEconPy/data/trade-terrorism_classification.csv'
)
df = pd.DataFrame(df)
print("Original Data:")
print(df.head())
''' DEFINE NEW TARGET VARIABLE: LEVEL OF TERRORISM '''

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
print("New Terrorism Target variable:")
print(df['terrorism'])

print("Data including the new target variable:")
print(df.head())
print(df.info())

print("Total counts for each category of terrorism:")
print(df.groupby('terrorism').count())

# examine distribution of target variable (in continuous form)
plt.figure()
sns.countplot(x='trans_terrorism_cj', data=df)
plt.xticks(range(0, 15))
plt.xlabel("Number of transnational terrorist events")
#plt.show()

# Create arrays for the features and target variable
y = df['terrorism'].values
X = df.drop(
    ['terrorism', 'trans_terrorism_cj', 'country1', 'country2'], axis=1).values
X_df = pd.DataFrame(X)
print("This is the training data:")
print(X_df.head())
X_df.columns = [
    'year', 'border', 'comlang', 'Currency_union', 'island', ',landl',
    'log_dist', 'log_RGDP', 'log_RGDP_percap', 'regional_trade_agreement',
    'total_exports', 'primary_exports', 'manfactured_exports', 'total_imports',
    'primary_imports', 'manfactured_imports', 'exp_index', 'imp_index'
]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21, stratify=y)
''' K Nearest Neighbors '''

# Create a for loop to test different values of k
neighbors = np.arange(1, 9)
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
#plt.show()
# neighbors=7 showed the highest accuracy

y_pred = model.predict(X_test)
print("KNN accuracy score:")
print(model.score(X_test, y_test))

# Evaluate level of confidence in the model
print(confusion_matrix(y_test, y_pred))
print("KNN Classification report:")
print(classification_report(y_test, y_pred))
''' Compare classifiers '''

# list all the names of the Classifiers
names = [
    'K Nearest Neighbors', 'Gaussian Naive Bayes', 'Multinomial Naive Bayes',
    'Bernoulli Naive Bayes', 'Decision Tree', 'Random Forest', 'AdaBoost',
    'Support Vector Machine'
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

print(
    "Classification algorithms accuracy scores on prediciting terrorism levels:"
)
pprint(results)

# Random Forest Classifier (bootstrap aggreagted decision trees)
RFmodel = RandomForestClassifier(max_depth=4, random_state=0)
RFmodel.fit(X_train, y_train)

cv_scores = cross_val_score(RFmodel, X, y, cv=5)
print("RandomForestClassifier Cross Validation Scores:")
print(cv_scores)

# Explore the best performing model
ABmodel = AdaBoostClassifier()
ABmodel.fit(X_train, y_train)

# print feature importances, the higher the number the more important

feats = {}  # a dict to hold feature_name: feature_importance
for feature, importance in zip(X_df.columns, ABmodel.feature_importances_):
    feats[feature] = importance  #add the name/value pair

print("AdaBoost Feature Importances:")
pprint(feats)

plt.hist(y_test)
#plt.show()

plt.scatter(df['primary_exports'], df['trans_terrorism_cj'])
#plt.show()

new_data = pd.read_csv(
    '/Users/ivysandberg/Desktop/USxAustria_deltaExportValue.csv')
#new_pred = ABmodel.predict(new_data)
#print(new_pred)

print(
    "Pearson's correlation coefficient, 2-tailed p-value between primary exports and trans terrorism in country j",
    pearsonr(df['primary_exports'], df['trans_terrorism_cj']))
