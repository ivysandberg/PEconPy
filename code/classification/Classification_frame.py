import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

# import Data
df = 'path to data'

# note you must have 'tidy' data: each feature is a column & each row is an instance
df.shape    # gives (#instances, #features)

# Exploratory Data Analysis (EDA)
df.head()
df.info()
df.describe()


# Create a plots to examine specific features
plt.figure()
sns.countplot(x='variable', hue='target', data=, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()


# Create arrays for the features and target variable
y = df['target'].values
X = df.drop('target', axis=1).values

''' K Nearest neighbors
Predict the label of a data point by looking at the 'k' closedst
labeled data points and taking a majority vote'''

model = KNeighborsClassifier(n_neighbors=6)
'''larger k=smoother decision boundary=less complex model
smaller k=more complex model=can lead to overfitting

all machine learning models are implemented as Python classes'''

'''Training a model on the data='fitting' a model to the Data
use .fit() method
have the features/data as a numpy array and the target variable as
a numpy array the features must also be continuous values as opposed to
categorical & no missing values'''

model.fit(X, y)

# To predict labels of new data: use .predict() method
model.predict(new_data)


'''To measure model performance, split data into training & test data
 default training data 75% testing data 25%

stratify=y has the training and testing data represent the
same proportion of labels as the original dataset'''

X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.25, random_state=21, stratify=y)

model.fit(X-train, y_train)
y_pred = model.predict(X_test)
model.score(X_test, y_test)

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


# Evaluate level of confidence in the model
print (confusion_matrix(y_test, y_pred))
print (classification_report(y_test, y_pred))
