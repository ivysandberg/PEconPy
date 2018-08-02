import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


path = '/Users/ivysandberg/mycode/PEconPy/default of credit card clients.csv'
data = pd.read_csv(path, header=1)
data = pd.DataFrame(data)


# assign the design array - attributes excluding ID & target
X = np.array(data.iloc[:, 1:23])
# assign target variable as a numpy array
y = np.array(data.iloc[:, 24]) # target = default payment next month(0/1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                    random_state=42)

# instantiate learning model
model = tree.DecisionTreeClassifier(max_depth=3)

# originally had max_dept=None & accuracy score=0.723
# now with max_depth=3, accuracy score = 0.82

# fitting the model
model.fit(X_train, y_train)

# predict the response
pred = model.predict(X_test)

prob = model.predict_proba(X_test)

# evaluate accuracy
print (y_test, pred)
print (prob)
print(accuracy_score(y_test, pred) * 100)

from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_test, pred))

from sklearn.metrics import classification_report
print (classification_report(y_test, pred))


# Now try ensemble methods of decision trees

# Random Forest Classifier (bootstrap aggreagted decision trees)
from sklearn.ensemble import RandomForestClassifier
RFmodel = RandomForestClassifier(max_depth=4, random_state=0)
RFmodel.fit(X_train, y_train)

# print feature importances, the higher the number the more important
print(RFmodel.feature_importances_)

# print Random Forest prediction accuracy score
RFpred = RFmodel.predict(X_test)
print (accuracy_score(y_test, RFpred) * 100)


from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(y_test, RFpred))
print(cm)



# AdaBoost (Boosted Tree)
from sklearn.ensemble import AdaBoostClassifier
ABmodel = AdaBoostClassifier()
ABmodel.fit(X_train, y_train)
ABpred = ABmodel.predict(X_test)
print(accuracy_score(y_test, ABpred) * 100)


# Compare Decision Tree, Random Forest, AdaBoost
DTtest = accuracy_score(y_test, pred) * 100
RFtest = accuracy_score(y_test, RFpred) * 100
ABtest = accuracy_score(y_test, ABpred) * 100

print ("Prediction Accuracy Scores:")
print ("Decision Tree: ", DTtest)
print ("Random Forest: ", RFtest)
print ("AdaBoost: ", ABtest)
