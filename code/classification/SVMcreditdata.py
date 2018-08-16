import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

# import data
path = '/Users/ivysandberg/mycode/PEconPy/data/default of credit card clients.csv'
data = pd.read_csv(path, header=1)
data = pd.DataFrame(data)


# assign the design array - attributes excluding ID & target
X = np.array(data.iloc[:, 1:23])
# assign target variable as a numpy array
y = np.array(data.iloc[:, 24]) # target = default payment next month(0/1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                    random_state=42)

model = svm.SVC(kernel='rbf', C=1, gamma='auto')

model.fit(X_train, y_train)
pred = model.predict(X_test)
print (pred)
print (accuracy_score(y_test, pred) * 100)
