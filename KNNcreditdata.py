import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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
model = KNeighborsClassifier(n_neighbors=6)

# fitting the model
model.fit(X_train, y_train)

# predict the response
pred = model.predict(X_test)

# evaluate accuracy
print (y_test, pred)
print(accuracy_score(y_test, pred))
