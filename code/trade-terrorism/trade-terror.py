''' DATA ANALYSIS TO EXPLORE THE RELATIONSHIPS BETWEEN TRADE & TERRORISM '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from pprint import pprint

from yapf.yapflib.yapf_api import FormatFile
FormatFile(
    "/Users/ivysandberg/mycode/PEconPy/code/trade-terrorism/trade-terror.py",
    in_place=True)
''' IMPORT DATA '''

data = pd.io.stata.read_stata(
    '/Users/ivysandberg/Downloads/trade-terrorism.dta.dta')
data.to_csv('/Users/ivysandberg/Desktop/trade-terrorism.csv')
data = pd.DataFrame(data)
''' RAW DATA EXPLORATION '''

print(data.info())
print(data.head())
print(data.shape)
print(data.describe())
print(data.columns)

print(data['iex_v1'].describe())
print(data.groupby('country1').count())

print(data.loc[(data.country1 == 'France')])
print(data.loc[(data.country1 == 'Cuba')].groupby('year').count())

print(data.loc[(data.country2 == 'Libya')])

#plt.scatter(data['year'], data['incdr']+data['c2incdr']+data['inctr']+data['c2inctr'])
#plt.show()

#plt.scatter(data['year'], data['iex_v1'])
#plt.show()
''' EXPLORATORY & TEMPORAL DATA ANALYSIS OF CLEAN DATA '''

df = pd.read_csv(
    '/Users/ivysandberg/Desktop/trade-terrorism–clean.csv', header=0)
df = pd.DataFrame(df)
print(df.head())
columns = df.columns
print(columns)

df.country2 = df.country2.astype("category")
Libya = df.loc[df.country2 == "Libya"]

#plt.scatter(Libya['year'], Libya['TOT_IMPORTS'])
#plt.xlabel('Year')
#plt.ylabel('Total imports')
#plt.show()

terror_year = df.groupby('year').count()
terror_year = pd.DataFrame(terror_year)
print(terror_year)

# Explore temporal trends
plt.scatter(df['year'], df['DOM_TERROR_cj'] + df['TRANS_TERROR_cj'])
plt.title('Terrorism Events')
plt.xlabel('Year')
plt.ylabel('Total terrorism events')
#plt.show()

plt.scatter(df['year'], df['TOT_EXPORTS'])
plt.title('US Exports')
plt.xlabel('Year')
plt.ylabel('Total Exports')
#plt.show()

plt.scatter(df['year'], df['TOT_IMPORTS'])
plt.title('US Imports')
plt.xlabel('Year')
plt.ylabel('Total Imports')
#plt.show()
''' REGRESSION ANALYSIS '''

# define target (y)
y = df['TOT_IMPORTS'].values
X = np.array(df.loc[:, [
    'border', 'comlang', 'Currency_Union', 'landl', 'log_dist', 'log_RGDP',
    'log_RGDP_percap', 'regional_tradeagreement', 'TOT_EXPORTS',
    'DOM_TERROR_cj', 'TRANS_TERROR_cj'
]])

model_features = [
    'border', 'comlang', 'Currency_Union', 'landl', 'log_dist', 'log_RGDP',
    'log_RGDP_percap', 'regional_tradeagreement', 'TOT_EXPORTS',
    'DOM_TERROR_cj', 'TRANS_TERROR_cj'
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=20)

model = RandomForestRegressor(max_depth=2, random_state=0)
model.fit(X_train, y_train)

# print feature importances, the higher the number the more important
print(model.feature_importances_)

#print (model.predict(X_test))
print(model.score(X_test, y_test, sample_weight=None))

# Linear Regression
lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

# Compute R squared
R2 = lm.score(X_test, y_test)
print("R^2: {}".format(R2))

# Estimate coefficients
coefficients = lm.coef_
print('Coefficients: ', lm.coef_)

pprint(dict(zip(model_features, coefficients)))
