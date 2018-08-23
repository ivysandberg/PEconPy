# PEconPy
python classification algorithms on economic data as examples.

This repo is divided by the python (3.6) code and the datasets. The code is divided into files showing how to implement classification algorithms and a more in depth economic study into the relationship between trade & terrorism.


code/classification
Entails a bare python file drawing out a starting Classification framework for general best practices. Followed by a series of 'how-to' implementation examples of classification algorithms (K Nearest Neighbors, Naive Bayes, Decision Trees, Support Vector Machines) using credit default data as an example. Finally, CompareClassifiers shows how to loop all the classifiers demonstrated here for comparison.

for more information on each of the algorithms explored here: https://docs.google.com/document/d/1vjbHKgNyfhAkRXsfoe0Gs3DkdW_wPKLxpPAkMKu4eGA/edit?usp=sharing

code/trade-terrorism
trade-terror.py contains exploratory data analysis on the raw data.
trade-terror.py explores the classification problem presented in the prepped trade-terrorism data.


data/
'default of credit card clients' data is used in all the classification algorithm framework examples.
'trade-terrorism.csv' is the data as is after converting it from a .dta stata file. The data on trade is from the UNCTAD (United Nations Conference of Trade & Development). The data on terrorism is from GTD (Global Terrorism Database from the National Consortium for the Study for Terrorism & Responses to Terrorism)
'trade-terrorism-classification.csv' is the data after is has been cleaned of all missing values and prepped to be used to predict the amount of transnational terrorism of a trading country with the United States.


libraries:
pandas
numpy
matplotlib
sklearn
