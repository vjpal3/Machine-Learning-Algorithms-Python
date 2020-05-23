# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('PS_20174392719_1491204439457_log.csv')
X = dataset.iloc[:, [2, 4, 5, 7, 8]].values
y = dataset.iloc[:, 9].values

print(dataset.shape)
print(dataset['isFraud'].value_counts())

#Check null values
print(dataset.isnull().values.any())

y_target = dataset['isFraud']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify=y_target)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

"""score = classifier.score(X_test, y_test)
print(score)"""

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
"""print(cm)"""

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

from sklearn import metrics
print("Accuracy:     ", round(metrics.accuracy_score(y_test, y_pred),4)*100)
print("Precision:    ", round(metrics.precision_score(y_test, y_pred),4)*100)
print("Recall:       ", round(metrics.recall_score(y_test, y_pred),4)*100)
