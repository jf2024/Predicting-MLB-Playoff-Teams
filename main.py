import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Read the dataset
df = pd.read_csv('baseball.csv')

# Histogram of runs scored 
plt.hist(df['RS'])
plt.title('Distribution of Runs Scored')
plt.xlabel('Runs Scored')
plt.ylabel('Count') 
plt.show()

# Scatter plot runs scored vs runs against
plt.scatter(df['RS'], df['RA'])
plt.title('Runs Scored vs Runs Against')
plt.xlabel('Runs Scored')
plt.ylabel('Runs Against')
plt.show()

# Features and Targets
X = df[['RS', 'RA', 'OBP', 'SLG']] 
y = df['Playoffs']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Print the number of samples in the training and testing datasets
print("Number of samples in training set:", X_train.shape[0])
print("Number of samples in testing set:", X_test.shape[0])

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
train_score_log_reg = log_reg.score(X_train, y_train)
test_score_log_reg = log_reg.score(X_test, y_test)
print("Logistic Regression Training Accuracy:", train_score_log_reg)
print("Logistic Regression Test Accuracy:", test_score_log_reg)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train) 
train_score_rf = rf.score(X_train, y_train)
test_score_rf = rf.score(X_test, y_test)
print("Random Forest Training Accuracy:", train_score_rf)
print("Random Forest Test Accuracy:", test_score_rf)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
train_score_svm = svm.score(X_train, y_train)
test_score_svm = svm.score(X_test, y_test)
print("SVM Training Accuracy:", train_score_svm)
print("SVM Test Accuracy:", test_score_svm)

# KNN
knn = KNeighborsClassifier()  
knn.fit(X_train, y_train)
train_score_knn = knn.score(X_train, y_train)
test_score_knn = knn.score(X_test, y_test)
print("KNN Training Accuracy:", train_score_knn)
print("KNN Test Accuracy:", test_score_knn)
