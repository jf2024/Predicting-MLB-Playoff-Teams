import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

#combine both of the files into one file 
#no hyperparameter tuning vs hyperparameter tuning
#get classification report for each of the models with and without hyperparameter tuning
#finding the feature that doesn't help the model
#abalative anayslsis --> just remove the features one at a time  (13-applying ml slide 44 for reference)

#also cook up some graphs to show the difference between the models with and without hyperparameter tuning and maybe get specific teams

# Read the dataset
df = pd.read_csv('baseball.csv')

# Features and Targets
X = df[['RS', 'RA', 'OBP', 'SLG', "OOBP", "OSLG"]]
y = df['Playoffs']

# Split the data into training and test sets
# 75% of the data will be used for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Impute missing values using mean strategy
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_imputed, y_train)
train_score_log_reg = log_reg.score(X_train_imputed, y_train)
test_score_log_reg = log_reg.score(X_test_imputed, y_test)
print("Logistic Regression Training Accuracy:", train_score_log_reg)
print("Logistic Regression Test Accuracy:", test_score_log_reg)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_imputed, y_train)
train_score_rf = rf.score(X_train_imputed, y_train)
test_score_rf = rf.score(X_test_imputed, y_test)
print("Random Forest Training Accuracy:", train_score_rf)
print("Random Forest Test Accuracy:", test_score_rf)

# SVM
svm = SVC()
svm.fit(X_train_imputed, y_train)
train_score_svm = svm.score(X_train_imputed, y_train)
test_score_svm = svm.score(X_test_imputed, y_test)
print("SVM Training Accuracy:", train_score_svm)
print("SVM Test Accuracy:", test_score_svm)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train_imputed, y_train)
train_score_knn = knn.score(X_train_imputed, y_train)
test_score_knn = knn.score(X_test_imputed, y_test)
print("KNN Training Accuracy:", train_score_knn)
print("KNN Test Accuracy:", test_score_knn)
