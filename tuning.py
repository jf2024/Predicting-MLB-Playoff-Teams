import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

#getting (accuracy, f1score, auroc, precision, sensitivity, specificity) --> use classification report
from sklearn.metrics import classification_report, confusion_matrix

# explain the different hyperparameters for each model
# explain what gridsearchcv does
# show classification report for each model and put it on slides

# Read the dataset
df = pd.read_csv('baseball.csv')

# Features and Targets
X = df[['RS', 'RA', 'OBP', 'SLG', 'OOBP', 'OSLG']]
y = df['Playoffs']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Impute missing values in both training and test sets
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Logistic Regression with Hyperparameter Tuning
log_reg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
log_reg_tuned = LogisticRegression()
log_reg_grid = GridSearchCV(log_reg_tuned, log_reg_params, cv=5)
log_reg_grid.fit(X_train_imputed, y_train)
print("Logistic Regression Best Parameters:", log_reg_grid.best_params_)
print("Logistic Regression Training Accuracy (Hyperparameter Tuning):", log_reg_grid.best_score_)
print("Logistic Regression Test Accuracy (Hyperparameter Tuning):", log_reg_grid.score(X_test_imputed, y_test))

# Random Forest with Hyperparameter Tuning
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
rf_tuned = RandomForestClassifier()
rf_grid = GridSearchCV(rf_tuned, rf_params, cv=5)
rf_grid.fit(X_train_imputed, y_train)
print("Random Forest Best Parameters:", rf_grid.best_params_)
print("Random Forest Training Accuracy (Hyperparameter Tuning):", rf_grid.best_score_)
print("Random Forest Test Accuracy (Hyperparameter Tuning):", rf_grid.score(X_test_imputed, y_test))

# SVM with Hyperparameter Tuning
svm_params = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_tuned = SVC()
svm_grid = GridSearchCV(svm_tuned, svm_params, cv=5)
svm_grid.fit(X_train_imputed, y_train)
print("SVM Best Parameters:", svm_grid.best_params_)
print("SVM Training Accuracy (Hyperparameter Tuning):", svm_grid.best_score_)
print("SVM Test Accuracy (Hyperparameter Tuning):", svm_grid.score(X_test_imputed, y_test))

# Classification Report for svm with hyperparameter tuning
print(classification_report(y_test, svm_grid.predict(X_test_imputed)))

# KNN with Hyperparameter Tuning
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn_tuned = KNeighborsClassifier()
knn_grid = GridSearchCV(knn_tuned, knn_params, cv=5)
knn_grid.fit(X_train_imputed, y_train)
print("KNN Best Parameters:", knn_grid.best_params_)
print("KNN Training Accuracy (Hyperparameter Tuning):", knn_grid.best_score_)
print("KNN Test Accuracy (Hyperparameter Tuning):", knn_grid.score(X_test_imputed, y_test))
