import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Read the dataset
df = pd.read_csv('baseball.csv')

# Features and Targets
X = df[['RS', 'RA', 'OBP', 'SLG', 'OOBP', 'OSLG']]
y = df['Playoffs']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Impute missing values using mean strategy
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Models without Hyperparameters

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_imputed, y_train)
train_score_log_reg = log_reg.score(X_train_imputed, y_train)
test_score_log_reg = log_reg.score(X_test_imputed, y_test)
print("Logistic Regression Training Accuracy (No Hyperparameter Tuning):", train_score_log_reg)
print("Logistic Regression Test Accuracy (No Hyperparameter Tuning):", test_score_log_reg)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_imputed, y_train)
train_score_rf = rf.score(X_train_imputed, y_train)
test_score_rf = rf.score(X_test_imputed, y_test)
print("\nRandom Forest Training Accuracy (No Hyperparameter Tuning):", train_score_rf)
print("Random Forest Test Accuracy (No Hyperparameter Tuning):", test_score_rf)

# SVM
svm = SVC()
svm.fit(X_train_imputed, y_train)
train_score_svm = svm.score(X_train_imputed, y_train)
test_score_svm = svm.score(X_test_imputed, y_test)
print("\nSVM Training Accuracy (No Hyperparameter Tuning):", train_score_svm)
print("SVM Test Accuracy (No Hyperparameter Tuning):", test_score_svm)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train_imputed, y_train)
train_score_knn = knn.score(X_train_imputed, y_train)
test_score_knn = knn.score(X_test_imputed, y_test)
print("\nKNN Training Accuracy (No Hyperparameter Tuning):", train_score_knn)
print("KNN Test Accuracy (No Hyperparameter Tuning):", test_score_knn)

# Models with Hyperparameters
print("------------------------------------------------------------------------------")

# Logistic Regression with Hyperparameter Tuning
log_reg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
log_reg_tuned = LogisticRegression()
log_reg_grid = GridSearchCV(log_reg_tuned, log_reg_params, cv=5)
log_reg_grid.fit(X_train_imputed, y_train)
print("\nLogistic Regression Best Parameters:", log_reg_grid.best_params_)
print("Logistic Regression Training Accuracy (Hyperparameter Tuning):", log_reg_grid.best_score_)
print("Logistic Regression Test Accuracy (Hyperparameter Tuning):", log_reg_grid.score(X_test_imputed, y_test))

# Random Forest with Hyperparameter Tuning
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
rf_tuned = RandomForestClassifier()
rf_grid = GridSearchCV(rf_tuned, rf_params, cv=5)
rf_grid.fit(X_train_imputed, y_train)
print("\nRandom Forest Best Parameters:", rf_grid.best_params_)
print("Random Forest Training Accuracy (Hyperparameter Tuning):", rf_grid.best_score_)
print("Random Forest Test Accuracy (Hyperparameter Tuning):", rf_grid.score(X_test_imputed, y_test))

# SVM with Hyperparameter Tuning
svm_params = {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_tuned = SVC()
svm_grid = GridSearchCV(svm_tuned, svm_params, cv=5)
svm_grid.fit(X_train_imputed, y_train)
print("\nSVM Best Parameters:", svm_grid.best_params_)
print("SVM Training Accuracy (Hyperparameter Tuning):", svm_grid.best_score_)
print("SVM Test Accuracy (Hyperparameter Tuning):", svm_grid.score(X_test_imputed, y_test))

# KNN with Hyperparameter Tuning
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn_tuned = KNeighborsClassifier()
knn_grid = GridSearchCV(knn_tuned, knn_params, cv=5)
knn_grid.fit(X_train_imputed, y_train)
print("\nKNN Best Parameters:", knn_grid.best_params_)
print("KNN Training Accuracy (Hyperparameter Tuning):", knn_grid.best_score_)
print("KNN Test Accuracy (Hyperparameter Tuning):", knn_grid.score(X_test_imputed, y_test))

# Ablative Analysis on Features

# Define a function to perform ablative analysis
def ablative_analysis(model, features, X_train, X_test, y_train, y_test):
    scores = {}
    for feature in features:
        reduced_X_train = X_train.drop(columns=[feature])
        reduced_X_test = X_test.drop(columns=[feature])

        model.fit(reduced_X_train, y_train)
        test_score = model.score(reduced_X_test, y_test)
        scores[feature] = test_score

    return scores

# Convert NumPy arrays back to DataFrames
X_train_df = pd.DataFrame(X_train_imputed, columns=X.columns)
X_test_df = pd.DataFrame(X_test_imputed, columns=X.columns)

# Perform ablative analysis for each model (no hyperparameter tuning)
ablative_scores_log_reg = ablative_analysis(LogisticRegression(), X.columns, X_train_df, X_test_df, y_train, y_test)
ablative_scores_rf = ablative_analysis(RandomForestClassifier(n_estimators=100), X.columns, X_train_df, X_test_df, y_train, y_test)
ablative_scores_svm = ablative_analysis(SVC(), X.columns, X_train_df, X_test_df, y_train, y_test)
ablative_scores_knn = ablative_analysis(KNeighborsClassifier(), X.columns, X_train_df, X_test_df, y_train, y_test)

# Print ablative analysis results
print("\nAblative Analysis Results (No hyperparameters):")
print("Logistic Regression:", ablative_scores_log_reg)
print("Random Forest:", ablative_scores_rf)
print("SVM:", ablative_scores_svm)
print("KNN:", ablative_scores_knn)


# Perform ablative analysis for each hyperparameter-tuned model
ablative_scores_log_reg_tuned = ablative_analysis(log_reg_grid.best_estimator_, X.columns, X_train_df, X_test_df, y_train, y_test)
ablative_scores_rf_tuned = ablative_analysis(rf_grid.best_estimator_, X.columns, X_train_df, X_test_df, y_train, y_test)
ablative_scores_svm_tuned = ablative_analysis(svm_grid.best_estimator_, X.columns, X_train_df, X_test_df, y_train, y_test)
ablative_scores_knn_tuned = ablative_analysis(knn_grid.best_estimator_, X.columns, X_train_df, X_test_df, y_train, y_test)


# Print ablative analysis results for hyperparameter-tuned models
print("\nAblative Analysis Results (Hyperparameter-Tuned Models):")
print("Logistic Regression:", ablative_scores_log_reg_tuned)
print("Random Forest:", ablative_scores_rf_tuned)
print("SVM:", ablative_scores_svm_tuned)
print("KNN:", ablative_scores_knn_tuned)
