# this will be our final submission for the project
# just need to write descriptions for each of the functions 
#change name later 

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def read_dataset(file_path):
    return pd.read_csv(file_path)

def split_data(X, y, test_size=0.25):
    return train_test_split(X, y, test_size=test_size)

def impute_missing_values(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    return train_score, test_score, y_pred

def perform_hyperparameter_tuning(model, param_grid, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    test_score = grid_search.score(X_test, y_test)
    y_pred = grid_search.predict(X_test)
    return best_params, best_score, test_score, y_pred

def ablative_analysis(model, features, X_train, X_test, y_train, y_test):
    scores = {}
    imputer = SimpleImputer(strategy='mean')
    
    for feature in features:
        reduced_X_train = X_train.drop(columns=[feature])
        reduced_X_test = X_test.drop(columns=[feature])

        reduced_X_train_imputed = imputer.fit_transform(reduced_X_train)
        reduced_X_test_imputed = imputer.transform(reduced_X_test)

        model.fit(reduced_X_train_imputed, y_train)
        test_score = model.score(reduced_X_test_imputed, y_test)
        scores[feature] = test_score

    return scores

def plot_playoff_counts(actual_playoff_counts, predicted_playoff_counts, model_name):
    plt.figure(figsize=(12, 6))

    # Actual Playoff Appearances
    plt.subplot(1, 2, 1)
    sns.barplot(x='Actual_Playoff_Appearances', y='Team', data=actual_playoff_counts.sort_values(by='Actual_Playoff_Appearances', ascending=False))
    plt.title(f'{model_name} Actual Playoff Appearances')

    # Predicted Playoff Appearances
    plt.subplot(1, 2, 2)
    sns.barplot(x='Predicted_Playoff_Appearances', y='Team', data=predicted_playoff_counts.sort_values(by='Predicted_Playoff_Appearances', ascending=False))
    plt.title(f'{model_name} Predicted Playoff Appearances')

    plt.tight_layout()
    plt.show()

def main():
    file_path = 'baseball.csv'
    df = read_dataset(file_path)

    X = df[['RS', 'RA', 'OBP', 'SLG', 'OOBP', 'OSLG']]
    y = df['Playoffs']
    

# Check for missing values in the dataset
    missing_values = X.isnull().sum()

# Display columns with missing values, if any
    print("Columns with missing values:")
    print(missing_values[missing_values > 0])

	# Impute missing values only for columns with missing values
    cols_with_missing_values = missing_values[missing_values > 0].index

	# Exclude 'RankSeason' and 'RankPlayoffs' from the imputation
    cols_to_impute = [col for col in cols_with_missing_values if col not in ['RankSeason', 'RankPlayoffs']]
    imputer = SimpleImputer(strategy='mean')
    X[cols_to_impute] = imputer.fit_transform(X[cols_to_impute])

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_imputed, X_test_imputed = impute_missing_values(X_train, X_test)

    # Print the number of samples in the training and testing datasets
    print("Number of samples in training set:", X_train.shape[0])
    print("Number of samples in testing set:", X_test.shape[0])

    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier()
    }

    for model_name, model in models.items():
        print(f"\n{model_name} Training and Test Accuracy (No Hyperparameter Tuning):")
        train_score, test_score, y_pred = train_and_evaluate_model(model, X_train_imputed, y_train, X_test_imputed, y_test)
        print(f"{model_name} Training Accuracy: {train_score}")
        print(f"{model_name} Test Accuracy: {test_score}")
        print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))

        # Add predictions to the original DataFrame
        df['Predicted_Playoffs'] = model.predict(X)

        # Get playoff counts based on the actual dataset
        actual_playoff_counts = df.groupby("Team")["Playoffs"].sum().reset_index(name="Actual_Playoff_Appearances")

        # Get playoff counts based on the model predictions
        predicted_playoff_counts = df.groupby("Team")["Predicted_Playoffs"].sum().reset_index(name="Predicted_Playoff_Appearances")

        # Print counts
        print("\nActual Playoff Appearances:")
        print(actual_playoff_counts)

        print("\nPredicted Playoff Appearances:")
        print(predicted_playoff_counts)

        # Plot playoff counts
        plot_playoff_counts(actual_playoff_counts, predicted_playoff_counts, model_name)

        if model_name == 'Random Forest':
            feature_importances = model.feature_importances_
            features = X.columns
            importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title(f'{model_name} Feature Importance')
            plt.show() 

    for model_name, model in models.items():
        print("\n------------------------------------------------------------------------------")
        print(f"\n{model_name} Hyperparameter Tuning:")
        param_grid = get_hyperparameter_grid(model_name)
        best_params, best_score, test_score, y_pred = perform_hyperparameter_tuning(model, param_grid, X_train_imputed, y_train, X_test_imputed, y_test)
        print(f"{model_name} Best Parameters: {best_params}")
        print(f"{model_name} Training Accuracy (Hyperparameter Tuning): {best_score}")
        print(f"{model_name} Test Accuracy (Hyperparameter Tuning): {test_score}")
        print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))

    print("\nAblative Analysis Results (No Hyperparameters):")
    for model_name, model in models.items():
        ablative_scores = ablative_analysis(model, X.columns, X_train, X_test, y_train, y_test)
        print(f"{model_name}:", ablative_scores)

    print("\nAblative Analysis Results (Hyperparameter-Tuned Models):")
    for model_name, model in models.items():
        param_grid = get_hyperparameter_grid(model_name)
        _, _, test_score, y_pred = perform_hyperparameter_tuning(model, param_grid, X_train_imputed, y_train, X_test_imputed, y_test)
        ablative_scores_tuned = ablative_analysis(model, X.columns, X_train, X_test, y_train, y_test)
        print(f"{model_name}:", ablative_scores_tuned)

def get_hyperparameter_grid(model_name):
    if model_name == 'Logistic Regression':
        return {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
    elif model_name == 'Random Forest':
        return {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    elif model_name == 'SVM':
        return {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    elif model_name == 'KNN':
        return {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    else:
        return {}

if __name__ == "__main__":
    main()

