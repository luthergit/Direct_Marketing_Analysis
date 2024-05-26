# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# Define functions for loading data and initial display
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=';')
    return df

def display_data_info(df):
    print('\nDirect Marketing Data')
    print(df.value_counts())
    print(df.head())
    print('\nChecking unique values in categorical variables')
    for col in df.select_dtypes(include='object').columns:
        print(df[col].unique())
    print('\nChecking for missing values and the data type')
    print(df.info())
    print('\nDescriptive Statistics:')
    print(df.describe())
    return df


# Define function for feature distribution visualization
def distribution_of_features(df, columns):
    plt.figure(figsize=(20, 6))
    sns.countplot(data=df, x=columns, hue='y')
    sns.set(style='darkgrid')
    plt.xticks()
    plt.title(f'Distribution of {columns} based on responses')
    plt.show()



# Define function for campaign performance visualization
def campaign_performance(df):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    sns.boxplot(x="y", y='duration', data=df, ax=axes[0], hue='y')
    axes[0].set_title('Last Contact Duration in Seconds Vs Response')

    sns.boxplot(x="y", y='pdays', data=df, ax=axes[1], hue='y')
    axes[1].set_title('Number of days since the client was last contacted from a previous campaign Vs Response')

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    sns.countplot(data=df, x='campaign', hue='y', ax=axes[0])
    axes[0].set_title('Number of contacts performed during this campaign with Response')
    plt.xticks()

    sns.countplot(data=df, x='previous', hue='y', ax=axes[1])
    axes[1].set_title('Number of contacts performed before this campaign with Response')
    plt.xticks()

    plt.show()




# Define functions for correlation and chi-squared tests
def biserial_correlation(df):
    df['y_numeric'] = df['y'].map({'yes': 1, 'no': 0})
    numeric_columns = list(df.select_dtypes(include=np.number).columns)
    numeric_columns.remove('y_numeric')

    corr_results = {}
    for column in numeric_columns:
        corr, p_value = pointbiserialr(x=df['y_numeric'], y=df[column])
        corr_results[column] = {'correlation': corr, 'p_value': p_value}

    corr_df = pd.DataFrame(corr_results).T
    filtered_corr_df = corr_df[corr_df.p_value <= 0.05]
    filtered_numerical_features = list(filtered_corr_df.index)
    print('\n----------------------')
    print(corr_df.round(6))
    print('selected filtered_numerical_features:', filtered_numerical_features)
    return filtered_numerical_features

def chi_squared_test(df):
    categorical_columns = list(df.select_dtypes(include='object').columns)
    categorical_columns.remove('y')

    def chi_sqaure_test(column):
        crosstab_table = pd.crosstab(df[column], df['y'])
        chi2, p_value, dof, expected = chi2_contingency(crosstab_table)
        return chi2, p_value

    chi_results = {}
    for column in categorical_columns:
        chi2, p_value = chi_sqaure_test(column)
        chi_results[column] = {'chi2': chi2, 'p_value': p_value}

    chi_df = pd.DataFrame(chi_results).T
    filtered_chi_df = chi_df[chi_df.p_value <= 0.05]
    filtered_categorical_features = list(filtered_chi_df.index)
    print('\n----------------------')
    print(chi_df.round(6))
    print('selected filtered_categorical_features:', filtered_categorical_features)

    return filtered_categorical_features


# Define function for addressing categorical data
def address_categorical_data(df, filtered_numerical_features, filtered_categorical_features):
    le = LabelEncoder()
    df_copy = df.copy()

    df_copy.drop(columns=['y'], inplace=True)
    for col in df_copy.select_dtypes(include='object').columns:
        df_copy[col] = le.fit_transform(df_copy[col])

    plt.figure(figsize=(20, 20))
    sns.set(font_scale=1.5)
    sns.heatmap(df_copy.corr(), cmap='coolwarm', annot=True)
    plt.title('Correlation matrix')
    plt.show()

    select_features = filtered_numerical_features + filtered_categorical_features
    print('\n----------------------')
    print('selected features:', select_features)
    return df_copy, select_features



# Define function for balancing data
def balance_data(df_copy):
    count_class_0, count_class_1 = df_copy.y_numeric.value_counts()

    df_class_0 = df_copy[df_copy['y_numeric'] == 0]
    df_class_1 = df_copy[df_copy['y_numeric'] == 1]

    df_class_0_under = df_class_0.sample(count_class_1, random_state=1)
    df_under_sample = pd.concat([df_class_0_under, df_class_1], axis=0)

    return df_under_sample


# Define function for balancing data
def balance_data(df_copy):
    count_class_0, count_class_1 = df_copy.y_numeric.value_counts()

    df_class_0 = df_copy[df_copy['y_numeric'] == 0]
    df_class_1 = df_copy[df_copy['y_numeric'] == 1]

    df_class_0_under = df_class_0.sample(count_class_1, random_state=1)
    df_under_sample = pd.concat([df_class_0_under, df_class_1], axis=0)
    
    print('\n----------------------')
    print('Balanced class distribution:\n', df_under_sample.y_numeric.value_counts())

    return df_under_sample


# Define function for splitting data
def split_data(df_under_sample, select_features):
    X = df_under_sample[select_features]
    y = df_under_sample['y_numeric']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)
    
    print('\n----------------------')
    print('Training data class distribution:\n', y_train.value_counts())
    print('Testing data class distribution:\n', y_test.value_counts())
    return X_train, X_test, y_train, y_test


def select_model(X, y):
    all_X = X
    all_y = y
    
    models = [
        {
            "name": "LogisticRegression",
            "estimator": LogisticRegression(),
            "hyperparameters":
                {
                    "solver": ["newton-cg", "liblinear"]
                }
        },
        {
            "name": "KNeighborsClassifier",
            "estimator": KNeighborsClassifier(),
            "hyperparameters":
                {
                    "n_neighbors": range(1,20,2),
                    "weights": ["distance", "uniform"],
                    "algorithm": ["ball_tree", "kd_tree", "brute"],
                    "p": [1,2]
                }
        },
        {
            "name": "RandomForestClassifier",
            "estimator": RandomForestClassifier(random_state=1),
            "hyperparameters":
                {
                    "n_estimators": [4, 6, 9],
                    "criterion": ["entropy", "gini"],
                    "max_depth": [2, 5, 10],
                    "max_features": ["log2", "sqrt"],
                    "min_samples_leaf": [1, 5, 8],
                    "min_samples_split": [2, 3, 5]

                }
        }
    ]

    best_model = None
    best_precision_score = 0
    
    for model in models:
        print (model['name'])
        print ('-'*len(model['name']))
    
        grid = GridSearchCV(model['estimator'], param_grid=model['hyperparameters'], cv=10, scoring ='precision')
        grid.fit(all_X, all_y)
        model['best_params'] = grid.best_params_
        model['best_precision_score'] = grid.best_score_
        model['best_model'] = grid.best_estimator_

    
        print('Best Precision Score: {}'.format(model['best_precision_score']))
        print('Best parameters: {} \n'.format(model['best_params']))

        if model['best_precision_score'] > best_precision_score:
            best_precision_score = model['best_precision_score']
            best_model = model['best_model']

    
    print ('-'*len(model['name']))
    print('Best overall model:')
    print('Model:', best_model)
    print('Best Precision Score:', best_precision_score)
    
    return best_model, models


# Define function for evaluating model
def evaluate_model(best_model, X_test, y_test):
    predictions = best_model.predict(X_test)
    cl_rep = classification_report(y_test, predictions)
    return cl_rep



# Define function for feature importance with recursive feature elimination
def recursive_feature_elimination(models, X_train, y_train, select_features):
    def get_best_random_forest_model(models):
        for model in models:
            if model["name"] == "RandomForestClassifier":
                return model["best_model"]

    best_rf_model = get_best_random_forest_model(models)

    rfecv = RFECV(estimator=best_rf_model, step=2, scoring='precision', min_features_to_select=len(select_features))
    rfecv.fit(X_train, y_train)

    dset = pd.DataFrame()
    dset['attr'] = X_train.columns
    dset['importance'] = rfecv.estimator_.feature_importances_

    dset = dset.sort_values(by='importance', ascending=True)

    plt.figure(figsize=(16, 14))
    plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
    plt.title('RFECV - Feature importance', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=14, labelpad=20)
    plt.show()



# Main function to run 
import argparse

def main(file_path):
    df = load_data(file_path)
    display_data_info(df)

    distribution_of_features(df, 'age')
    distribution_of_features(df, 'job')
    distribution_of_features(df, 'marital')
    distribution_of_features(df, 'education')
    distribution_of_features(df, 'y')

    campaign_performance(df)

    distribution_of_features(df, 'contact')
    distribution_of_features(df, 'day')
    distribution_of_features(df, 'month')

    filtered_numerical_features = biserial_correlation(df)
    filtered_categorical_features = chi_squared_test(df)

    df_copy, select_features = address_categorical_data(df, filtered_numerical_features, filtered_categorical_features)

    df_under_sample = balance_data(df_copy)

    X_train, X_test, y_train, y_test = split_data(df_under_sample, select_features)

    best_model, models = select_model(X_train, y_train)

    print('The Best Model:', best_model)

    print('Model Metrics:')
    print(evaluate_model(best_model, X_test, y_test))

    recursive_feature_elimination(models, X_train, y_train, select_features)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get some hyperparameters.")


    # Create an arg for data directory 
    parser.add_argument("--file_path",
                        default="bank.csv",
                        type=str,
                        help="directory file path to data")
    
    # Get our arguments from the parser
    args = parser.parse_args()
    
    main(args.file_path)





