from sklearn.model_selection import ShuffleSplit
import pandas as pd
import random
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


def load_data():
    filepath = 'data\Wholesale_Customers_Data.csv'
    df = pd.read_csv(filepath)
    df = df.drop(['Channel', 'Region'], axis=1)
    return df, df.columns


def information_about_data(df):
    print('Shape of data', df.shape)
    print('Description of dataset \n', df.describe())
    print('Head of dataset \n', df.head())
    return df, df.columns


def spliting_data_SS(df, predict):
    X = df.drop([predict], axis=1)
    y = df[predict]
    shsp = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    CVSS = {}
    i = 0
    for train, test in shsp.split(X):
        CVSS[i] = {'train': train, 'test': test}
        i += 1
    return CVSS


def preprocessing_data(df, predict):
    X_data = df.drop([predict], axis=1)
    y_data = df[predict]
    knn = KNeighborsRegressor(n_neighbors=3)
    return X_data, y_data, knn


def cross_validation(X_data, y_data, knn):
    cross_val = cross_val_score(knn, X_data, y_data, cv=4)
    return cross_val


if __name__ == '__main__':
    df, col = load_data()
    n = random.randint(0, len(col) - 1)
    print('Predict feature: ', col[n])

    X_data, y_data, knn = preprocessing_data(df, col[n])
    cross_val = cross_validation(X_data, y_data, knn)

    print('Score of cross validation', cross_val)
    print('Mean of result', round(np.mean(cross_val), 6))

