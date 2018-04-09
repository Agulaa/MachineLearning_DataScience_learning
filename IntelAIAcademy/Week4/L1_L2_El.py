from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
import pandas as pd
import matplotlib.pyplot as plt

def _load_data():
    filepath = 'data\Orange_Telecom_Churn_Data.csv'
    df = pd.read_csv(filepath)
    df = df.drop(['state', 'phone_number', 'intl_plan',  'voice_mail_plan', 'churned'], axis=1)
    return df

def _train_test(df, name_col):
    X_data = df.drop([name_col], axis=1)
    y_data = df[name_col]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def _lasso(X_train, y_train):
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    return lasso.coef_

def _ridge(X_train, y_train):
    rigde = Ridge(alpha=0.1)
    rigde.fit(X_train, y_train)
    return rigde.coef_

def _elastic(X_train, y_train):
    el = ElasticNet(alpha=1.0, l1_ratio=0.7)
    el.fit(X_train, y_train)
    return el.coef_

def visualization_coef():
    df = _load_data()
    X_train, _, y_train, _ = _train_test(df, 'total_day_minutes')

    l1 = _lasso(X_train, y_train)
    l2 = _ridge(X_train, y_train)
    ela = _elastic(X_train, y_train)
    plt.subplot(1,3,1)
    plt.plot(l1, color='navy', label='Lasso')
    plt.legend(loc='best')
    plt.subplot(1, 3, 2)
    plt.plot(l2, color='cyan', label='Ridge')
    plt.legend(loc='best')
    plt.subplot(1, 3, 3)
    plt.plot(ela, color='orange', label='ElasticNet')
    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    visualization_coef()
