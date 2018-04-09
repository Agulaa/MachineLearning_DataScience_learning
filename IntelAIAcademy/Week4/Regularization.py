import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import re


def _load_data():
    filepath = 'data\Wholesale_Customers_Data.csv'
    df = pd.read_csv(filepath)
    df = df.drop(['Channel', 'Region'], axis=1)
    return df

def _train_test(df, name_col):
    X_data = df.drop([name_col], axis=1)
    y_data = df[name_col]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def _ridge_regularization(X_train, X_test, y_train, y_test):
    ridge = Ridge(alpha=1.0, normalize=True)
    ridge.fit(X_train, y_train)
    y_predict = ridge.predict(X_test)
    score = ridge.score(X_test, y_test)
    err = mean_squared_error(y_test, y_predict)
    return score, err


def _lasso_regularization(X_train, X_test, y_train, y_test):
    lasso = Lasso(alpha=1.0, normalize=True)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    score = lasso.score(X_test, y_test)
    err = mean_squared_error(y_test, y_pred)
    return score, err


def _elastic_net_regularization(X_train, X_test, y_train, y_test):
    ela = ElasticNet(alpha=1.0, l1_ratio=2)
    ela.fit(X_train, y_train)
    y_pred = ela.predict(X_test)
    score = ela.score(X_test, y_test)
    err = mean_squared_error(y_test, y_pred)

    return score, err


def comparison_regularization():
    df = _load_data()
    dic = {}
    for i in df.columns:
        X_train, X_test, y_train, y_test = _train_test(df, i)
        score1, err1 = _lasso_regularization(X_train, X_test, y_train, y_test)
        score2, err2 = _ridge_regularization(X_train, X_test, y_train, y_test)
        scoreE, errE = _elastic_net_regularization(X_train, X_test, y_train, y_test)
        dic[i] = {'score_L1' : round(score1, 4), 'err_L1' : round(err1), 'score_L2' : round(score2, 4), 'err_L2' : round(err2), 'score_E' :round(scoreE, 4) , 'err_E' : round(errE)}
    return dic


def vis_score(dic):
    pattern = 'score'
    i = 1
    for key,value in dic.items():
        x = []
        y = []
        title = key
        for key,value in dic[title].items():
            match = re.search(pattern, key)
            if match:
                y.append(value)
                x.append(key)
        plt.subplot(3,2,i)
        plt.tight_layout()
        plt.axhline(color='black')
        plt.bar(x,y)
        plt.title(title)
        i+=1
    plt.suptitle('Regularization score')
    plt.show()


def vis_err(dic):
    pattern = 'err'
    i = 1
    for key,value in dic.items():
        x = []
        y = []
        title = key
        for key,value in dic[title].items():
            match = re.search(pattern, key)
            if match:
                y.append(value)
                x.append(key)
        plt.subplot(3,2,i)
        plt.tight_layout()
        plt.axhline(color='black')
        plt.bar(x,y)
        plt.title(title)
        i+=1
    plt.suptitle('Regularization error')
    plt.show()


def elimination_feature():
    df = _load_data()
    X_train, X_test, y_train, y_test = _train_test(df, 'Milk')
    linear = LinearRegression()
    rfe = RFE(linear, n_features_to_select=3)
    rfe.fit(X_train, y_train)
    y_predict = rfe.predict(X_test)
    score = rfe.score(X_test, y_test)
    err = mean_squared_error(y_test, y_predict)
    return score, err, y_predict





if __name__ == '__main__':
    dic = comparison_regularization()
    vis_err(dic)

