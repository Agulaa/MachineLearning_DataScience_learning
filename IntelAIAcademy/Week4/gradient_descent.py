import numpy as np
from numpy import zeros
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_and_split_data():
    diabetes = datasets.load_diabetes()
    #0 33 = Regular insulin dose
    #19 71 = Less-than-usual exercise activity
    X_diabetes = diabetes.data[:,0].reshape(-1,1)
    y_diabetes = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def train_test_data(X_train, X_test, y_train):
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    y_pred = linear.predict(X_test)
    return y_pred, linear.coef_, linear.intercept_


def visualization():
    X_train, X_test, y_train, y_test = load_and_split_data()
    y_pred, _, _ = train_test_data(X_train, X_test, y_train)
    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred, color='cyan', linewidth=3)
    plt.show()


def cost_function(X, y_train, beta):
    m = len(y_train)
    predicted_value = X.dot(beta)
    sum = ((predicted_value - y_train)**2).sum()
    cost = (1 / (2 * m)) * sum
    return cost


def gradient_descent(X, y_train, alpha, n):
    m = len(y_train)
    beta = zeros(2)

    for i in range(n):
        beta_0 = beta[0] - (alpha / m) * (X.dot(beta) - y_train ).sum()
        beta_1 = beta[1] - (alpha / m) * ((X.dot(beta) - y_train ) * X[:,1]).sum()
        beta = np.array([beta_0, beta_1])

    return beta



if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_split_data()
    y_pred, a, b = train_test_data(X_train, X_test, y_train)
    print('Coeff:', a,'Intercept:', b)

    X = np.column_stack((np.ones(len(X_train)), X_train))
    beta = zeros(2)
    cost = cost_function(X, y_train, beta)
    print('Beta', beta)
    print('Cost', cost)

    print('After gradient descent:')
    n = 100
    alpha = 0.1
    beta = gradient_descent(X, y_train, alpha, n)
    cost = cost_function(X, y_train, beta)
    print('Beta', beta)
    print('Cost', cost)