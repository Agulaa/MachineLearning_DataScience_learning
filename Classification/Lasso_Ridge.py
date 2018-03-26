from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


def train_test_datasets():

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    return X_train, X_test, y_train, y_test, X


def ridge_(X_train, X_test, y_train, y_test):
    ridge = Ridge(alpha=0.1, normalize=True)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)
    score = ridge.score(X_test, y_test)
    return pred, score, ridge


def plot_(X_train, X_test, y_train, y_test, X):
    _, _, ridge = ridge_(X_train, X_test, y_train, y_test)
    ride_coef = ridge.coef_
    plt.plot(range(len(X)), ride_coef)
    plt.xticks(range(len(X)), X)
    plt.ylabel('coefficients')
    plt.show()


def main_1():
    X_train, X_test, y_train, y_test, X = train_test_datasets()
    pred, score, ridge = ridge_(X_train, X_test, y_train, y_test)
    print('Score', score)
    plot_(X_train, X_test, y_train, y_test, X)


if __name__ == "__main__":
    main_1()
