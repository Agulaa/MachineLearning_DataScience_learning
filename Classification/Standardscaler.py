from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def train_test_datasets():

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    return X_train, X_test, y_train, y_test


def standard_scaler(X_train, X_test):

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std


def perceptron(X_train_std,X_test_std, y_train, y_test):
    # eta0 is equivalent to the learning rate eta that we used in our own perceptron implementation,
    # n_iter defines the number of epochs (passes over the training set)
    perc = Perceptron(max_iter=40, eta0=0.1, random_state=0)
    perc.fit(X_train_std, y_train)
    y_pred = perc.predict(X_test_std)
    return y_pred


def metrics(y_test, y_pred):
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Classification report: \n ', classification_report(y_test, y_pred))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = train_test_datasets()
    X_train_std, X_test_std = standard_scaler(X_train, X_test)
    y_pred = perceptron( X_train_std, X_test_std, y_train, y_test)
    metrics(y_test, y_pred)

