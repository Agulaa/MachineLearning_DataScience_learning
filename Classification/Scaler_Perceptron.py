from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions_(X, y, classifier, test_idx=None, resolution=0.02):
    """small convenience function to visualize the decision boundaries for 2D datasets"""

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # create a color map from the list of colors via ListedColormap
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    # determine the minimum and maximum values for the two feature
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # create a pair of grid arrays use feature vectors
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # trained our perceptron classifier on two feature dimensions
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # reshaping the predicted class labels Z into a grid with the same dimensions as xx1 and xx2
    Z = Z.reshape(xx1.shape)
    #  draw a contour plot
    #  function that maps the different decision regions to different colors for each predicted class in the grid array
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')


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


def logistic_reg(X_train_std, X_test_std, y_train, y_test):
    log = LogisticRegression(C=1000.0, random_state=42)
    log.fit (X_train_std, y_train)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    predictions = log.predict_proba(X_test_std[0, :])
    return X_combined_std, y_combined, log, predictions


def plot_(X_combined_std, y_combined, log):
    plot_decision_regions_(X_combined_std, y_combined, classifier=log, test_idx=range(105, 150))
    plt.xlabel('Pental length [standaridized]')
    plt.ylabel('pental width [standardized]')
    plt.legend(loc='upper left')
    plt.show()


def main_1():
    X_train, X_test, y_train, y_test = train_test_datasets()
    X_train_std, X_test_std = standard_scaler(X_train, X_test)
    y_pred = perceptron(X_train_std, X_test_std, y_train, y_test)
    metrics(y_test, y_pred)


def main_2():
    X_train, X_test, y_train, y_test = train_test_datasets()
    X_train_std, X_test_std = standard_scaler(X_train, X_test)
    X_combined_std, y_combined, log, predictions = logistic_reg(X_train_std, X_test_std, y_train, y_test)
    plot_(X_combined_std, y_combined, log)


if __name__ == '__main__':
   main_2()

