from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def train_test_datasets():

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    tree.fit(X_train, y_train)
    return X_train, X_test, y_train, y_test, tree


def visualization():
    X_train, X_test, y_train, y_test, tree = train_test_datasets()

    plot_decision_regions(X_train, y_train, clf=tree, legend=2)
    plt.title('Without test set')
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    visualization()

