from sklearn import datasets
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.4)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Confusion matrix \n", cm)
print("Classification_report \n", cr)

True_Positive = np.diag(cm)
Lower_triangular = np.tril(cm, k=-1) #without diagonal
Upper_triangular = np.triu(cm, k=1)

print("True Positive", True_Positive, "\n")
print('Lower Triangular', Lower_triangular, "\n")
print('Upper Triangular', Upper_triangular, "\n")

precision = True_Positive.sum() / (True_Positive.sum()+Lower_triangular.sum(axis=0).sum())
recall = True_Positive.sum() /(True_Positive.sum() +Upper_triangular.sum(axis=0).sum())

print("Precision: ", precision)
print("Recall: ", recall)

