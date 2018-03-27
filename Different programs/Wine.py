from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

wine = datasets.load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('effectiveness KNeighborsClassifier: '+str(knn.score(X_test, y_test)))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for c, i in zip(colors, [0,1,2]):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], color=c, alpha=0.5, lw=lw, label='class %s' % i)

plt.title('Classifier wine')
plt.legend(loc='best')
plt.show()