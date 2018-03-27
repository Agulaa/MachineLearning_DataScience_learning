import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_csv(filepath)
    X_data = df.drop(['salary', 'department'], axis=1)
    y_data = df['salary']
    return X_data, y_data

def test_train_data(X_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    score = knn.score(X_test, y_test)
    return y_predict, score, X_train, X_test, y_train, y_test


def visualisation(X_train, y_train):
    df = pd.DataFrame(X_train, columns=X_train.columns)
    df['salary'] = y_train
    df = df[:100]
    salary = ['low', 'high', 'medium']
    color = ['navy', 'turquoise', 'darkorange']
    #is_medium = df.salary == 'medium'
    #is_low = df.salary == 'low'
    #is_high = df.salary == 'high'
    for c, i in zip(color, [0,1,2]):
        plt.scatter(df.loc[df['salary'] == salary[i], ['years_at_company']], df.loc[df['salary'] == salary[i], ['satisfaction_level']],marker='>' ,color=c, label='%s'%salary[i])

    plt.xlabel('years at company')
    plt.ylabel('satisfaction_level')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    filepath = 'data\Human_Resources_Employee_Attrition.csv'
    X_data, y_data = load_data(filepath)
    y_predict, score, X_train, _ , y_train, _  = test_train_data(X_data, y_data)
    visualisation(X_train, y_train)
    print('Score', score)
