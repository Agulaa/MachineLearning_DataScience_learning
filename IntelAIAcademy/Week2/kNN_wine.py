import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def load_data(filepath):
    df = pd.read_csv(filepath)
    X_data = df.drop(['color','chlorides', 'total_sulfur_dioxide', 'chlorides', 'free_sulfur_dioxide', 'quality', 'citric_acid', 'volatile_acidity'], axis=1)
    y_data = df['color']
    return df, X_data, y_data


def preprocessing_data_SS(X_data):
    standSc = StandardScaler()
    standSc.fit(X_data)
    data = standSc.transform(X_data)
    X_scaled = pd.DataFrame(data,  columns=X_data.columns)
    return  X_scaled


def preprocessing_data_MMS(X_data):
    mms = MinMaxScaler()
    mms.fit(X_data)
    data = mms.transform(X_data)
    X_scaled = pd.DataFrame(data, columns=X_data.columns)
    return X_scaled


def preprocessing_data_MAS(X_data):
    mas = MaxAbsScaler()
    mas.fit(X_data)
    data = mas.transform(X_data)
    X_scaled = pd.DataFrame(data, columns=X_data.columns)
    return X_scaled


def model_clasiffication(X_scaled, y_data):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, train_size=0.3, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)
    score = knn.score(X_test, y_test)
    return score, y_predict, X_test


def split_data(df):
    is_red = df.predict == "red"  # return true or false
    df_red = df.loc[is_red, :]
    df_white = df.loc[~is_red, :]
    return df_red, df_white

def visualization_alcohl_ph(X_test, y_predict):
    df = pd.DataFrame(X_test)
    df['predict'] = y_predict
    df_red, df_white = split_data(df)

    plt.scatter(df_red.alcohol[:30], df_red.pH[:30], marker='o',label='red')
    plt.scatter(df_white.alcohol[:30], df_white.pH[:30], marker='o', label='white')
    plt.xlabel('Alcohol')
    plt.ylabel('pH')
    plt.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    filepath = 'data\Wine_Quality_Data.csv'
    df, X_data, y_data = load_data(filepath)

    X_scaled = preprocessing_data_SS(X_data)
    score, y_predict, X_test = model_clasiffication(X_scaled, y_data)
    print('Score', score)
    visualization_alcohl_ph(X_test, y_predict)





