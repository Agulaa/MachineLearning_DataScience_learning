from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler



def load_data():
    filepath = 'data\Wholesale_Customers_Data.csv'
    df = pd.read_csv(filepath)
    df = df.drop(['Channel', 'Region'], axis=1)
    return df, df.columns


def preprocessing_data_SS(X_data):
    standSc = StandardScaler()
    standSc.fit(X_data)
    data = standSc.transform(X_data)
    X_scaled = pd.DataFrame(data,  columns=X_data.columns)
    return  X_scaled


def train_test():
    df, col = load_data()
    df = preprocessing_data_SS(df)
    predict = 'Fresh'
    X_data = df.drop([predict], axis=1)
    y_data = df[predict]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def create_instance():
    X_train, X_test, y_train,y_test = train_test()
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_predict = linreg.predict(X_test)
    err = mean_squared_error(y_test, y_predict, multioutput='raw_values')
    scor = linreg.score(X_test, y_test)

    return err, scor, y_predict



if __name__ == '__main__':
    err, scor, y_predict = create_instance()

