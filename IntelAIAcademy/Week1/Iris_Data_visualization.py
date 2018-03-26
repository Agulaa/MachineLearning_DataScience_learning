import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def create_new_columns(df):
    df['sepal_area'] = df.sepal_length * df.sepal_width
    df['petal_area'] = df.petal_length * df.petal_width
    df['abbrev'] = df.species.apply(lambda x:x.replace('Iris-', ''))
    return df


def create_small_data(df):
    small_data = pd.concat([df.iloc[0:2], df.iloc[-2:]])
    return small_data


def group_by(df, by):
    group = (df.groupby(by).size())
    return group


def statistic_all_columns(df):
    print('Statistic', df.describe())


def sample_data(df, n):
    sample = (df.sample(n=n, replace=False, random_state=21))
    return sample


def plot_data(df):
    plt.plot(df.sepal_length, df.sepal_area, ls='', marker='v', label='sepal')
    plt.plot(df.petal_length, df.petal_area, ls='', marker='+', label='petal')
    plt.legend(loc='best')
    plt.xlabel('length')
    plt.ylabel('area')
    plt.show()


def pair_plot(df):
    sns.pairplot(df, hue='species', size=3)
    plt.show()


def violin_plot(df):
    sns.violinplot(x='abbrev', y='sepal_area', data=df)
    plt.show()


def box_plot(df):
    sns.boxplot(x='abbrev', y='petal_area', data=df)
    plt.show()


def sub_plot(df):
    plt.subplot(2,2,1)
    sns.violinplot(x='abbrev', y='sepal_area', data=df)
    plt.subplot(2,2,2)
    sns.violinplot(x='abbrev', y='petal_area', data=df)
    plt.subplot(2, 2,3)
    sns.boxplot(x='abbrev', y='sepal_area', data=df)
    plt.subplot(2, 2, 4)
    sns.boxplot(x='abbrev', y='petal_area', data=df)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filepath = 'data/Iris_Data.csv'
    df = load_data(filepath)
    df = create_new_columns(df)
    print(df.head())
    sub_plot(df)












