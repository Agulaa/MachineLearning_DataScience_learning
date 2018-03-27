import pandas as pd
import numpy as np
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


nltk.download('names') #english list of name -> male nad female

male_names = nltk.corpus.names.words('male.txt')
female_names = nltk.corpus.names.words('female.txt')

#create dataFrame -> direction
df_male = pd.DataFrame({'name': np.array(male_names), 'sex' : 1})
df_female = pd.DataFrame({'name': np.array(female_names), 'sex' : 0})


#concatenate 2 data frame
df = pd.concat([df_male, df_female], axis=0)

#create feature (first letter, last letter, len )
df['first'] = df.name.apply(lambda x: x[0])
df['last'] = df.name.apply(lambda x: x[-1])
df['len'] = df.name.apply(lambda x: len(x))

#create train data frame
df_train = df.copy()
print(df_train.head())
y = df_train['sex'].values
df_train = df_train.drop(['name', 'sex'], axis=1)
df_train = pd.get_dummies(df_train)
X = df_train.values


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=29, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

print('effectiveness:', clf.score(X_test, y_test))



wyn = pd.DataFrame({'value': clf.feature_importances_, 'name': df_train.columns})
print(wyn.head())

x = wyn['name']
y = wyn['value']
plt.bar(x, y, color='turquoise')
plt.xticks(rotation=60)
plt.show()