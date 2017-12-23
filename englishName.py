import pandas as pd
import numpy as np
import random, nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


nltk.download('names') #english list of name -> male nad female

male_names = nltk.corpus.names.words('male.txt')
female_names = nltk.corpus.names.words('female.txt')

#concatenate 2 nupmy array
np_list = np.concatenate((np.array(male_names), np.array(female_names)))

#create dataFrame -> direction
df_male = pd.DataFrame({'name' : np.array(male_names), 'class' : 1})
df_female = pd.DataFrame({'name' : np.array(female_names), 'class' : 0})


#concatenate 2 data frame
df = pd.concat([df_male, df_female], axis=0)

#create feature (first letter, last letter, len )
df['first'] = df.name.apply(lambda x: x[0])
df['last'] = df.name.apply(lambda x: x[-1])
df['len'] = df.name.apply(lambda x: len(x))


print(df.head())

print(df.nunique())

#create train data frame
df_train = df.copy()
y = df_train['class'].values
df_train = df_train.drop(['name', 'class'], axis=1)
df_train = pd.get_dummies(df_train, drop_first=True)

X = df_train.values

#print(df_train.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=29, test_size=0.2)

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#print(confusion_matrix(y_test, y_pred))

ef=(y_pred == y_test).sum()/y_pred.shape[0]
print('effectiveness:', ef)


wyn = pd.DataFrame({'value': clf.feature_importances_, 'name': df_train.columns})
print(wyn.head())

x=wyn['name']
y=wyn['value']
plt.bar(x,y, align='center')
#or scatter plot
#plt.scatter(x,y)
plt.xticks(rotation=60)
plt.legend(loc='upper right')




plt.show()