from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('ggplot')

iris=datasets.load_iris()

print(iris.keys())

X=iris.data
y=iris.target

df=pd.DataFrame(X, columns=iris.feature_names)
print(df.head())


pd.plotting.scatter_matrix(df,c=y,figsize=[8,8], s=150, marker='D')
plt.show()


