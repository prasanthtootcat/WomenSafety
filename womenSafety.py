import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn import preprocessing

df=pd.read_csv("crimes_c.csv")

X=np.array(df.drop(['type'],1))
y=np.array(df['type'])
df=df[:10]
print(df)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example=np.array([[21,13.01,80.25]])
p=clf.predict(example)
print(p)
