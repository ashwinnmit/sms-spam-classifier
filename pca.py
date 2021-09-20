from numpy.core.fromnumeric import reshape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

# df = pd.read_csv('train.csv')
# df = df.drop(['Survived','Name','Sex','Embarked','Ticket','Cabin','Age'] , axis = 1)
# print(df.info())
# df = np.array([1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5])
# df = df.reshape([5,-1])
iris = load_breast_cancer()
df = iris.data
pca = PCA(n_components=2)
x_pca = pca.fit_transform(df)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0] , x_pca[:,1] , c=iris.target)
plt.xlabel('First Feature')
plt.ylabel('Second Feature')
plt.show()