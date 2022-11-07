import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import perceptron

s = 'data\IRIS.csv'
df = pd.read_csv(s, header=None, encoding='utf-8')
Y = df.iloc[0:100,4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)

X = df.iloc[1:101,[0,2]].values
for i in range(len(X)):
    for j in range(2):
        X[i,j]=float(X[i,j])

X_std = np.copy(X)
X_std[:,0] = (X[:,0]-X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1]-X[:,1].mean()) / X[:,0].std()


neuron = perceptron.Perceptron(n_iter=15,eta=0.001)

neuron.fit(X_std,Y)
cnt = 0

pred=neuron.predict(X_std)
pred = np.array([np.where(y==p,1,0) for y,p in zip(Y,pred)])
print(pred.sum())
plt.plot(neuron.cost)
plt.show()









