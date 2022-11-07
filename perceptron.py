import numpy as np
#ADALINE
class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1, shuffle=True):
        self.eta=eta
        self.n_iter=n_iter
        self.w_initialized=False
        self.random_state=random_state
        self.shuffle=shuffle

    def fit(self,X,Y):
        self.initialize_weights(X.shape[1])
        self.cost=[]

        for i in range(self.n_iter):
            if self.shuffle:
                X,Y=self.Shuffle(X,Y)
            C = []
            for xi,yi in zip(X,Y):
                C.append(self.update_weights(xi,yi))
            avg=sum(C)/len(Y)
            self.cost.append(avg)
        return self

    def partial_fit(self, X, Y):
        if not self.w_initialized:
            self.initialize_weights(X.shape[1])
        if Y.ravel().shape[0]>1:
            for xi,yi in zip(X,Y):
                self.update_weights(xi,yi)
        else:
            self.update_weights(X,Y)
        return self

    def Shuffle(self, X, Y):
        r = self.rgen.permutation(len(Y))
        return X[r],Y[r]

    def initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w = self.rgen.normal(loc=0.0,scale=0.01,size=1+m)
        self.w_initialized=True

    def update_weights(self, xi, yi):
        output = self.activation(self.net_input(xi))
        error = (yi - output)

        self.w[1:] = np.add(self.w[1:],self.eta*xi.dot(error))
        self.w[0] = np.add(self.w[0],self.eta*error)
        cost = 0.5*error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X))>=0.0, 1, -1)

