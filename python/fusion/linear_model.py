# python/fusion/linear_model.py
from .tensor import zeros

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        # X: shape (n, m), y: shape (n, 1)
        n = X.shape[0]
        m = X.shape[1]
        self.weights = zeros([m, 1])
        for _ in range(self.epochs):
            pred = X.matmul(self.weights)  # (n, 1)
            error = pred - y               # subtract
            grad = X.transpose().matmul(error) * (2.0 / n)
            self.weights = self.weights - grad * self.lr
        return self

    def predict(self, X):
        return X.matmul(self.weights)