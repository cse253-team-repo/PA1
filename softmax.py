from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
import numpy as np
np.random.seed(13)


class SoftmaxRegression:

    def __init__(self):
        pass

    def train_batch(self, X, y, classes, iterations=100, learning_rate=0.1):
        self.n_samples, n_features = X.shape
        self.classes = classes

        self.weights = np.random.rand(self.classes, n_features)
        self.bias = np.zeros((1, self.classes))
        all_losses = []

        for i in range(iterations):
            scores = self.compute_scores(X)
            probs = self.softmax(scores)
            y_predict = np.argmax(probs, axis=1)[:, np.newaxis]
            y_one_hot = self.one_hot(y_true)

            loss = self.cross_entropy(y_one_hot, probs)
            all_losses.append(loss)

            dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))
            db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)

            self.weights = self.weights - learning_rate * dw.T
            self.bias = self.bias - learning_rate * db

            if i % 100 == 0:
                print(f'Iteration number: {i}, loss: {np.round(loss, 4)}')

        return self.weights, self.bias, all_losses

    def train_stochastic(self, X, y, classes, iterations=100, learning_rate=0.1):
        self.n_samples, n_features = X.shape
        self.classes = classes

        self.weights = np.random.rand(self.classes, n_features)
        self.bias = np.zeros((1, self.classes))
        all_losses = []

        for i in range(iterations):
            scores = self.compute_scores(X)
            probs = self.softmax(scores)
            y_predict = np.argmax(probs, axis=1)[:, np.newaxis]
            y_one_hot = self.one_hot(y_true)

            loss = self.cross_entropy(y_one_hot, probs)
            all_losses.append(loss)

            dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))
            db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)

            self.weights = self.weights - learning_rate * dw.T
            self.bias = self.bias - learning_rate * db

            if i % 100 == 0:
                print(f'Iteration number: {i}, loss: {np.round(loss, 4)}')

        return self.weights, self.bias, all_losses

    def predict(self, X):
        scores = self.compute_scores(X)
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)[:, np.newaxis]

    def softmax(self, scores):
        exp = np.exp(scores)
        sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
        softmax = exp / sum_exp

        return softmax

    def compute_scores(self, X):
        return np.dot(X, self.weights.T) + self.bias

    def cross_entropy(self, y_true, scores):
        loss = - (1 / self.n_samples) * np.sum(y_true * np.log(scores))
        return loss

    def one_hot(self, y):
        one_hot = np.zeros((self.n_samples, self.classes))
        one_hot[np.arange(self.n_samples), y.T] = 1
        return one_hot


X, y_true = make_blobs(centers=4, n_samples=5000)
y_true = y_true[:, np.newaxis]
X_train, y_train = X, y_true

regressor = SoftmaxRegression()
w_trained, b_trained, loss = regressor.train_batch(
    X_train, y_train, learning_rate=0.1, iterations=800, classes=4)


print('Training Data Structure:', X_train[0])
print('Training Label Structure:', y_train[0])
print('Training Label Structure:', y_train[1])
print('Training Label Structure:', y_train[2])
print('Training Label Structure:', y_train[3])
print('Training Label Structure:', y_train[4])
