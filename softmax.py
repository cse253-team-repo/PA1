from dataloader import *
import numpy as np


class SoftmaxRegression:

    def __init__(self):
        pass

    def train_batch(self, X, y, classes, iterations=100, learning_rate=0.1):
        self.n_samples, self.n_features = X.shape
        self.classes = classes

        self.theta = np.random.rand(self.classes, self.n_features)
        self.bias = np.zeros((1, self.classes))

        for i in range(iterations):
            scores = np.dot(X, self.theta.T) + self.bias
            probs = self.softmax(scores)
            y_predict = np.argmax(probs, axis=1)[:, np.newaxis]
            y_one_hot = self.one_hot(y)

            dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))
            db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)

            self.theta = self.theta - learning_rate * dw.T
            self.bias = self.bias - learning_rate * db

            if i % 100 == 0:
                print('Iteration:', i)

        return self.theta, self.bias

    def train_stochastic(self, X, y, classes, iterations=100, learning_rate=0.1):
        self.n_samples, self.n_features = X.shape
        self.classes = classes

        self.theta = np.random.rand(self.classes, self.n_features)
        self.bias = np.zeros((1, self.classes))

        for i in range(iterations):
            scores = np.dot(X, self.theta.T) + self.bias
            probs = self.softmax(scores)
            y_one_hot = self.one_hot(y)

            rand1 = np.random.randint(0, self.n_samples, 1)
            h = np.dot(self.theta, X[rand1].T)
            for i in range(self.n_features):
                self.theta[:, i] = self.theta[:, i] - \
                    (1 / self.n_samples) * learning_rate * \
                    (h.T - y_one_hot[rand1]) * X[rand1, i].item(0)

            #dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))
            db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)

            #self.theta = self.theta - learning_rate * dw.T
            self.bias = self.bias - learning_rate * db

            if i % 100 == 0:
                print('Iteration:', i)

        return self.theta, self.bias

    def predict(self, X):
        scores = np.dot(X, self.theta.T) + self.bias
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)[:, np.newaxis]

    def softmax(self, scores):
        exp = np.exp(scores)
        sum_exp = np.sum(np.exp(scores))
        if sum_exp == float("inf"):
            print('Overflow Warning')
        softmax = exp / sum_exp
        return softmax

    def cross_entropy(self, y, scores):
        loss = - (1 / self.n_samples) * np.sum(y * np.log(scores))
        return loss

    def one_hot(self, y):
        one_hot = np.zeros((self.n_samples, self.classes))
        one_hot[np.arange(self.n_samples), y.T] = 1
        return one_hot


def PCA(X, pc_num=1):
    average_data = np.mean(X, axis=0)
    centered_data = X - average_data

    A_T = centered_data
    A = A_T.T
    eigenvalues, eigenvectors = np.linalg.eig(
        np.dot(A_T, A) / len(centered_data))

    projector = []
    for i in range(len(centered_data)):
        vector = np.dot(A, eigenvectors[:, i]).flatten().A1
        normalized_vector = (
            vector / (np.linalg.norm(vector) * (eigenvalues[i] ** 0.5)))
        projector.append(normalized_vector)
    projector = np.matrix(projector).T

    X_PCA = np.dot(X - average_data, projector[:, 0:pc_num])
    return X_PCA, average_data, projector


def PCA_project(X, average_data, projector, pc_num=1):

    X_PCA = np.dot(X - average_data, projector[:, 0:pc_num])
    return X_PCA


def get_accuracy(y_trained, y_true):
    if len(y_trained) != len(y_true):
        print('Different Data Dimension!')
        return 0

    match = 0
    for i in range(len(y_trained)):
        if y_trained[i] == y_true[i]:
            match += 1
    return match / len(y_trained)


data_dir = "./aligned/"
dataset, cnt = load_data(data_dir)
images = balanced_sampler(dataset, cnt, emotions=[
                          'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])

emotion_num = 0
X = []
y = []
for k in images.keys():
    X = X + [i.flatten() for i in images[k]]
    y = y + [emotion_num] * len(images[k])
    emotion_num += 1

X = np.matrix(X)
y = np.matrix(y).T

X_PCA, average_data, projector = PCA(X, pc_num=15)

softmax = SoftmaxRegression()
w_trained, b_trained = softmax.train_batch(
    X_PCA, y, classes=6, iterations=800, learning_rate=0.1)

y_predict = softmax.predict(X_PCA)
print(get_accuracy(y_predict.A1, y.A1))
