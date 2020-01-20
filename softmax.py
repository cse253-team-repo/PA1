from dataloader import *
import numpy as np


class SoftmaxRegression:

    def __init__(self):
        pass

    def train_batch(self, X, y, classes, epoch=100, learning_rate=0.01):
        self.n_samples, self.n_features = X.shape
        self.classes = classes

        self.theta = np.random.rand(self.classes, self.n_features)
        self.bias = np.zeros((1, self.classes))

        for i in range(epoch):
            scores = np.dot(X, self.theta.T) + self.bias
            probs = self.softmax(scores)
            y_predict = np.argmax(probs, axis=1)[:, np.newaxis]
            y_one_hot = self.one_hot(y)

            dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))
            db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)

            self.theta = self.theta - learning_rate * dw.T
            self.bias = self.bias - learning_rate * db

        return self.theta, self.bias

    def train_stochastic(self, X, y, classes, epoch=100, learning_rate=0.01):
        self.n_samples, self.n_features = X.shape
        self.classes = classes

        self.theta = np.random.rand(self.classes, self.n_features)
        self.bias = np.zeros((1, self.classes))

        for i in range(epoch):
            random_order = np.arange(self.n_samples)
            np.random.shuffle(random_order)

            for j in range(len(random_order)):
                scores = np.dot(X, self.theta.T) + self.bias
                probs = self.softmax(scores)
                y_one_hot = self.one_hot(y)

                rand = random_order[j]
                h = np.dot(self.theta, X[rand].T)
                dw = (1 / self.n_samples) * \
                    np.outer(h.T - y_one_hot[rand], X[rand])
                self.theta = self.theta - learning_rate * dw

                db = (1 / self.n_samples) * \
                    np.sum(probs - y_one_hot, axis=0)
                self.bias = self.bias - learning_rate * db

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


def cross_fold(X, fold_num=2):
    indices = []
    num = int(len(X) / fold_num)
    for i in range(fold_num):
        if i != (fold_num - 1):
            temp_index = [i for i in range(i * num, (i + 1) * num)]
        else:
            temp_index = [i for i in range(i * num, len(X))]
        indices.append(temp_index)
    return indices


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


def get_accuracy(y_predict, y_true):
    if len(y_predict) != len(y_true):
        print('Different Data Dimension!')
        return 0

    match = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_true[i]:
            match += 1
    return match / len(y_predict)


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


shuffle_order = np.arange(len(X))
np.random.shuffle(shuffle_order)
X = np.matrix(X)[shuffle_order]
y = np.matrix(y).T[shuffle_order]

num_folds = 10
folds = cross_fold(X, num_folds)
for fold in range(num_folds):
    X_val = X[folds[fold]]
    y_val = y[folds[fold]]
    X_test = X[folds[(fold + 1) % num_folds]]
    y_test = y[folds[(fold + 1) % num_folds]]

    rest_folds = []
    for i in range(num_folds):
        if i != fold and i != ((fold + 1) % num_folds):
            rest_folds = rest_folds + folds[i]
    X_train = X[rest_folds]
    y_train = y[rest_folds]

    X_train_PCA, average_data, projector = PCA(X_train, pc_num=40)
    X_test_PCA = PCA_project(X_test, average_data, projector, pc_num=40)
    X_val_PCA = PCA_project(X_val, average_data, projector, pc_num=40)

    softmax = SoftmaxRegression()
    w_trained, b_trained = softmax.train_stochastic(
        X_train_PCA, y_train, classes=6, epoch=100, learning_rate=0.1)

    y_train_predict = softmax.predict(X_train_PCA)
    y_test_predict = softmax.predict(X_test_PCA)
    y_val_predict = softmax.predict(X_val_PCA)
    print('Training Accuracy:', get_accuracy(y_train_predict.A1, y_train.A1))
    print('Testing Accuracy:', get_accuracy(y_test_predict.A1, y_test.A1))
    print('Validation Accuracy:', get_accuracy(y_val_predict.A1, y_val.A1))
