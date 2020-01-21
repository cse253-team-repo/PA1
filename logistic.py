from dataloader import *
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self):
        pass

    def train_batch(self, X, y, epoch=50, learning_rate=0.1, threshold=0.5):
        self.samples, self.features = X.shape
        self.threshold = threshold
        self.loss = []

        X = np.insert(X, 0, 1, axis=1)
        self.theta = np.zeros(self.features + 1)

        for i in range(epoch):
            probs = self.sigmoid(X.dot(self.theta))
            print("probs shape: ", probs.shape)
            print("targets shape: ", y.shape)
            print("inputs shape: ", X.shape)
            dw = (1 / self.samples) * np.dot(X.T, (probs - y))
            print("dw shape: ", dw.shape)
            self.theta = self.theta - learning_rate * dw

        return self.theta, self.loss

    def train_stochastic(self, X, y, epoch=50, learning_rate=0.1, threshold=0.5):
        self.samples, self.features = X.shape
        self.threshold = threshold
        self.loss = []

        X = np.insert(X, 0, 1, axis=1)
        self.theta = np.random.rand(self.features + 1)

        for i in range(epoch):
            random_order = np.arange(self.samples)
            np.random.shuffle(random_order)

            for j in range(len(random_order)):
                rand = random_order[j]
                prob = self.sigmoid(X[rand].dot(self.theta))
                dw = (1 / self.samples) * np.dot(X[rand].T, (prob - y[rand]))
                self.theta = self.theta - learning_rate * dw

            # self.loss.append(self.cross_entropy(X, y_one_hot))

        return self.theta, self.loss

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # def cross_entropy(self, X, y_one_hot):
    #    ce = - np.sum(np.multiply(y_one_hot, np.log(self.sigmoid(X))))
    #    return ce

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        probs = self.sigmoid(X.dot(self.theta))
        return np.round(probs).astype(int)


def cross_fold(X, fold_num):
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
        vector = np.dot(A, eigenvectors[:, i]).flatten()
        normalized_vector = (
            vector / (np.linalg.norm(vector) * (eigenvalues[i] ** 0.5)))
        projector.append(normalized_vector)
    projector = np.array(projector).T

    X_PCA = np.dot(X - average_data, projector[:, 0:pc_num])

    return X_PCA, average_data, projector


def PCA_project(X, average_data, projector, pc_num=1):
    X_PCA = np.dot(X - average_data, projector[:, 0:pc_num])
    return X_PCA


def standardize(X):
    mean = np.mean(X, axis=1)[:, np.newaxis]
    std = np.std(X, axis=1)[:, np.newaxis]
    return (X - mean) / std


def get_accuracy(y_predict, y_true):
    if len(y_predict) != len(y_true):
        return 0

    match = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_true[i]:
            match += 1
    return match / len(y_predict)


data_dir = "./aligned/"
dataset, cnt = load_data(data_dir)
images = balanced_sampler(dataset, cnt, emotions=['fear', 'surprise'])

emotion_num = 0
X = []
y = []
for k in images.keys():
    X = X + [i.flatten() for i in images[k]]
    y = y + [emotion_num] * len(images[k])
    emotion_num += 1

shuffle_order = np.arange(len(X))
np.random.shuffle(shuffle_order)
X = np.array(X)[shuffle_order]
y = np.array(y)[shuffle_order]

losses = []
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

    X_train_PCA, average_data, projector = PCA(X_train, pc_num=10)
    X_test_PCA = PCA_project(X_test, average_data, projector, pc_num=10)
    X_val_PCA = PCA_project(X_val, average_data, projector, pc_num=10)

    X_train_PCA = standardize(X_train_PCA)
    X_test_PCA = standardize(X_test_PCA)
    X_val_PCA = standardize(X_val_PCA)

    logistic = LogisticRegression()
    theta, loss = logistic.train_batch(
        X_train_PCA, y_train, epoch=50, learning_rate=0.2)

    y_train_predict = logistic.predict(X_train_PCA)
    y_test_predict = logistic.predict(X_test_PCA)
    y_val_predict = logistic.predict(X_val_PCA)
    print('Training Accuracy:', get_accuracy(y_train_predict, y_train))
    print('Testing Accuracy:', get_accuracy(y_test_predict, y_test))
    print('Validation Accuracy:', get_accuracy(y_val_predict, y_val))
    losses.append(loss)

average_losses = [0] * len(losses[0])
for loss in losses:
    for i in range(len(loss)):
        average_losses[i] += loss[i]

average_losses = np.array(average_losses) / len(losses)
print(average_losses)

plt.plot(list(range(len(average_losses))), average_losses)
plt.show()
# 