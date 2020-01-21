from dataloader import *
import numpy as np
import matplotlib.pyplot as plt
import collections
class SoftmaxRegression:

    def __init__(self):
        pass

    def train_batch(self, X, X_test, X_val, y, y_test, y_val, classes, epoch=50, learning_rate=0.001):
        # This function is for Batch Gradient Descent.

        # We pass the parameters to the class attributes
        self.n_samples, self.n_features = X.shape
        self.classes = classes
        self.loss_train = []
        self.loss_test = []
        self.loss_val = []

        # Initialize the weights randomly and initialize the bias term.
        self.theta = np.random.rand(self.classes, self.n_features)
        self.bias = np.zeros((1, self.classes))
        y_one_hot = self.one_hot(y)

        # Training
        for i in range(epoch):
            probs = self.softmax(X)                                         # Compute the probability using softmax function
            y_predict = np.argmax(probs, axis=1)[:, np.newaxis]             # Make preduction based on the maximal probability

            dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))    # Compute the derivative w.r.t. the weights
            self.theta = self.theta - learning_rate * dw.T                  # Update the weights

            db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)   # Compute the derivative w.r.t. the bias term
            self.bias = self.bias - learning_rate * db                      # Update the bias term

            self.loss_train.append(self.cross_entropy(X, y_one_hot))        # Log the loss for training and validation
            self.loss_test.append(self.cross_entropy(                       # Here, we also do test and log the loss
                X_test, self.one_hot(y_test)))                              #
            self.loss_val.append(self.cross_entropy(                        #
                X_val, self.one_hot(y_val)))                                #

        return self.theta, self.bias, self.loss_train, self.loss_test, self.loss_val

    def train_stochastic(self, X, X_test, X_val, y, y_test, y_val, classes, epoch=50, learning_rate=0.1):
        # This function is for stochastic gradient descent
        self.n_samples, self.n_features = X.shape
        self.classes = classes
        self.loss_train = []
        self.loss_test = []
        self.loss_val = []

        self.theta = np.random.rand(self.classes, self.n_features)
        self.bias = np.zeros((1, self.classes))
        y_one_hot = self.one_hot(y)

        # Training
        for i in range(epoch):
            random_order = np.arange(self.n_samples)
            np.random.shuffle(random_order)                                 # Pick sample randomly
            
            # Stochastic Gradient Descent
            for j in range(len(random_order)):
                rand = random_order[j]
                prob = self.softmax(X[rand])

                dw = (1 / self.n_samples) * \
                    np.outer(prob - y_one_hot[rand], X[rand])
                self.theta = self.theta - learning_rate * dw

                db = (1 / self.n_samples) * \
                    np.sum(prob - y_one_hot[rand])
                self.bias = self.bias - learning_rate * db

            self.loss_train.append(self.cross_entropy(X, y_one_hot))
            self.loss_test.append(self.cross_entropy(
                X_test, self.one_hot(y_test)))
            self.loss_val.append(self.cross_entropy(
                X_val, self.one_hot(y_val)))

        return self.theta, self.bias, self.loss_train, self.loss_test, self.loss_val

    def one_hot(self, y):
        # One hot encoding, to help compute the loss and derivatives
        one_hot = np.zeros((len(y), self.classes))
        one_hot[np.arange(len(y)), y.T] = 1
        return one_hot

    def softmax(self, X):
        # Softmax function, to return the probability
        values = np.dot(X, self.theta.T) + self.bias
        shifted_values = values - np.max(values)
        exp_values = np.exp(shifted_values)
        return exp_values / np.sum(exp_values)

    def cross_entropy(self, X, y_one_hot):
        # Compute the Cross Entropy Loss
        ce = - np.mean(np.multiply(y_one_hot, np.log(self.softmax(X))))
        return ce

    def predict(self, X):
        # Make predictions based on the maximal probability
        probs = self.softmax(X)
        return np.argmax(probs, axis=1)[:, np.newaxis]


def cross_fold(X, fold_num):
    # K_fold Cross Validation
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
    # PCA, for dimensionality reduction
    average_data = np.mean(X, axis=0)
    centered_data = X - average_data

    # Turk and Pentland Trick
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
    # Map the raw image data into a lower-dim space
    X_PCA = np.dot(X - average_data, projector[:, 0:pc_num])
    return X_PCA


def standardize(X):
    # Standardize the vector, for numeric stability of computation
    mean = np.mean(X, axis=1)[:, np.newaxis]
    std = np.std(X, axis=1)[:, np.newaxis]
    return (X - mean) / std


def get_accuracy(y_predict, y_true):
    # Compute the accurracy
    match = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_true[i]:
            match += 1
    return match / len(y_predict)

# Main part of the script

# Load data from files
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


losses_train_sto = []
losses_train_bat = []

losses_test = []
losses_val = []
num_folds = 10
num_epoch = 50
folds = cross_fold(X, num_folds)
targets_count = []
preds_count = []
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

    # PCA mapping
    X_train_PCA, average_data, projector = PCA(X_train, pc_num=40)
    X_test_PCA = PCA_project(X_test, average_data, projector, pc_num=40)
    X_val_PCA = PCA_project(X_val, average_data, projector, pc_num=40)

    #X_train_PCA = standardize(X_train_PCA)
    #X_test_PCA = standardize(X_test_PCA)
    #X_val_PCA = standardize(X_val_PCA)

    # Training, Validation, Test
    softmax = SoftmaxRegression()
    theta, bias, loss_train_sto, loss_test, loss_val = softmax.train_stochastic(
        X_train_PCA, X_test_PCA, X_val_PCA, y_train, y_test, y_val, classes=6, epoch=num_epoch, learning_rate=0.1)

    theta, bias, loss_train_bat, loss_test, loss_val = softmax.train_batch(
        X_train_PCA, X_test_PCA, X_val_PCA, y_train, y_test, y_val, classes=6, epoch=num_epoch, learning_rate=0.1)

    losses_train_sto.append(loss_train_sto)
    losses_train_bat.append(loss_train_bat)

    losses_test.append(loss_test)
    losses_val.append(loss_val)

    y_train_predict = softmax.predict(X_train_PCA)
    y_test_predict = softmax.predict(X_test_PCA)
    y_val_predict = softmax.predict(X_val_PCA)
    # print('Training Accuracy:', get_accuracy(y_train_predict.A1, y_train.A1))
    # print('Testing Accuracy:', get_accuracy(y_test_predict.A1, y_test.A1))
    # print('Validation Accuracy:', get_accuracy(y_val_predict.A1, y_val.A1))

    targets_count += list(y_test)
    preds_count += list(y_test_predict)

# confusion_mat = np.zeros((6,6))
# print("confusion mat shape: ", confusion_mat.shape)
# print("len: ", len(targets_count))
# for i in range(len(targets_count)):
#     if targets_count[i] == preds_count[i]:
#         id = targets_count[i].A1[0]
#         confusion_mat[id][id] += 1
#     else:
#         target = targets_count[i]
#         pred = preds_count[i]
#         confusion_mat[target.A1[0]][pred.A1[0]] += 1

# confusion_mat = confusion_mat/len(targets_count)
# print("confusion mat: ", confusion_mat)
# print("sum: ", np.sum(confusion_mat, axis=1))
# a = np.eye(6,6)
# diag = np.sum(confusion_mat * a)
# print("diag: ", diag)


# Process loss lists for visualization
average_train_losses_sto = [0] * len(losses_train_sto[0])
average_train_losses_bat = [0] * len(losses_train_bat[0])

# average_test_losses = [0] * len(losses_test[0])
# average_val_losses = [0] * len(losses_val[0])
std_train_losses_sto = []
std_train_losses_bat = []
# std_test_losses = []
# std_val_losses = []

for i in range(len(losses_train_bat)):
    for j in range(len(losses_train_bat[0])):
        average_train_losses_sto[j] += losses_train_sto[i][j]
        average_train_losses_bat[j] += losses_train_bat[i][j]
        # average_test_losses[j] += losses_test[i][j]
        # average_val_losses[j] += losses_val[i][j]

for i in range(num_epoch):
    std_train_losses_sto.append(np.std(np.array(losses_train_sto)[:, i]))
    std_train_losses_bat.append(np.std(np.array(losses_train_bat)[:, i]))
    
    # std_test_losses.append(np.std(np.array(losses_test)[:, i]))
    # std_val_losses.append(np.std(np.array(losses_val)[:, i]))

average_train_losses_sto = np.array(average_train_losses_sto) / len(losses_train_sto)
average_train_losses_bat = np.array(average_train_losses_bat) / len(losses_train_bat)
# average_test_losses = np.array(average_test_losses) / len(losses_test)
# average_val_losses = np.array(average_val_losses) / len(losses_val)

plt.errorbar(list(range(len(average_train_losses_bat))),
             average_train_losses_bat, yerr=std_train_losses_bat, label='Training Error Batch')

plt.errorbar(list(range(len(average_train_losses_sto))),
             average_train_losses_sto, yerr=std_train_losses_sto, label='Training Error Stochastic')
# plt.errorbar(list(range(len(average_test_losses))),
#              average_test_losses, yerr=std_test_losses, label='Testing Error')
# plt.errorbar(list(range(len(average_val_losses))), average_val_losses,
#              yerr=std_val_losses, label='Validation Error')

# plt.plot(list(range(len(average_train_losses))),
#         average_train_losses, label='Training Error')
# plt.plot(list(range(len(average_test_losses))),
#         average_test_losses, label='Testing Error')
# plt.plot(list(range(len(average_val_losses))),
#         average_val_losses, label='Validation Error')

plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.show()

# eigenfaces = projector[:, 0:40] * theta.T
# for i in range(6):
#     vector = eigenfaces[:, i].A1

#     norm = np.linalg.norm(vector)
#     mean = np.mean(vector)
#     std = np.std(vector)

#     vector = (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
#     vector = vector * 255
#     #display_face(vector.reshape((224, 192)))
