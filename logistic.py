from dataloader import *
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self):
        pass

    def train_batch(self, X, X_test, X_val, y, y_test, y_val, epoch=50, learning_rate=0.1, threshold=0.5):
        # This function is for Batch Gradient Descent
        
        # Pass the parameters to the class attributes  
        self.samples, self.features = X.shape
        self.threshold = threshold
        self.loss_train = []
        self.loss_test = []
        self.loss_val = []

        # Add bias term to the data matrix and initialize the weights
        X = np.insert(X, 0, 1, axis=1)
        self.theta = np.zeros(self.features + 1)


        # Training
        for i in range(epoch):
            probs = self.sigmoid(X.dot(self.theta))                 # Compute outputs of sigmoid function, this is probability
            dw = (1 / self.samples) * np.dot(X.T, (probs - y))      # Compute the derivative of the loss funciton w.r.t. weights
            self.theta = self.theta - learning_rate * dw            # Update the weights

            self.loss_train.append(self.cost(X, y))                 # Log the losses for visualization
            self.loss_test.append(self.cost(X_test, y_test))        # Here, we logged train loss, val loss
            self.loss_val.append(self.cost(X_val, y_val))           # and did test as well.

        return self.theta, self.loss_train, self.loss_test, self.loss_val

    def train_stochastic(self, X, X_test, X_val, y, y_test, y_val, epoch=50, learning_rate=0.1, threshold=0.5):
        # This function is for Stochastic Gradient Descent

        # Pass the parameters to class attributes
        self.samples, self.features = X.shape
        self.threshold = threshold
        self.loss_train = []
        self.loss_test = []
        self.loss_val = []

        # Add bias term to the data matrix and initialize the weights
        X = np.insert(X, 0, 1, axis=1)
        self.theta = np.random.rand(self.features + 1)

        # Training and logging the losses
        for i in range(epoch):
            random_order = np.arange(self.samples)
            np.random.shuffle(random_order)

            for j in range(len(random_order)):
                rand = random_order[j]                                          # Pick data points randomly
                prob = self.sigmoid(X[rand].dot(self.theta))                    # Compute outputs of model
                dw = (1 / self.samples) * np.dot(X[rand].T, (prob - y[rand]))   # Compute derivatives w.r.t. the weights
                self.theta = self.theta - learning_rate * dw                    # Update the weights

            self.loss_train.append(self.cost(X, y))                             # Log the losses
            self.loss_test.append(self.cost(X_test, y_test))
            self.loss_val.append(self.cost(X_val, y_val))

        return self.theta, self.loss_train, self.loss_test, self.loss_val

    def sigmoid(self, x):
        # Compute the sigmoid output
        return 1 / (1 + np.exp(-x))

    def cost(self, X, y):
        # Compute the Binary Cross Entropy loss of the Logistic Regression model 
        try:
            probs = self.sigmoid(X.dot(self.theta))
        except:
            X = np.insert(X, 0, 1, axis=1)
            probs = self.sigmoid(X.dot(self.theta))

        cost = -(np.sum(y * np.log(probs) + (1-y) * np.log(1 - probs)))
        return cost / len(X)

    def predict(self, X):
        # Make predictions based on the maximal probability

        X = np.insert(X, 0, 1, axis=1)  # insert column of value 1 for bias
        probs = self.sigmoid(X.dot(self.theta))
        return np.round(probs).astype(int)


def cross_fold(X, fold_num):
    # K_fold Cross Validation, split data into K folds
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
        vector = np.dot(A, eigenvectors[:, i]).flatten()
        # if i < 4:
        #    display_face(vector.reshape((224, 192)))

        normalized_vector = (
            vector / (np.linalg.norm(vector) * (eigenvalues[i] ** 0.5)))
        projector.append(normalized_vector)
    projector = np.array(projector).T

    X_PCA = np.dot(X - average_data, projector[:, 0:pc_num])

    return X_PCA, average_data, projector


def PCA_project(X, average_data, projector, pc_num=1):
    # PCA mapping, dimensionality reduction
    X_PCA = np.dot(X - average_data, projector[:, 0:pc_num])
    return X_PCA


def standardize(X):
    # Standardize the vector, for numeric stability of computation 
    mean = np.mean(X, axis=1)[:, np.newaxis]
    std = np.std(X, axis=1)[:, np.newaxis]
    return (X - mean) / std


def get_accuracy(y_predict, y_true):
    # Compute the accurracy
    if len(y_predict) != len(y_true):
        return 0

    match = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_true[i]:
            match += 1
    return match / len(y_predict)


# Main part of the script

# Load data from files
data_dir = "./aligned/"
dataset, cnt = load_data(data_dir)
images = balanced_sampler(dataset, cnt, emotions=['happiness', 'anger'])

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

losses_train = []
losses_test = []
losses_val = []
num_folds = 10
num_epoch = 100
folds = cross_fold(X, num_folds)
# Cross Validation
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
    X_train_PCA, average_data, projector = PCA(X_train, pc_num=10)
    X_test_PCA = PCA_project(X_test, average_data, projector, pc_num=10)
    X_val_PCA = PCA_project(X_val, average_data, projector, pc_num=10)

    X_train_PCA = standardize(X_train_PCA)
    X_test_PCA = standardize(X_test_PCA)
    X_val_PCA = standardize(X_val_PCA)

    logistic = LogisticRegression()
    theta, loss_train, loss_test, loss_val = logistic.train_stochastic(
        X_train_PCA, X_test_PCA, X_val_PCA, y_train, y_test, y_val, epoch=num_epoch, learning_rate=0.5)

    losses_train.append(loss_train)
    losses_test.append(loss_test)
    losses_val.append(loss_val)

    y_train_predict = logistic.predict(X_train_PCA)
    y_test_predict = logistic.predict(X_test_PCA)
    y_val_predict = logistic.predict(X_val_PCA)
    print('Training Accuracy:', get_accuracy(y_train_predict, y_train))
    print('Testing Accuracy:', get_accuracy(y_test_predict, y_test))
    print('Validation Accuracy:', get_accuracy(y_val_predict, y_val))

average_train_losses = [0] * len(losses_train[0])
average_test_losses = [0] * len(losses_test[0])
average_val_losses = [0] * len(losses_val[0])
std_train_losses = []
std_test_losses = []
std_val_losses = []
for i in range(len(losses_train)):
    for j in range(len(losses_train[0])):
        average_train_losses[j] += losses_train[i][j]
        average_test_losses[j] += losses_test[i][j]
        average_val_losses[j] += losses_val[i][j]

for i in range(num_epoch):
    std_train_losses.append(np.std(np.array(losses_train)[:, i]))
    std_test_losses.append(np.std(np.array(losses_test)[:, i]))
    std_val_losses.append(np.std(np.array(losses_val)[:, i]))

average_train_losses = np.array(average_train_losses) / len(losses_train)
average_test_losses = np.array(average_test_losses) / len(losses_test)
average_val_losses = np.array(average_val_losses) / len(losses_val)

plt.errorbar(list(range(len(average_train_losses))),
             average_train_losses, yerr=std_train_losses, label='Training Error')
plt.errorbar(list(range(len(average_test_losses))),
             average_test_losses, yerr=std_test_losses, label='Testing Error')
plt.errorbar(list(range(len(average_val_losses))), average_val_losses,
             yerr=std_val_losses, label='Validation Error')

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
