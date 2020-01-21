import numpy as np
import random
import argparse

from dataloader import *
from PCA import *
from model import *
from optimizer import *
from loss import *


    
def train(args):
    folds = split_folder(args.data_dir, args.classes, args.k_cross_validation)
    optimizer = Optimizer(args.learning_rate)
    if len(args.classes) == 2:
        classifier = LR_Classifier(args.num_pc)
        criterion = BinaryCrossEntropyLoss()

    else:
        classifier = None
        criterion = None

    acc_list = []
    for fold in range(args.k_cross_validation):
        train_loader, val_loader, test_loader = return_folder(classes=args.classes, folds=folds, fold=fold,
                                                              k_cross_validation=args.k_cross_validation, num_pc=args.num_pc, 
                                                              shuffle=True, batch_size=args.batch_size)
        
        for epoch in range(args.epoch):
            for i, (inputs, targets) in enumerate(train_loader):
                print("inputs shape: ", inputs.shape)
                outputs, predictions = classifier.forward(inputs)
                loss = criterion.compute(outputs,targets)
                grad = criterion.backward(inputs)
                # new_weights = optimizer.update(classifier.Linear1, grad)
                # print(classifier.Linear1[:3])
                classifier.Linear1 -= args.learning_rate * grad
                acc_list.append(np.mean(predictions==targets))
            acc = np.mean(np.array(acc_list))
            print("acc: ", acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./aligned/")
    parser.add_argument('--classes', type=list , default=['happiness', 'anger'])

    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--k_cross_validation', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_pc', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=100)
    args = parser.parse_args()
    print(args)
    train(args)
