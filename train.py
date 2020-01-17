import numpy as np
import random
from dataloader import *
from PCA import *
from model import *

    
def train(data_dir=None, classes=None, split_folder=None, return_folder=None, k_cross_validation=1, num_pc=10, batch_size=1):
    folds = split_folder(data_dir, classes, k_cross_validation)
    if len(classes) == 2:
        classifier = LR_Classifier(num_pc)
    else:
        classifier = None
        
    for fold in range(k):
        train_loader, val_loader, test_loader = return_folder(classes=classes, folds=folds, fold=fold,
                                                              k_cross_validation=k, num_pc=num_pc, 
                                                              shuffle=True, batch_size=batch_size)
        for i, (inputs, targets) in enumerate(train_loader):
            print("intpus shape: ", inputs.shape)
            print("targets shape: ", targets.shape)
            outputs = classifier.forward(inputs)
            classifier.backward(inputs,outputs,targets)
            print("outptushape: ", outputs.shape)
            pass

if __name__ == '__main__':
    data_dir = "./aligned/"
    classes = ['happiness', 'anger']
    batch_size = 2
    k = 10
    num_pc = 40
    train(data_dir=data_dir, classes=classes, split_folder=split_folder, return_folder=return_folder, k_cross_validation=k, num_pc=num_pc, batch_size=batch_size)