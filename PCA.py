import numpy as np
import random
from dataloader import *

def split_folder(data_dir=None, classes=None, k_cross_validation=1):
    #################################################################
    #################################################################
    # This function splits the entire dataset into k folders.       #
    # Inputs: data_dir: string, the path to the data folder         #
    #         classes: list, contains the string name of classes    #
    #         k_cross_validation: int, k-fold cross validation      #
    # Output: folds: list, contains the splitted datasets. Each     #
    #                element contains the splitted folds of         #
    #                one of the classes of emotion.                 #
    #################################################################
    #################################################################
    
    # load data and call balanced_sampler to get a balanced dataset 
    dataset, cnt = load_data(data_dir)
    images = balanced_sampler(dataset, cnt, emotions=classes)

    # calculate the fold size for later use
    num_classes = len(classes)
    fold_size = int(len(images[classes[0]]) / k_cross_validation)
    # print("fold size: ", fold_size) 

    folds = []
    # to store the subfolds
    for c in classes:
        folds_c_list = []
    
        # to log the extra samples to average the size of subfold as possible as we cann
        extra_c = []
        # extra images
        extra_c = images[c][k_cross_validation * fold_size:]
        num_extra = len(extra_c)
        # print("num extra: ", num_extra)
        
        # split the dataset into subfolds and average the size of each subfold
        for fold in range(k_cross_validation):
            if fold < num_extra:
                folds_c = images[c][fold * fold_size:(fold+1) * fold_size]
                folds_c.append(extra_c[fold])
                folds_c_list.append(folds_c)


            else:
                folds_c = images[c][fold * fold_size:(fold+1) * fold_size]
                folds_c_list.append(folds_c)
            
            folds.append(folds_c_list)

    return folds

def PCA(train_set=None, num_pc=1, sanity_check=True):
    #################################################################
    #################################################################
    # This function performs PCA calculation.                       #
    # Inputs: training set: list, contains all training images,     #
    #                       splitted into classes                   #
    #         num_pc: int, number of principal components.          #
    #         sanity_check: boolean, if True, check if              #
    #                       PCA is correct.                         #
    # Output: mean_face: array, mean face of training faces         #
    #         eigenvalue_selected: array, num_pc largest eigenvalues#
    #         projector: array, principal axes.                     #
    #################################################################
    #################################################################
    classes = list(train_set.keys())
    all_face = []
    for c in classes:
        all_face += list(train_set[c])
    
    all_face = np.array(all_face)
    num_faces,h,w = all_face.shape[0], all_face.shape[-2], all_face.shape[-1]

    mean_face = np.mean(all_face, axis=0)
    variance_T = all_face - mean_face
    print("variance_T shape: ", variance_T.shape)
    A_T = variance_T.reshape(num_faces,h*w)
    A = np.transpose(A_T,axes=(1,0))
    A = A_T.T
    covariance = np.dot(A_T, A) / num_faces
    eigenvalue, eigenvector = np.linalg.eig(covariance) # Eigen vectors are in column
    print("eigen value: ", eigenvalue[:10])
    eigenvalue_selected = eigenvalue[:num_pc]
    eigenvector_selected = eigenvector[:,:num_pc]
    print("num pc: ", num_pc)
    print("A: ", A.shape)

    # sanity check
    # step 1: 
    if sanity_check == True:
        for i in range(num_pc):
            results = np.dot(A_T, np.dot(A,eigenvector[:,i])) / np.linalg.norm(np.dot(A,eigenvector[:,i]))
            mean_1 = np.mean(results)
            std_1 = np.sqrt(np.sum((results-mean_1)**2)/ num_faces)

            # step 2
            results = results / (eigenvalue[i]**0.5)
            mean_2 = np.mean(results)
            std_2 = np.mean((results-mean_2)**2)
            print("Pass sanity check? ", (mean_1 < 1e-6)&(mean_2 <1e-6)&(np.abs(std_2-1)<1e-6)&(np.abs(std_1-eigenvalue[i]**0.5)<1e-6))
    
    projector = np.dot(A, eigenvector_selected) / np.linalg.norm(np.dot(A, eigenvector_selected))
    print("projector shape: ", projector.shape)
    return mean_face, eigenvalue_selected, projector

def PCA_project(mean_face=None, train_set=None, val_set=None, test_set=None, eigenvalue_selected=None, projector=None):
    #################################################################
    #################################################################
    # This function performs dimensionality reduction.              #
    # Inputs: mean_face: array, mean face of training faces.        #
    #         train_set: list, training set                         #
    #         val_set: list, val set                                #
    #         test_set: list, test set                              #
    #         eigenvalue_selected: array, the largest eigenvalues   #
    #         projector: array, the axes of principal componets     #
    # Outputs: train_set: list, dimensionality reduced training set #
    #          val_set: list, dimensionality reduced val set        #
    #          test_set: list, dimensionality reduced test set      #
    #################################################################
    #################################################################
    classes = list(train_set.keys())
    h,w = train_set[classes[0]].shape[-2], train_set[classes[0]].shape[-1]

    for c in classes:
        train_set[c] = np.dot(train_set[c].reshape(-1,h*w), projector) / eigenvalue_selected**0.5
        val_set[c] = np.dot(val_set[c].reshape(-1,h*w), projector) / eigenvalue_selected**0.5   
        test_set[c] = np.dot(test_set[c].reshape(-1,h*w), projector) / eigenvalue_selected**0.5

    return train_set, val_set, test_set

def mini_batch(train_loader=None, val_loader=None, test_loader=None, batch_size=1):
    train_len = len(train_loader)
    val_len = len(val_loader)
    test_len = len(test_loader)

    train_loader_mini = []
    val_loader_mini = []
    test_loader_mini = []

    # patterns are stored as row vectors here (batch_size, shape)
    for i in range(int(train_len/batch_size)):
        inputs =[]
        targets = []
        for j in range(batch_size):
            # print("data point shape: ", train_loader[i*batch_size+j][0].shape)
            inputs.append(train_loader[i*batch_size+j][0])
            targets.append(train_loader[i*batch_size+j][1])
        train_loader_mini.append((np.array(inputs), np.array(targets)))
    inputs =[]
    targets = []
    for rest in train_loader[((i+1) * batch_size):]:
        inputs.append(rest[0])
        targets.append(rest[1])
    if inputs == []:
        pass
    else: 
        train_loader_mini.append((np.array(inputs),np.array(targets)))
    
    for i in range(int(val_len/batch_size)):
        inputs =[]
        targets = []
        for j in range(batch_size):
            inputs.append(val_loader[i*batch_size+j][0])
            targets.append(val_loader[i*batch_size+j][1])
        val_loader_mini.append((np.array(inputs), np.array(targets)))
    inputs =[]
    targets = []
    for rest in val_loader[((i+1) * batch_size):]:
        inputs.append(rest[0])
        targets.append(rest[1])
    if inputs == []:
        pass
    else: 
        val_loader_mini.append((np.array(inputs),np.array(targets)))
    
    for i in range(int(test_len/batch_size)):
        inputs =[]
        targets = []
        for j in range(batch_size):
            inputs.append(test_loader[i*batch_size+j][0])
            targets.append(test_loader[i*batch_size+j][1])
        test_loader_mini.append((np.array(inputs), np.array(targets)))
    inputs =[]
    targets = []
    for rest in test_loader[((i+1) * batch_size):]:
        inputs.append(rest[0])
        targets.append(rest[1])
    if inputs == []:
        pass
    else: 
        test_loader_mini.append((np.array(inputs),np.array(targets)))

    return train_loader_mini, val_loader_mini, test_loader_mini
            
def return_folder(classes=None, folds=None, fold=0, k_cross_validation=1, num_pc=1, shuffle=True, batch_size=1):
    #################################################################
    #################################################################
    # This function construct the train_loader, the val_loader,     #
    # and the test_loader.                                          #
    # Inputs: class_0: string, the first selected class of emotion  #
    #         class_1: string, the second selected class of emotion #
    #         folds: list, splitted dataset                         #
    #         fold: iteration index                                 #
    # Output: train_loader: dict {target:[images]}                  #
    #         val_loader: dict {target:[images]}                    #
    #         test_loader: dict {target:[images]}                   #
    #################################################################
    #################################################################
    if len(classes) > 2:
        classes_hash = {'anger':0, 'disgust':1, 'fear':2, 
                        'happiness':3, 'sadness':4, 'surprise':5}
    else:
        classes_hash = {classes[0]:0, classes[1]:1}

    # store the images in dicts
    val_set = {}
    test_set = {}
    train_set = {}
    num_classes = len(classes)
    for i in range(num_classes):
        c = classes[i]
        train_set[c] = []

        # first select subfolds for val_loader and test_loader, the rest images for train_loader
        val_set[c] = np.array(folds[i][fold])
        test_set[c] = np.array(folds[i][(fold+1)%k_cross_validation])
        for j in range(k_cross_validation-2):
            if fold == k_cross_validation-1:
                train_set[c] += folds[i][1:-1][j]
                
            else: 
                trainset_c = folds[i][0:fold] + folds[i][fold+2:]
                train_set[c] += trainset_c[j]
        # print("!!!!!!!!!!!",np.array(train_set[c]).shape)
        train_set[c] = np.array(train_set[c])

    train_num = len(train_set[classes[0]])
    val_num = len(val_set[classes[0]])
    test_num = len(test_set[classes[0]])
    
    train_loader = []
    val_loader = []
    test_loader = []

    # PCA preprocessing:
    mean_face, eigenvalues, projector = PCA(train_set, num_pc, False)
    train_set, val_set, test_set = PCA_project(mean_face, train_set, val_set, test_set, eigenvalues, projector)

    for c in classes:
        for i in range(train_num):
            train_loader.append((np.array(list(train_set[c][i])+[1]), classes_hash[c]))
        
        for i in range(val_num):
            val_loader.append((np.array(list(val_set[c][i])+[1]), classes_hash[c]))

        for i in range(test_num):
            test_loader.append((np.array(list(test_set[c][i])+[1]), classes_hash[c]))
    
    if shuffle == True:
        train_loader, val_loader, test_loader = random.sample(train_loader,num_classes*train_num), random.sample(val_loader, num_classes*val_num), random.sample(test_loader,num_classes*test_num)
        train_loader, val_loader, test_loader = mini_batch(train_loader, val_loader, test_loader, batch_size)
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader

