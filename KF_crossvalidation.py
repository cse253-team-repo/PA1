from dataloader import *

def split_folder(data_dir=None, class_1=None, class_2=None, k=1):
    dataset, cnt = load_data(data_dir)
    images = balanced_sampler(dataset, cnt, emotions=[class_1,class_2])
    batch_size = int(len(images[class_1]) / k)
    print("batch size: ", batch_size) 
    folds_class1_list = []
    folds_class2_list = []
    
    extra_class1 = []
    extra_class2 = []
    # extra images
    extra_class1 = images[class_1][k * batch_size:]
    extra_class2 = images[class_2][k * batch_size:]
    num_extra = len(extra_class1)
    print("num extra: ", num_extra)
    
    for fold in range(k):
        if fold < num_extra:
            folds_class1 = images[class_1][fold * batch_size:(fold+1) * batch_size]
            folds_class1.append(extra_class1[fold])
            folds_class1_list.append(folds_class1)

            folds_class2 = images[class_2][fold * batch_size:(fold+1) * batch_size]
            folds_class2.append(extra_class2[fold])
            folds_class2_list.append(folds_class2)

        else:
            folds_class1 = images[class_1][fold * batch_size:(fold+1) * batch_size]
            folds_class1_list.append(folds_class1)

            folds_class2 = images[class_2][fold * batch_size:(fold+1) * batch_size]
            folds_class2_list.append(folds_class2)

    return folds_class1_list, folds_class2_list

if __name__ == '__main__':
    # The relative path to your image directory
    data_dir = "./aligned/"
    class_1 = 'happiness'
    class_2 = 'anger'
    k = 10
    subset = split_folder(data_dir,class_1,class_2,k)
