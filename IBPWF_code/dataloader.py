"""Dataset loader to be used during training."""

import numpy as np
from sklearn.utils import shuffle
from utilities import utils

DATA_SEED = 1234

def load_splitMNIST(data_dir):
    
    def get_task(task, dataset, get_valid=False):
        """
            Args:
            
            task: integer task_id
            dataset: Complete dataset
            get_valid: Whether to return validation data. (Basline methods were not using this, so we also ignore it by default).
        """

        d_tr = dataset[0]
        d_te = dataset[1]
        n_input = dataset[2]
        n_output = dataset[3]
        num_tasks = dataset[4]

        X, Y, X_test, Y_test = d_tr[task][1], d_tr[task][2], d_te[task][1], d_te[task][2]
        X, Y = shuffle(X, Y, random_state=DATA_SEED+task)
        X = (X-0.1307)/(0.3081)
        X_test = (X_test-0.1307)/(0.3081)
        # X = 2*(X - 0.5)
        # X_test = 2*(X_test - 0.5)
        # X = np.reshape(X, [-1, 28,28,1])
        # X_test = np.reshape(X_test, [-1, 28,28,1])

        print (X.shape, X_test.shape) 
        print (np.max(X), np.min(X), np.max(X_test), np.min(X_test))
        print (np.max(Y), np.min(Y), np.max(Y_test), np.min(Y_test))

        if get_valid:
            valid_size = int(0.1*X.shape[0])
            print ("Using {:d}/{:d} as the validation set".format(valid_size, X.shape[0]))
            
            X_valid = X[:valid_size]
            Y_valid = Y[:valid_size]
            X = X[valid_size:]
            Y = Y[valid_size:]

            Y = utils.one_hot_encode(Y, n_output)
            Y_valid = utils.one_hot_encode(Y_valid, n_output)
            Y_test = utils.one_hot_encode(Y_test, n_output)
            
            return X, Y, X_valid, Y_valid, X_test, Y_test
        else:
            Y = utils.one_hot_encode(Y, n_output)
            Y_test = utils.one_hot_encode(Y_test, n_output)
            return X, Y, X_test, Y_test

    # d_tr, d_te, n_input, n_output, num_tasks = load_data(data_dir)
    dataset = load_data(data_dir)
    return dataset, get_task

def load_permutedMNIST(data_dir, num_tasks=5):

    def get_task(task, dataset, get_valid=False):

        data_dir = dataset[0]
        data_file = data_dir + str(task) + '.npz'    

        data = np.load(data_file)

        X_train, Y_train = data['X_train'], data['Y_train']
        X_test, Y_test = data['X_test'], data['Y_test']
        
        X_train = 2*(X_train - 0.5)
        X_test = 2*(X_test - 0.5)

        X_train, Y_train = shuffle(X_train, Y_train, random_state=DATA_SEED+task)

        print ('Train Dataset: X:{}, Y:{}'.format(X_train.shape, Y_train.shape))
        print ('Test Dataset: X:{}, Y:{}'.format(X_test.shape, Y_test.shape))
        print (np.max(X_train), np.min(X_train), np.max(X_test), np.min(X_test))
        print (np.max(Y_train), np.min(Y_train), np.max(Y_test), np.min(Y_test))
        
        return X_train, Y_train, X_test, Y_test

    # to be consistent with other datasets
    # get_task for permuted mnist is different
    dataset = (data_dir, None, 784, num_tasks*10, num_tasks)

    return dataset, get_task
    
def load_splitCIFAR10(data_dir):

    def get_task(task_id, dataset, get_valid=False):
	
        d_tr = dataset[0]
        d_te = dataset[1]
        n_input = dataset[2]
        n_output = dataset[3]
        num_tasks = dataset[4]
        
        X, Y, X_test, Y_test = d_tr[task_id][1], d_tr[task_id][2], d_te[task_id][1], d_te[task_id][2]
        X, Y = shuffle(X, Y, random_state=DATA_SEED+task_id)
        
        Y = utils.one_hot_encode(Y, n_output)
        Y_test = utils.one_hot_encode(Y_test, n_output)
        
        X = X.reshape(X.shape[0], 3, 32, 32)
        X = np.transpose(X, (0, 2, 3, 1))
        X = X*255.0
        
        X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
        X_test = np.transpose(X_test, (0, 2, 3, 1))
        X_test = X_test*255.0
        
        # print (np.max(X), np.max(Y), np.max(X_test), np.max(Y_test))
        # print (np.min(X), np.min(Y), np.min(X_test), np.min(Y_test))
        # print (X.shape, Y.shape)
        
        if get_valid:
            valid_size = int(0.1*X.shape[0])
            print ("Using {:d}/{:d} as the validation set".format(valid_size, X.shape[0]))
            
            X_valid = X[:valid_size]
            Y_valid = Y[:valid_size]
            X = X[valid_size:]
            Y = Y[valid_size:]
            return X, Y, X_valid, Y_valid, X_test, Y_test
        else:
            return X, Y, X_test, Y_test

    # Load all data
    # d_tr, d_te, n_input, n_output, num_tasks
    dataset = load_data(data_dir)

    return dataset, get_task
    
def load_data(data_file):
    'Adapted from: https://github.com/rahafaljundi/Gradient-based-Sample-Selection/blob/master/main.py' 
            
    data = np.load(data_file, allow_pickle=True, encoding='latin1')
    
    d_tr = data['tasks_tr']
    d_te = data['tasks_te']
    
    n_inputs = d_tr[0][1].shape[1]
    n_outputs = 0
    for i in range(len(d_tr)):
        n_outputs = max(n_outputs, d_tr[i][2].max())
        n_outputs = max(n_outputs, d_te[i][2].max())
        
    return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)
