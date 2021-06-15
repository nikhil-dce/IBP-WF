import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow as tf
from copy import deepcopy

seed = 1234
np.random.seed(seed)
tf.random.set_random_seed(seed)
n_tasks = 5

def unpickle(file):
    import pickle
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo , encoding='bytes')
    
    return dict

data_dir = os.path.join(os.getcwd(), 'data/cifar10/cifar-10-batches-py/')

if not os.path.exists(data_dir):
    print ('\n\nDownload CIFAR-10 dataset!')
    print ('Use `wget -P data/cifar10 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz` to download the CIFAR-10 dataset and unzip using `tar -C data/cifar10 -xvf cifar-10-python.tar.gz`.')

    sys.exit()
        
for i in range(1, 6):
    fname = data_dir + 'data_batch_' + str(i)
    batch_dict = unpickle(fname)

    print (batch_dict.keys())
    
    if (i == 1):
        x_tr = batch_dict[b'data']
        y_tr = batch_dict[b'labels']
    else:
        x_tr = np.concatenate((x_tr, batch_dict[b'data']), axis=0)
        y_tr = y_tr + batch_dict[b'labels']

y_tr = np.array(y_tr)

# Testing data
fname = data_dir + 'test_batch'
batch_dict = unpickle(fname)

x_te = batch_dict[b'data']
y_te = np.array(batch_dict[b'labels'])

x_tr = x_tr / 255.0
x_te = x_te / 255.0

#########################################################################################

tasks_tr = []
tasks_te = []

cpt = int(10 / n_tasks)

for t in range(n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt

    print (c1, c2)

    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero()
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero()

    print (i_tr)
    print (y_tr[i_tr])
    print (x_tr.shape)

    tasks_tr.append([(c1, c2), x_tr[i_tr], y_tr[i_tr]])
    tasks_te.append([(c1, c2), x_te[i_te], y_te[i_te]])


task_dir = os.path.join(os.getcwd(), 'data/cifar10/')
if not os.path.exists(task_dir):
    os.makedirs(task_dir)

np.savez('{}split_cifar10.npz'.format(task_dir), tasks_tr=tasks_tr, tasks_te=tasks_te)