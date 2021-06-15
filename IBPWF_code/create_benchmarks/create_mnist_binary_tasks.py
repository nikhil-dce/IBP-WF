import os
from os.path import expanduser

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow as tf

seed = 1234
np.random.seed(seed)
tf.random.set_random_seed(seed)

data_dir = os.path.join(os.getcwd(), "data/MNIST")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
mnist=tf.keras.datasets.mnist.load_data(path=data_dir+'/mnist.npz')

data_train, data_test = mnist
n_tasks = 5

##################################################################33

tasks_tr = []
tasks_te = []

x_tr, y_tr = (data_train[0], data_train[1])
x_te, y_te = (data_test[0], data_test[1])

x_tr = np.reshape(x_tr, [x_tr.shape[0], -1])
x_te = np.reshape(x_te, [x_te.shape[0], -1])

x_tr = x_tr / 255.0
x_te = x_te / 255.0

print (x_tr.shape, y_tr.shape)
print (x_te.shape, y_te.shape)

cpt = int(10 / n_tasks)

for t in range(n_tasks):
    c1 = t * cpt
    c2 = (t + 1) * cpt

    print (c1, c2)

    i_tr = ((y_tr >= c1) & (y_tr < c2)).nonzero()
    i_te = ((y_te >= c1) & (y_te < c2)).nonzero()

    print (i_tr)
    print (y_tr[i_tr])

    tasks_tr.append([(c1, c2), x_tr[i_tr], y_tr[i_tr]])
    tasks_te.append([(c1, c2), x_te[i_te], y_te[i_te]])

task_dir = os.path.join(os.getcwd(), 'data/binary_mnist/')
if not os.path.exists(task_dir):
    os.makedirs(task_dir)

np.savez('{}split_mnist.npz'.format(task_dir), tasks_tr=tasks_tr, tasks_te=tasks_te)