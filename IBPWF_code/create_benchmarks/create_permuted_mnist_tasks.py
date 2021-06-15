import os
from os.path import expanduser

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
from copy import deepcopy
from sklearn.utils import shuffle
HOME_DIR = os.getcwd()

seed = 1731
np.random.seed(seed)
tf.random.set_random_seed(seed)
continual = True

class PermutedMnistGenerator():
	
	def __init__(self, max_iter=5):

		data_dir = "{}/data/MNIST".format(HOME_DIR)
		if not os.path.exists(data_dir):
			os.makedirs(data_dir)

		mnist=tf.keras.datasets.mnist.load_data(data_dir+'/mnist.npz')
		data_train, data_test = mnist
		
		self.X_train = data_train[0] / 255.0
		self.Y_train = data_train[1]
		self.X_test = data_test[0] / 255.0
		self.Y_test = data_test[1]

		self.X_train = np.reshape(self.X_train, [self.X_train.shape[0], -1])
		self.X_test = np.reshape(self.X_test, [self.X_test.shape[0], -1])

		self.max_iter = max_iter
		self.cur_iter = 0
		
		self.num_classes = 10*max_iter
		
	def get_dims(self):
		# Get data input and output dimensions
		return self.X_train.shape[1], 10

	def next_task(self, task_id):
		if self.cur_iter >= self.max_iter:
			raise Exception('Number of tasks exceeded!')
		else:
			np.random.seed(self.cur_iter)
			perm_inds = np.arange(self.X_train.shape[1])
			np.random.shuffle(perm_inds)
			# print (perm_inds[:10])

			# Retrieve train data
			next_x_train = deepcopy(self.X_train)
			next_x_train = next_x_train[:,perm_inds]
			# next_y_train = np.eye(10)[self.Y_train]
			
			# Retrieve test data
			next_x_test = deepcopy(self.X_test)
			next_x_test = next_x_test[:,perm_inds]
			# next_y_test = np.eye(10)[self.Y_test]
						
			if continual:
				next_y_train = np.eye(self.num_classes)[self.Y_train + (task_id*10)]
				next_y_test = np.eye(self.num_classes)[self.Y_test + (task_id*10)]
			else:
				next_y_train = np.eye(10)[self.Y_train]
				next_y_test = np.eye(10)[self.Y_test]

			self.cur_iter += 1

			return next_x_train, next_y_train, next_x_test, next_y_test

num_tasks = 10
data_gen = PermutedMnistGenerator(num_tasks)

task_dir = "{}/data/permuted_mnist_10/".format(HOME_DIR)
if not os.path.exists(task_dir):
	os.makedirs(task_dir)

for task in np.arange(num_tasks):
	x_train, y_train, x_test, y_test = data_gen.next_task(task)
	x_train, y_train = shuffle(x_train, y_train, random_state=seed+task)
	print (x_train.shape, y_train.shape, x_test.shape, y_test.shape)
	print (task_dir)
	print (np.argmax(y_train, axis=1))
	np.savez('{}continual_task_{}.npz'.format(task_dir, task), X_train = x_train, Y_train = y_train, X_test = x_test, Y_test = y_test)
