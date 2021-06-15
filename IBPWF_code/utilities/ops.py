from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math

from utilities import inference_utils

def calculate_gain(nonlinearity='leaky_relu', param=0.1):
	return math.sqrt(2.0/(1.0 + param**2))

def kaiming_uniform_init(shape, fan=None):
	
	if fan == None:
		fan = shape[1]

	gain = calculate_gain()
	stdev = gain / math.sqrt(fan)
	bound = math.sqrt(3.0)
	return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=tf.float32)

def factorized_W(dim_a, dim_b, num_factors, task_id = -1, prior_alpha = 100, **kwargs):
	"""
	Function to create a 2D-WF layer.

	dim_a: input dimension
	dim_b: output dimension
	num_factors: Number of total factors to select from
	prior_alpha: IBP prior parameter
	task_id: The task_id in the continual learning setup
	"""

	assert task_id >= 0, "Task id cannot be less than 0"
	scope_name = tf.get_variable_scope().name
	num_factors = 1000
	
	shape = (dim_a, num_factors)
	init=tf.contrib.layers.xavier_initializer_conv2d()
	W_a = tf.get_variable('W_a', shape=shape, initializer=init, dtype=tf.float32)

	shape = (num_factors, dim_b)
	init=tf.contrib.layers.xavier_initializer_conv2d()
	W_b = tf.get_variable('W_b', shape=shape, initializer=init, dtype=tf.float32)
	
	task_name = 'task_'+ str(task_id)
	# alpha = int(prior_alpha)
	# incr_alpha = int(prior_alpha // 2)
	alpha = 100
	incr_alpha = 100
	if task_id > 0:
		
		factors_available = np.ones((num_factors, 1), dtype=np.float32)
		factors_available[:(alpha+(task_id-1)*incr_alpha)] = 0.0
		mask = factors_available
		mask_T = tf.transpose(mask)

		W_a = inference_utils.stop_gradients(W_a, mask_T)
		W_b = inference_utils.stop_gradients(W_b, mask)

		prev_task_name = 'task_'+ str(task_id-1)
		init_r = tf.stop_gradient(tf.get_variable(prev_task_name+'_r')) 
	else:
		init_r = tf.random_normal((num_factors, 1), mean=0.0, stddev=0.1, dtype=tf.float32) 


	b = np.ones((num_factors, 1), dtype=np.float32)
	b[(alpha+incr_alpha*task_id):] = 0.
	r = tf.get_variable(task_name+'_r', initializer=init_r)
	Z = tf.squeeze(r*b)
	D = tf.matrix_diag(Z)
	W = tf.matmul(tf.matmul(W_a, D), W_b)

	# is_training = kwargs.get('is_training', False)
	# if is_training:

	# 	Ws_dict = {}
	# 	Ws_dict[scope_name+'_W_a'] = W_a
	# 	Ws_dict[scope_name+'_W_b'] = W_b
	# 	tf.add_to_collection(inference_utils.GLOBAL_W_VARS, Ws_dict)

	return W

def factorized_IBP(dim_a, dim_b, num_factors, prior_alpha, task_id = -1, **kwargs):
	"""
	Function to create a 2D-WF layer.

	dim_a: input dimension
	dim_b: output dimension
	num_factors: Number of total factors to select from
	prior_alpha: IBP prior parameter
	task_id: The task_id in the continual learning setup
	"""
	assert task_id >= 0, "Task id cannot be less than 0"
	scope_name = tf.get_variable_scope().name
	
	P_THRESHOLD = kwargs['P_THRESHOLD']
	lambd = kwargs.get('LAMBD')#, inference_utils.LAMBD
	is_training = kwargs.get('is_training', False)
	fine_tuning = kwargs.get('fine_tuning', False)
	in_expectation = kwargs.get('in_expectation', False) # Nonlinear function => Biased estimate!

	shape = (dim_a, num_factors)
	init=tf.contrib.layers.xavier_initializer_conv2d()
	W_a = tf.get_variable('W_a', shape=shape, initializer=init, dtype=tf.float32)

	shape = (num_factors, dim_b)
	init=tf.contrib.layers.xavier_initializer_conv2d()
	W_b = tf.get_variable('W_b', shape=shape, initializer=init, dtype=tf.float32)
	
	task_name = 'task_'+ str(task_id)

	if task_id > 0:
		
		if is_training and scope_name+'_c' not in kwargs:
			print ('Warning: prev_c not provided for training!')
		
		if is_training:
			factors_available = tf.ones((num_factors, 1))
			for t_prev in range(task_id):
				prev_task_name = 'task_'+ str(t_prev)
				prev_pi = tf.stop_gradient(tf.nn.sigmoid(tf.get_variable(prev_task_name+'_pi')))
				factors_available = tf.where(prev_pi >= P_THRESHOLD, tf.zeros((num_factors, 1)), factors_available)
			
			mask = factors_available
			mask_T = tf.transpose(mask)

			W_a = inference_utils.stop_gradients(W_a, mask_T)
			W_b = inference_utils.stop_gradients(W_b, mask)
		
			# mask = 1-factors_available
			# mask_T = tf.transpose(mask)
			# # tf.sqrt and tf.norm return nan when arg is 0. 
			# # See Github issue here: https://github.com/tensorflow/tensorflow/issues/11427
			# # Adding const = 0.001 => This will not change anything.
			# prev_W_a = kwargs.get(scope_name+'_W_a')
			# prev_W_b = kwargs.get(scope_name+'_W_b')
			# norm_Wa = tf.sqrt(0.001+tf.reduce_sum(tf.square(W_a*mask_T - prev_W_a*mask_T)))
			# norm_Wb = tf.sqrt(0.001+tf.reduce_sum(tf.square(W_b*mask - prev_W_b*mask)))
			# norm_W = norm_Wa + norm_Wb
			# tf.add_to_collection(inference_utils.TF_W_NORM, norm_W)

		prev_c = kwargs.get(scope_name+'_c')
		prev_d = kwargs.get(scope_name+'_d')
		prev_task_name = 'task_'+ str(task_id-1)
		# initialize pi
		init_pi = tf.stop_gradient(tf.get_variable(prev_task_name+'_pi'))
		# init_r = tf.stop_gradient(tf.get_variable(prev_task_name+'_r')) # was working fine with splitmnist 99%>
		# init_r = tf.random_normal((num_factors, 1), mean=0.0, stddev=0.01, dtype=tf.float32) # splitMNIST
		init_r = tf.random_normal((num_factors, 1), mean=0.0, stddev=0.1, dtype=tf.float32) # PermutedMNIST
	else:
		init_r = tf.random_normal((num_factors, 1), mean=0.0, stddev=0.1, dtype=tf.float32) # PermutedMNIST
		
		init_pi = np.ones((num_factors, 1), dtype=np.float32)*(prior_alpha / (prior_alpha+1))
		init_pi = np.cumprod(init_pi, axis=0)
		init_pi = np.clip(init_pi, a_min = 0.001, a_max = 0.999) 
		init_pi = np.log(init_pi / (1-init_pi))
		
		prev_c = tf.constant(np.ones((num_factors, 1))*prior_alpha, dtype=np.float32)
		prev_d = tf.constant(np.ones((num_factors, 1)), dtype=np.float32)

	# task-specific r.v.s
	pi_logit = tf.get_variable(task_name+'_pi', initializer=init_pi)
	pi_post = tf.nn.sigmoid(pi_logit)

	r = tf.get_variable(task_name+'_r', initializer=init_r)
	if in_expectation or fine_tuning:
		pi_post = tf.squeeze(pi_post)
		pi_post = tf.where(pi_post >= P_THRESHOLD, pi_post, tf.zeros((num_factors)))
		pi_post = tf.reshape(pi_post, (num_factors,1))
		Z = tf.squeeze(pi_post*r) # remove stochastic nature
	# elif fine_tuning:
	# 	Y_post, b = inference_utils.tf_sample_BernConcrete(pi_post, lambd)
	# 	# do not use the sample, use directly pi_post
	# 	b = pi_post
	# 	Z = tf.squeeze(pi_post*r)
	else:
		# sampling the binary vector
		Y_post, b = inference_utils.tf_sample_BernConcrete(pi_post, lambd)
		Z = tf.squeeze(b*r)

	D = tf.matrix_diag(Z)
	W = tf.matmul(tf.matmul(W_a, D), W_b)

	if is_training:

		if fine_tuning:
			# # Specify prior for the real-valued vector
			# kld_r = tf.reduce_sum(inference_utils.kullback_normal_normal(r, .1, 0.0, .1))
			# tf.add_to_collection(inference_utils.TF_KLR, kld_r)

			# Ws_dict = {}
			# Ws_dict[scope_name+'_W_a'] = W_a
			# Ws_dict[scope_name+'_W_b'] = W_b
			# tf.add_to_collection(inference_utils.GLOBAL_W_VARS, Ws_dict)

			return W


		c = inference_utils.get_tf_pos_variable("c", init_value=prior_alpha, shape=(num_factors, 1))
		d = inference_utils.get_tf_pos_variable("d", init_value=1.0, shape=(num_factors, 1))

		pi_prior_log = inference_utils.tf_stick_breaking_weights(c, d) # sample
		pi_prior_log = tf.reshape(pi_prior_log, (-1,1))
		pi_prior = tf.exp(pi_prior_log)

		# Specify prior for the binary-mask vector
		log_q = inference_utils.tf_log_density_logistic(pi_post, lambd, Y_post)
		log_p = inference_utils.tf_log_density_logistic(pi_prior, lambd, Y_post)
		kld_binary = tf.reduce_sum(log_q-log_p)

		# Specify a prior for the posterior Kumaraswamy distribution using prev_c and prev_d 
		kld_v = tf.reduce_sum(inference_utils.kullback_kumar_kumar(c, d, prev_c, prev_d))
		
		# Specify prior for the real-valued vector
		kld_r = tf.reduce_sum(inference_utils.kullback_normal_normal(r, .1, 0.0, .1))

		# Add KL losses	
		tf.add_to_collection(inference_utils.TF_KLB, kld_binary)
		tf.add_to_collection(inference_utils.TF_KLV, kld_v)
		tf.add_to_collection(inference_utils.TF_KLR, kld_r)

		kumar_dict = {}
		kumar_dict[scope_name+'_c'] = c
		kumar_dict[scope_name+'_d'] = d

		Ws_dict = {}
		Ws_dict[scope_name+'_W_a'] = W_a
		Ws_dict[scope_name+'_W_b'] = W_b

		# Add tensors to tf collection to use for regulaization/online priors.
		tf.add_to_collection(inference_utils.KUMAR_VARS, kumar_dict)
		tf.add_to_collection(inference_utils.GLOBAL_W_VARS, Ws_dict)

	return W

def simple_shared_conv2d(input_, output_dim,
		k_h=3, k_w=3, d_h=2, d_w=2, scope='conv2d', 
		task_id=-1, num_factors=50, prior_alpha=10.0, **kwargs):

	with tf.variable_scope(scope):
	
		dim_a = int (k_h*k_w*input_.get_shape()[-1])
		dim_b = output_dim
		w = factorized_W(dim_a, dim_b, num_factors=num_factors, 
							prior_alpha=prior_alpha, task_id=task_id, **kwargs)
				
		l2_loss = tf.nn.l2_loss(w)
		tf.add_to_collection('L2_LOSS', l2_loss)

		w = tf.reshape(w, (k_h, k_w, input_.get_shape()[-1], output_dim))
				
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

	return conv

def simple_shared_linear(x, output_size, scope=None, bias_start=0.0, task_id=-1,
			num_factors=50, prior_alpha=10.0, **kwargs):
	"""Creates a linear layer.

	Args:
		x: 2D input tensor (batch size, features)
		output_size: Number of features in the output layer
		scope: Optional, variable scope to put the layer's parameters into
		bias_start: The bias parameters are initialized to this value

	Returns:
		The normalized tensor
	"""

	x = tf.concat([x, tf.ones((tf.shape(x)[0], 1))], axis=1)
	shape = x.get_shape().as_list()
	assert len(shape) == 2, "Shape is not 2D"

	with tf.variable_scope(scope or 'Linear') as scope:
		
		dim_a = shape[1]
		dim_b = output_size
		matrix = factorized_W(dim_a, dim_b, num_factors=num_factors, 
								prior_alpha=prior_alpha, task_id=task_id, **kwargs)
				
		l2_loss = tf.nn.l2_loss(matrix)
		tf.add_to_collection('L2_LOSS', l2_loss)
		
		# 	tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
		# bias = tf.get_variable('bias_'+task_name, [output_size], initializer=tf.constant_initializer(bias_start))
	
	# Bias already included in the matrix
	out = tf.matmul(x, matrix)
	# out = tf.matmul(x, matrix) + bias
	return out

def conv2d(input_, output_dim,
		k_h=3, k_w=3, d_h=2, d_w=2, name='conv2d', use_bias=False, 
		task_id=-1, num_factors=50, prior_alpha=10.0, **kwargs):
	"""Creates convolutional layers.

	Args:
		input_: 4D input tensor (batch size, height, width, channel).
		output_dim: Number of features in the output layer.
		k_h: The height of the convolutional kernel.
		k_w: The width of the convolutional kernel.
		d_h: The height stride of the convolutional kernel.
		d_w: The width stride of the convolutional kernel.
		name: The name of the variable scope.
	Returns:
		conv: The normalized tensor.
	"""
	with tf.variable_scope(name) as scope:
		
		dim_a = int (k_h*k_w*input_.get_shape()[-1])
		dim_b = output_dim
		if kwargs.get('is_deterministic', False):
			w = factorized_W(dim_a, dim_b, num_factors=num_factors, 
								prior_alpha=prior_alpha, task_id=task_id, **kwargs)
		else:
			w = factorized_IBP(dim_a, dim_b, num_factors=num_factors, 
								prior_alpha=prior_alpha, task_id=task_id, **kwargs)
				
		l2_loss = tf.nn.l2_loss(w)
		tf.add_to_collection('L2_LOSS', l2_loss)

		w = tf.reshape(w, (k_h, k_w, input_.get_shape()[-1], output_dim))
				
		# w = tf.get_variable(
		# 	'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
		# 	initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
		
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		if use_bias:
			biases = tf.get_variable('biases', [output_dim],
									initializer=tf.zeros_initializer())
			l2_loss = tf.nn.l2_loss(biases)
			tf.add_to_collection('L2_LOSS', l2_loss)
			conv = tf.nn.bias_add(conv, biases)

		return conv

def linear(x, output_size, scope=None, bias_start=0.0, task_id=-1,
			num_factors=50, prior_alpha=10.0, **kwargs):
	"""Creates a linear layer.

	Args:
		x: 2D input tensor (batch size, features)
		output_size: Number of features in the output layer
		scope: Optional, variable scope to put the layer's parameters into
		bias_start: The bias parameters are initialized to this value

	Returns:
		The normalized tensor
	"""
	# assert task_id>=0, "task_id error"

	# shape = x.get_shape().as_list()
	# Adding bias
	x = tf.concat([x, tf.ones((tf.shape(x)[0], 1))], axis=1)
	shape = x.get_shape().as_list()
	assert len(shape) == 2, "Shape is not 2D"

	with tf.variable_scope(scope or 'Linear') as scope:
		
		dim_a = shape[1]
		dim_b = output_size
		if kwargs.get('is_deterministic', False):
			matrix = factorized_W(dim_a, dim_b, num_factors=num_factors, 
								prior_alpha=prior_alpha, task_id=task_id, **kwargs)
		else:
			matrix = factorized_IBP(dim_a, dim_b, num_factors=num_factors, 
								prior_alpha=prior_alpha, task_id=task_id, **kwargs)
			# matrix = factorized_IBP_dynamic_task_selection(dim_a, dim_b, num_factors=num_factors, prior_alpha=prior_alpha, num_tasks=5, task_id=task_id, **kwargs)
		
		l2_loss = tf.nn.l2_loss(matrix)
		tf.add_to_collection('L2_LOSS', l2_loss)
		# w = tf.reshape(w, (k_h, k_w, input_.get_shape()[-1], output_dim))

		# matrix = tf.get_variable(
		# 	'Matrix', [shape[1], output_size], tf.float32,
		# 	tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
		# bias = tf.get_variable('bias_'+task_name, [output_size], initializer=tf.constant_initializer(bias_start))
	
	# Bias already included in the matrix
	out = tf.matmul(x, matrix)
	# out = tf.matmul(x, matrix) + bias
	return out

def simple_linear(x, output_size, scope='', bias_start=0.0, **kwargs):

	shape = x.get_shape().as_list()
	with tf.variable_scope(scope):
		matrix = tf.get_variable('weight', [shape[1], output_size], tf.float32,
			tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(bias_start))
		# Bias already included in the matrix
		out = tf.matmul(x, matrix) + bias

		l2_loss = tf.nn.l2_loss(matrix)
		tf.add_to_collection('L2_LOSS', l2_loss)

		l2_loss = tf.nn.l2_loss(bias)
		tf.add_to_collection('L2_LOSS', l2_loss)

	return out

def simple_conv2d(input_, output_dim,
		k_h=3, k_w=3, d_h=2, d_w=2, scope='conv2d', use_bias=False, 
		**kwargs):
	"""Creates convolutional layers.

	Args:
		input_: 4D input tensor (batch size, height, width, channel).
		output_dim: Number of features in the output layer.
		k_h: The height of the convolutional kernel.
		k_w: The width of the convolutional kernel.
		d_h: The height stride of the convolutional kernel.
		d_w: The width stride of the convolutional kernel.
		name: The name of the variable scope.
	Returns:
		conv: The normalized tensor.
	"""

	with tf.variable_scope(scope) as scope:
				
		w = tf.get_variable(
			'w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
		
		l2_loss = tf.nn.l2_loss(w)
		tf.add_to_collection('L2_LOSS', l2_loss)
		
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		# if use_bias:
		# 	biases = tf.get_variable('biases', [output_dim],
		# 							initializer=tf.zeros_initializer())
		# 	l2_loss = tf.nn.l2_loss(biases)
		# 	tf.add_to_collection('L2_LOSS', l2_loss)
		# 	conv = tf.nn.bias_add(conv, biases)

		return conv

def simple_conv2d_transpose(input_, output_dim, output_shape,
		k=4, d_h=2, d_w=2, scope='conv2d',  use_bias=False,
		**kwargs):
	"""Creates convolutional layers.

	Args:
		input_: 4D input tensor (batch size, height, width, channel).
		output_dim: Number of features in the output layer.
		k: The height-width of the convolutional kernel.
		d_h: The height stride of the convolutional kernel.
		d_w: The width stride of the convolutional kernel.
		name: The name of the variable scope.
	Returns:
		conv: The normalized tensor.
	"""

	with tf.variable_scope(scope) as scope:
				
		w = tf.get_variable(
			'w', [k, k, output_dim, input_.get_shape()[-1]], initializer=tf.contrib.layers.xavier_initializer())

		l2_loss = tf.nn.l2_loss(w)
		tf.add_to_collection('L2_LOSS', l2_loss)
		
		conv = tf.nn.conv2d_transpose(input_, filter=w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding='SAME')

		# if use_bias:
		# 	biases = tf.get_variable('biases', [output_dim],
		# 							initializer=tf.zeros_initializer())
		# 	l2_loss = tf.nn.l2_loss(biases)
		# 	tf.add_to_collection('L2_LOSS', l2_loss)
		# 	conv = tf.nn.bias_add(conv, biases)

		return conv


def batch_norm(x, is_training, name='bn_layer', momentum=0.99, data_format='NHWC', task_id=-1):

	assert task_id >= 0, "task_id is {}<0".format(task_id)
	task_name = '_task_'+ str(task_id)

	if data_format == 'NHWC':
		bn_layer = tf.layers.BatchNormalization(momentum=momentum, center=True, scale=True, axis=-1, name=name+task_name)
	elif data_format == 'NCHW':
		bn_layer = tf.layers.BatchNormalization(momentum=momentum, center=True, scale=True, axis=1, name=name+task_name)

	return bn_layer(x, training=is_training)

def avg_pool(x, pool_size, stride, data_format = 'channels_last'):

	"https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/model_base.py"

	with tf.name_scope('avg_pool') as name_scope:
		x = tf.layers.average_pooling2d(x, pool_size, stride, 'SAME', data_format=data_format)

	# tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())

	return x

def global_avg_pool(x, data_format='channels_last'):
	
	"https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/model_base.py"

	with tf.name_scope('global_avg_pool') as name_scope:
		assert x.get_shape().ndims == 4
		if data_format=='channels_first':
			x = tf.reduce_mean(x, [2, 3])
		else:
			x = tf.reduce_mean(x, [1, 2])

	# tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
	
	return x

def factorized_IBP_dynamic_task_selection(dim_a, dim_b, num_factors, prior_alpha, num_tasks, task_id = -1, **kwargs):
	"""
	Function to create a 2D-WF layer.

	dim_a: input dimension
	dim_b: output dimension
	num_factors: Number of total factors to select from
	prior_alpha: IBP prior parameter
	task_id: The task_id in the continual learning setup
	"""
	# assert task_id >= 0, "Task id cannot be less than 0"
	scope_name = tf.get_variable_scope().name
	
	P_THRESHOLD = kwargs['P_THRESHOLD']
	lambd = kwargs.get('LAMBD')#, inference_utils.LAMBD
	is_training = kwargs.get('is_training', False)
	fine_tuning = kwargs.get('fine_tuning', False)
	in_expectation = kwargs.get('in_expectation', False) # Nonlinear function => Biased estimate!

	shape = (dim_a, num_factors)
	init=tf.contrib.layers.xavier_initializer_conv2d()
	W_a = tf.get_variable('W_a', shape=shape, initializer=init, dtype=tf.float32)

	shape = (num_factors, dim_b)
	init=tf.contrib.layers.xavier_initializer_conv2d()
	W_b = tf.get_variable('W_b', shape=shape, initializer=init, dtype=tf.float32)
	
	# task_name = 'task_'+ str(task_id)

	# task-specific r.v.s
	
	P_all = []
	for t in range(num_tasks):
		init_pi = np.ones((num_factors, 1), dtype=np.float32)*(prior_alpha / (prior_alpha+1))
		init_pi = np.cumprod(init_pi, axis=0)
		init_pi = np.clip(init_pi, a_min = 0.001, a_max = 0.999) 
		init_pi = np.log(init_pi / (1-init_pi))

		t_name = 'task_'+ str(t)
		pi_logit = tf.get_variable(t_name+'_pi', initializer=init_pi)
		P_all.append(pi_logit)


	task_one_hot = tf.one_hot([task_id], num_tasks)
	
	P_all = tf.squeeze(tf.stack(P_all, axis=1))
	P_all = tf.nn.sigmoid(P_all)
	P_all = inference_utils.stop_gradients(P_all, task_one_hot)
	pi_post = tf.reduce_sum(P_all*task_one_hot, axis=1)

	init_r = tf.random_normal((num_factors, num_tasks), mean=0.0, stddev=0.1, dtype=tf.float32)
	R_all = tf.get_variable('task_r', initializer=init_r)
	R_all = inference_utils.stop_gradients(R_all, task_one_hot)
	r = tf.reduce_sum(R_all*task_one_hot, axis=1)
		
	print (P_all.get_shape())
	print (R_all.get_shape())
	print (task_one_hot.shape)
	print (r.get_shape(), pi_post.get_shape())
	sys.exit()
	
	if in_expectation:
		pi_post = tf.squeeze(pi_post)
		pi_post = tf.where(pi_post >= P_THRESHOLD, pi_post, tf.zeros((num_factors)))
		pi_post = tf.reshape(pi_post, (num_factors,1))
		Z = tf.squeeze(pi_post*r) # remove stochastic nature
	elif fine_tuning:
		Y_post, b = inference_utils.tf_sample_BernConcrete(pi_post, lambd)
		# do not use the sample, use directly pi_post
		b = pi_post
		Z = tf.squeeze(pi_post*r)
	else:
		# sampling the binary vector
		Y_post, b = inference_utils.tf_sample_BernConcrete(pi_post, lambd)
		Z = tf.squeeze(b*r)
		
	D = tf.matrix_diag(Z)
	W = tf.matmul(tf.matmul(W_a, D), W_b)

	if task_id > 0:
		
		if is_training and scope_name+'_c' not in kwargs:
			print ('Warning: prev_c not provided for training!')
		
		if is_training:
			factors_available = tf.ones((num_factors, 1))
			for t_prev in range(task_id):
				prev_task_name = 'task_'+ str(t_prev)
				prev_pi = tf.stop_gradient(tf.nn.sigmoid(tf.get_variable(prev_task_name+'_pi')))
				factors_available = tf.where(prev_pi >= P_THRESHOLD, tf.zeros((num_factors, 1)), factors_available)
			
			mask = factors_available
			mask_T = tf.transpose(mask)

			W_a = inference_utils.stop_gradients(W_a, mask_T)
			W_b = inference_utils.stop_gradients(W_b, mask)
		
		prev_c = kwargs.get(scope_name+'_c')
		prev_d = kwargs.get(scope_name+'_d')
		prev_task_name = 'task_'+ str(task_id-1)
		
		# # initialize pi
		# init_pi = tf.stop_gradient(tf.get_variable(prev_task_name+'_pi'))
		# # init_r = tf.stop_gradient(tf.get_variable(prev_task_name+'_r')) # was working fine with splitmnist 99%>
		# # init_r = tf.random_normal((num_factors, 1), mean=0.0, stddev=0.01, dtype=tf.float32) # splitMNIST
		# init_r = tf.random_normal((num_factors, 1), mean=0.0, stddev=0.1, dtype=tf.float32) # PermutedMNIST
	else:
		# init_r = tf.random_normal((num_factors, 1), mean=0.0, stddev=0.1, dtype=tf.float32) # PermutedMNIST
		# init_pi = np.ones((num_factors, 1), dtype=np.float32)*(prior_alpha / (prior_alpha+1))
		# init_pi = np.cumprod(init_pi, axis=0)
		# init_pi = np.clip(init_pi, a_min = 0.001, a_max = 0.999) 
		# init_pi = np.log(init_pi / (1-init_pi))
		
		prev_c = tf.constant(np.ones((num_factors, 1))*prior_alpha, dtype=np.float32)
		prev_d = tf.constant(np.ones((num_factors, 1)), dtype=np.float32)


	if is_training:

		c = inference_utils.get_tf_pos_variable("c", init_value=prior_alpha, shape=(num_factors, 1))
		d = inference_utils.get_tf_pos_variable("d", init_value=1.0, shape=(num_factors, 1))

		pi_prior_log = inference_utils.tf_stick_breaking_weights(c, d) # sample
		pi_prior_log = tf.reshape(pi_prior_log, (-1,1))
		pi_prior = tf.exp(pi_prior_log)

		# Specify prior for the binary-mask vector
		log_q = inference_utils.tf_log_density_logistic(pi_post, lambd, Y_post)
		log_p = inference_utils.tf_log_density_logistic(pi_prior, lambd, Y_post)
		kld_binary = tf.reduce_sum(log_q-log_p)

		# Specify a prior for the posterior Kumaraswamy distribution using prev_c and prev_d 
		kld_v = tf.reduce_sum(inference_utils.kullback_kumar_kumar(c, d, prev_c, prev_d))
		
		# Specify prior for the real-valued vector
		kld_r = tf.reduce_sum(inference_utils.kullback_normal_normal(r, .1, 0.0, .1))

		# Add KL losses	
		tf.add_to_collection(inference_utils.TF_KLB, kld_binary)
		tf.add_to_collection(inference_utils.TF_KLV, kld_v)
		tf.add_to_collection(inference_utils.TF_KLR, kld_r)

		kumar_dict = {}
		kumar_dict[scope_name+'_c'] = c
		kumar_dict[scope_name+'_d'] = d

		Ws_dict = {}
		Ws_dict[scope_name+'_W_a'] = W_a
		Ws_dict[scope_name+'_W_b'] = W_b

		# Add tensors to tf collection to use for regulaization/online priors.
		tf.add_to_collection(inference_utils.KUMAR_VARS, kumar_dict)
		tf.add_to_collection(inference_utils.GLOBAL_W_VARS, Ws_dict)

	return W