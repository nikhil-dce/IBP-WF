import tensorflow as tf
from .base import Base

from utilities import ops
from utilities.ops import conv2d
from utilities.utils import preprocess_image
from utilities.inference_utils import *

class WideResNet_IBP_WF(Base):

	def __init__(self, config):
		Base.__init__(self, config)

		# Build graph
		data_dim = config['data_dim']
		W, H, C = data_dim[0], data_dim[1], data_dim[2]
		
		self.X_ph = tf.placeholder(shape=[None, W, H, C], dtype=tf.float32)
		self.Y_ph = tf.placeholder(shape=[None, config['num_classes']], dtype=tf.float32)
		self.lambd = tf.placeholder(dtype=tf.float32)
			
		self.num_factors = config['num_factors']
		self.weight_decay = config['weight_decay']
		self.num_classes = config['num_classes']
		self.momentum = config['momentum']
		self.use_bias = config['bias']
		self.prior_alphas = config['prior_alpha']
		self.P_THRESHOLD = config['kappa']
		self.separate_heads = config['separate_heads']
		self.is_deterministic = config.get('is_deterministic', False)

		num_f = config['num_f']
		assert num_f == 16, "num_f is expected to be 16. Found {}".format(num_f)

		width = 2
		self.num_blocks = config['num_blocks']
		self.filters = [num_f*1, num_f*1*width, num_f*2*width, num_f*4*width]
		self.strides = [1,2,2]

	def res_func(self, x, kernel_size, in_filter,
				out_filter, stride, activate_before_residual=False, 
				task_id=-1, num_factors=200, prior_alpha=50.0, name='residual_v1', **kwargs):
		
		is_training = kwargs.get('is_training', False)

		assert task_id >= 0, "task_id cannot be less than 0."

		with tf.variable_scope(name) as name_scope:
			num_layers = 2
			x = ops.batch_norm(x, is_training, momentum=self.momentum, name='bn1', task_id=task_id)
			x = tf.nn.relu(x)
	 		
			if stride != 1 or in_filter != out_filter:
				shortcut = ops.conv2d(x, out_filter, k_w=1, k_h=1, d_h=stride, d_w=stride, name='shortcut', use_bias=False, task_id=task_id, num_factors=100, prior_alpha=prior_alpha, **kwargs)
				num_layers += 1
			else:
				shortcut = x
				# shortcut = ops.avg_pool(shortcut, stride, stride)
				# pad = (out_filter - in_filter) // 2
				# shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [pad, pad]])
		 
			x = ops.conv2d(x, out_filter, k_w=kernel_size, k_h=kernel_size, d_h=stride, d_w=stride, name='conv1', use_bias=self.use_bias, task_id=task_id, num_factors=num_factors,prior_alpha=prior_alpha, **kwargs)

			x = ops.batch_norm(x, is_training, momentum=self.momentum, name='bn2', task_id=task_id)
			x = tf.nn.relu(x)

			x = ops.conv2d(x, out_filter, k_w=kernel_size, k_h=kernel_size, d_h=1, d_w=1, name='conv2', use_bias=self.use_bias, task_id=task_id, num_factors=num_factors, prior_alpha=prior_alpha, **kwargs)

			x = tf.add(x, shortcut)

			print('Block NumLayers {}'.format(num_layers))
			print('image after unit {}: {}'.format(x.name, x.get_shape()))

		return x
	
	# X_ph, Y_ph, is_training_ph, 
	def forward(self, task_id=-1, reuse=False, **kwargs):
		
		self.is_training = kwargs.get('is_training', True)
		self.expectation =  kwargs.get('in_expectation', False)
		kwargs['P_THRESHOLD'] = self.P_THRESHOLD
		kwargs['LAMBD'] = self.lambd
		kwargs['is_deterministic'] = self.is_deterministic
		fine_tuning = kwargs.get('fine_tuning', False)
		
		print ('Fine Tuning: {}'.format(fine_tuning))
		if self.is_training:
			X_ph = preprocess_image(self.X_ph, True)
			Y_ph = self.Y_ph
			
			if not fine_tuning:
				graph = tf.get_default_graph()
				graph.clear_collection(TF_KLB)
				graph.clear_collection(TF_KLV)
				graph.clear_collection(TF_KLR)
				graph.clear_collection(KUMAR_VARS)
				graph.clear_collection(GLOBAL_W_VARS)
				graph.clear_collection(TF_W_NORM)
				graph.clear_collection('L2_LOSS')
		else:
			X_ph = self.X_ph
			Y_ph = self.Y_ph

		x, y, is_training = X_ph, Y_ph, self.is_training
		print ('\nTraining Mode: {}'.format(is_training))
		
		assert task_id >= 0, "task_id cannot be less than 0."
		
		# standardization
		x = x / 128. - 1 
		with tf.variable_scope('resnet_arch', reuse=reuse):        
			self.conv1_x = conv2d(x, self.filters[0], k_h=3, k_w=3, d_h=1, d_w=1, name='conv1', use_bias=self.use_bias, task_id=task_id, num_factors=self.num_factors[0], prior_alpha=self.prior_alphas[0], **kwargs)
			x = self.conv1_x
						
			# 3 stages of blocking
			for i in range(3):
				with tf.name_scope('stage_' + str(i)):
					for block_ix in range(self.num_blocks):
						if (block_ix) == 0:
							# first block of ith stage
							x = self.res_func(x, 3, self.filters[i], self.filters[i+1], self.strides[i], task_id=task_id, num_factors=self.num_factors[i+1], name='residual_v1_'+str(i*self.num_blocks+block_ix), prior_alpha=self.prior_alphas[i+1], **kwargs)
						else:
							# rest of the blocks for the ith stage
							x = self.res_func(x, 3, self.filters[i+1], self.filters[i+1], 1,task_id=task_id, num_factors=self.num_factors[i+1], name='residual_v1_'+str(i*self.num_blocks+block_ix),prior_alpha=self.prior_alphas[i+1], **kwargs)

			x = ops.batch_norm(x, is_training, momentum=self.momentum, name='bn_last',task_id=task_id)
			x = tf.nn.relu(x)
	 
			x = ops.global_avg_pool(x)
			print ('Last Layer of the architecture: ' + str(x.get_shape()))

			if self.separate_heads:
				scope = 'output_task{}'.format(task_id)
				log_odds = ops.simple_linear(x, self.num_classes, scope=scope, **kwargs)
			else:
				prior_alpha = self.prior_alphas[-1]
				log_odds = ops.linear(x, self.num_classes, task_id=task_id, scope='output',
								num_factors=self.num_factors[0], prior_alpha=prior_alpha,
								**kwargs)

			NLL = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=log_odds)
		
		return NLL, log_odds
	
	@staticmethod
	def get_phi(model, sess, task_id, x, num_repeats=1):
		
		print ('@static: get_phi')
		# assert model.in_expectation, "In expectation not true"
		assert num_repeats > 0, "num_repeats invalid"
		
		_ = model.forward(task_id, reuse=True, is_training=False, in_expectation=True)

		for i in range(num_repeats):
			feed_dict = {model.X_ph: x} 
			# if FLAGS.use_conv1:
			Hi = sess.run(model.conv1_x, feed_dict=feed_dict)
			# else:
			# 	Hi = sess.run(model.pre_x, feed_dict=feed_dict)
			# print ('Hi shape:',Hi.shape)
			if i == 0:
				H = Hi
			else:
				H = np.concatenate([H, Hi], axis=0)

		# H.shape -> (10000, 32, 32, 16)
		H = np.reshape(H, (H.shape[0], -1, H.shape[-1])) # (10000, 1024, 16)
		H = np.mean(H, axis=1) # (10000, 16)

		return H