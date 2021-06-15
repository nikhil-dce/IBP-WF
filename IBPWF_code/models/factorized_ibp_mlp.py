import tensorflow as tf
from .base import Base

from utilities.ops import *
from utilities.utils import preprocess_mnist_image
from utilities.inference_utils import *

class MLP_IBP_WF(Base):

	def __init__(self, config):
		Base.__init__(self, config)
		self.model_arch = config['model_arch']
		self.num_layers = len(self.model_arch)
		self.P_THRESHOLD = config['kappa']

		self.num_factors = config['num_factors']
		self.weight_decay = config['weight_decay']
		self.prior_alpha = config['prior_alpha']
		self.is_deterministic = config.get('is_deterministic', False)

		self.regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay) if self.weight_decay > 0 else None
		
		self.task_infer_layer = config['task_infer_layer']
		self.separate_heads = config['separate_heads']
		# self.output_dim = config['output_dim']

		self.X_ph = tf.placeholder(shape=[None, config['data_dim']], dtype=tf.float32)
		# self.X_ph = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
		
		self.Y_ph = tf.placeholder(shape=[None, config['num_classes']], dtype=tf.float32)
		# self.Y_ph = tf.placeholder(shape=[None, self.output_dim], dtype=tf.float32)

		self.lambd = tf.placeholder(dtype=tf.float32)
		# self.task_id = tf.placeholder(dtype=tf.int32)

	def get_layer_name(self, ix):
		return "layer_{:d}".format(ix)

	def forward(self, task_id=-1, reuse=False, **kwargs):
		
		self.is_training = kwargs.get('is_training', True)
		self.expectation =  kwargs.get('in_expectation', False)
		kwargs['P_THRESHOLD'] = self.P_THRESHOLD
		kwargs['LAMBD'] = self.lambd
		kwargs['is_deterministic'] = self.is_deterministic
		fine_tuning = kwargs.get('fine_tuning', False)
		
		print ('Fine Tuning: {}'.format(fine_tuning))
		if self.is_training and not fine_tuning:
			# x = preprocess_mnist_image(self.X_ph, True)
			# y = self.Y_ph
			graph = tf.get_default_graph()
			graph.clear_collection(TF_KLB)
			graph.clear_collection(TF_KLV)
			graph.clear_collection(TF_KLR)
			graph.clear_collection(KUMAR_VARS)
			graph.clear_collection(GLOBAL_W_VARS)
			graph.clear_collection(TF_W_NORM)
			graph.clear_collection('L2_LOSS')
		# else:
		# 	x = self.X_ph
		# 	y = self.Y_ph
		
		# x = tf.reshape(x, [tf.shape(x)[0], 784])
		# is_training = self.is_training
		x, y, is_training = self.X_ph, self.Y_ph, self.is_training

		prior_alpha = self.prior_alpha
		num_factors = self.num_factors

		print ('\nTraining Mode: {}'.format(is_training))
		
		assert task_id >= 0, "task_id cannot be less than 0."
		
		Hs = []
		with tf.variable_scope('mlp_arch', reuse=reuse):        
			
			H = x
            
			for ix, L in enumerate(self.model_arch):
				
				if ix < self.num_layers-1:
					H = linear(H, L, regularizer=self.regularizer,
								task_id = task_id, scope=self.get_layer_name(ix), 
								num_factors=num_factors, prior_alpha=prior_alpha,
								**kwargs)
					Hs.append(H)
					H = tf.nn.relu(H)
				else:
					if self.separate_heads:
						scope =  self.get_layer_name(ix) + '_task{}'.format(task_id)
						log_odds = simple_linear(H, L, scope=scope, **kwargs)
						# log_odds = simple_linear(H, self.output_dim, scope=scope, **kwargs)
					else:
						log_odds = linear(H, L, regularizer=self.regularizer,
									task_id = task_id, scope=self.get_layer_name(ix), 
									num_factors=num_factors, prior_alpha=prior_alpha,
									**kwargs)

		self.phi = Hs[self.task_infer_layer]
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
			
			Hi = sess.run(model.phi, feed_dict=feed_dict)
						
			if i == 0:
				H = Hi
			else:
				H = np.concatenate([H, Hi], axis=0)

		return H

	def compute_number_of_parameters(self, sess, num_tasks, threshold):
		
		variables_names = [v.name for v in tf.trainable_variables()]
		values = sess.run(variables_names)
		for k, v in zip(variables_names, values):
				print ("Variable: ", k)
				print ("Shape: ", v.shape)

		return len(values)