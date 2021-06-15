import tensorflow as tf
import tensorflow_probability as tfp

from .base import Base
from utilities import utils
from utilities import ops


class CIFAR_VAE(Base):

	def __init__(self, config):

		data_dim = config['data_dim']
		W, H, C = data_dim[0], data_dim[1], data_dim[2]
		
		self.X_ph = tf.placeholder(shape=[None, W, H, C], dtype=tf.float32)
		self.is_training_ph = tf.placeholder(dtype=tf.bool)

		# self.beta = config.get('beta', 1.0)
		self.z_dim = config.get('z_dim', 64)
		self.nf = 32
		self.nf_ext = 4
		self.momentum = config['momentum']
		
		self.h_dim = [self.nf, 2*self.nf, self.nf*4]
		self.h_dim_ext = [self.nf_ext, 2*self.nf_ext, self.nf_ext]
		self.feature_volume = self.nf*2*(W // 4)*(H // 4)

		self.task_kl_loss = {}
		self.task_recon_loss = {}

	
	def forward(self, x, task_id=-1, reuse=False, **kwargs):
		"""Constructs the CIFAR10 VAE graph."""

		graph = tf.get_default_graph()
		graph.clear_collection('L2_LOSS')

		# standardization
		x = x / 128. - 1 
		# x = x_preprocessed / 255.0

		z_mean, z_log_var = self._encoder_new(x, task_id, reuse, **kwargs)
		z = self.reparameterize(z_mean, z_log_var)
		x_recon_logits = self._decoder_new(z, task_id, reuse, **kwargs)	
		x_recon_logits_flatten = tf.reshape(x_recon_logits, [tf.shape(x)[0], -1])
		
		x = tf.reshape(x, [tf.shape(x)[0], -1])	
		# self.recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_recon_logits_flatten), axis=1)
		recon_loss = -tfp.distributions.MultivariateNormalDiag(loc=x_recon_logits_flatten, scale_identity_multiplier=0.05).log_prob(x)

		kl_loss = -.5 * tf.reduce_sum(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis=1)

		self.task_kl_loss[task_id] = kl_loss
		self.task_recon_loss[task_id] = recon_loss

		# x_recon = tf.nn.sigmoid(x_recon_logits)	
		x_recon = x_recon_logits
		return x_recon, z

	def encoder_shared(self, x, task_id, reuse, **kwargs):
		
		with tf.variable_scope('encoder', reuse=reuse):
			
			h1 = ops.simple_shared_conv2d(x, self.h_dim[0], k_h=3, k_w=3, scope='conv1', task_id=task_id, prior_alpha=50, num_factors=200, **kwargs)
			h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
			h1 = tf.nn.relu(h1)

			h2 = ops.simple_shared_conv2d(h1, self.h_dim[1], k_h=3, k_w=3, scope='conv2', task_id=task_id, num_factors=200, prior_alpha=50, **kwargs)
			h2 = tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
			h2 = tf.nn.relu(h2)

			h3 = tf.reshape(h2, [tf.shape(h2)[0], self.feature_volume])
			h3 = ops.simple_shared_linear(h3, self.h_dim[-1], scope='fc', task_id=task_id, num_factors=400, prior_alpha=100)

		z_mu = ops.simple_linear(h3, self.z_dim, scope='z_mu_{}'.format(task_id), **kwargs)
		z_logvar = ops.simple_linear(h3, self.z_dim, scope='z_logvar_{}'.format(task_id), **kwargs)

		return z_mu, z_logvar

	# def encoder_recursive_shared(self, x, task_id, reuse, return_features=False, **kwargs):
		
	# 	if task_id == 0:
	# 		C_out = 



	def _encoder_new(self, x, task_id, reuse, **kwargs):
		
		with tf.variable_scope('encoder_{}'.format(task_id), reuse=reuse):
			
			h1 = ops.simple_conv2d(x, self.h_dim[0], k_h=3, k_w=3, scope='conv1')
			h1 = tf.nn.max_pool(h1, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
			h1 = tf.nn.relu(h1)

			h2 = ops.simple_conv2d(h1, self.h_dim[1], k_h=3, k_w=3, scope='conv2')
			h2 = tf.nn.max_pool(h2, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
			h2 = tf.nn.relu(h2)

			h3 = tf.reshape(h2, [tf.shape(h2)[0], self.feature_volume])
			h3 = ops.simple_linear(h3, self.h_dim[-1], scope='fc')
			h3 = tf.nn.relu(h3)

		z_mu = ops.simple_linear(h3, self.z_dim, scope='z_mu_{}'.format(task_id), **kwargs)
		z_logvar = ops.simple_linear(h3, self.z_dim, scope='z_logvar_{}'.format(task_id), **kwargs)

		return z_mu, z_logvar

	def _decoder_new(self, z, task_id, reuse, **kwargs):

		is_training = self.is_training_ph

		with tf.variable_scope('decoder_{}'.format(task_id), reuse=tf.AUTO_REUSE):

			h = ops.simple_linear(z, self.h_dim[-1], scope='fc')
			h = tf.nn.relu(h)
			
			h = ops.simple_linear(h, self.feature_volume, scope='fc_volume')
			h = tf.reshape(h, [tf.shape(h)[0], 8, 8, self.h_dim[1]])
			h = tf.nn.relu(h)

			h = ops.simple_conv2d_transpose(h, self.h_dim[0], output_shape=[tf.shape(h)[0], 16, 16, self.h_dim[0]], k=4, scope='conv1')
			h = tf.nn.relu(h)

			h = ops.simple_conv2d_transpose(h, 3, output_shape=[tf.shape(h)[0], 32, 32, 3], k=4, scope='conv3')
			x_recon = tf.nn.tanh(h)

		return x_recon
		
	def sample(self, num_samples=20):
		z = tf.random_normal(shape=(num_samples, self.z_dim))
		
	def reparameterize(self, mu, logvar):
		'''Perform "reparametrization trick" to make these stochastic variables differentiable.'''

		std = tf.exp(0.5*logvar)
		z = mu + std*tf.random_normal(shape=(tf.shape(std)[0], self.z_dim))
		return z

	# def _encoder(self, x, task_id, reuse, **kwargs):
		
	# 	is_training = self.is_training_ph
				
	# 	with tf.variable_scope('encoder'):
	# 		h = ops.simple_conv2d(x, self.h_dim[0], k_h=4, k_w=4, scope='conv1')
	# 		h = tf.nn.relu(h)

	# 		h = ops.simple_conv2d(h, self.h_dim[1], k_h=4, k_w=4, scope='conv2')
	# 		h = tf.nn.relu(h)
			
	# 		h = tf.reshape(h, [tf.shape(h)[0], self.feature_volume])
	# 		h = ops.simple_linear(h, self.h_dim[-1], scope='fc')
		
	# 	z_mu = ops.simple_linear(h, self.z_dim, scope='z_mu', **kwargs)
	# 	z_logvar = ops.simple_linear(h, self.z_dim, scope='z_logvar', **kwargs)

	# 	return z_mu, z_logvar

	# def _decoder(self, z, task_id, reuse, **kwargs):

	# 	is_training = self.is_training_ph

	# 	with tf.variable_scope('decoder'):

	# 		h = ops.simple_linear(z, self.h_dim[-1], scope='fc')
	# 		h = ops.simple_linear(h, self.feature_volume, scope='fc_volume')
	# 		h = tf.reshape(h, [tf.shape(h)[0], 8, 8, self.h_dim[1]])
	# 		h = tf.nn.relu(h)

	# 		h = ops.simple_conv2d_transpose(h, self.h_dim[0], output_shape=[tf.shape(h)[0], 16, 16, self.h_dim[0]], k=4, scope='conv1')
	# 		h = tf.nn.relu(h)

	# 		h = ops.simple_conv2d_transpose(h, 3, output_shape=[tf.shape(h)[0], 32, 32, 3], k=4, scope='conv3')
	# 		# x_recon = h
	# 		x_recon = tf.nn.tanh(h)

	# 	return x_recon