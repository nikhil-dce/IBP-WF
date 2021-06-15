import tensorflow as tf
import numpy as np

def preprocess_image(image, is_training):
	"""Preprocess a single image of layout [height, width, depth]."""
	
	if is_training:

		HEIGHT, WIDTH, NUM_CHANNELS = 32, 32, 3

		# Resize the image to add four extra pixels on each side.
		image = tf.image.resize_image_with_crop_or_pad(
				image, HEIGHT + 8, WIDTH + 8)

		# Randomly crop a [HEIGHT, WIDTH] section of the image.
		image = tf.random_crop(image, [tf.shape(image)[0], HEIGHT, WIDTH, NUM_CHANNELS])

		# Randomly flip the image horizontally.
		image = tf.image.random_flip_left_right(image)

		# Subtract off the mean and divide by the variance of the pixels.
		# Data is already normalized!
		# image = tf.image.per_image_standardization(image)

	return image

def preprocess_mnist_image(image, is_training):
	"""Preprocess a single image of layout [height, width, depth]."""
	
	if is_training:

		HEIGHT, WIDTH, NUM_CHANNELS = 28, 28, 1

		# Resize the image to add four extra pixels on each side.
		image = tf.image.resize_image_with_crop_or_pad(
				image, HEIGHT + 4, WIDTH + 4)

		# Randomly crop a [HEIGHT, WIDTH] section of the image.
		image = tf.random_crop(image, [tf.shape(image)[0], HEIGHT, WIDTH, NUM_CHANNELS])

		# Randomly flip the image horizontally.
		# image = tf.image.random_flip_left_right(image)

		# Subtract off the mean and divide by the variance of the pixels.
		# Data is already normalized!
		# image = tf.image.per_image_standardization(image)

	return image

def one_hot_encode(x, n_classes):
	"""
	One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
	: x: List of sample Labels
	: return: Numpy array of one-hot encoded labels
		"""
	
	# # used when separate heads.
	# x = x % n_classes
	return np.eye(n_classes)[x]