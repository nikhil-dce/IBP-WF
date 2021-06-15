import io
import os
import pprint
import sys
import time
import yaml

from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import dataloader
from models import MODEL
from models import cifar_vae as vae
from utilities.inference_utils import *
from utilities.estimator_utils import *

parser = ArgumentParser()
parser.add_argument('--config', '-c', default='configs/splitCIFAR10_ibpWF.yaml')
parser.add_argument('--load_ckpt', '-l', default=0)
parser.add_argument('--setting', '-s', default='continual')

def get_task_uncertainty(sess, model, config, x_te, task_id, ensemble_len=100):
	
	assert task_id >= 0

	print ('\nGetting uncertainty for task {}\n'.format(task_id))

	def softmax(z):
		
		assert len(z.shape) == 2
		
		s = np.max(z, axis=1)
		s = s[:, np.newaxis] # necessary step to do broadcasting
		e_x = np.exp(z - s)
		div = np.sum(e_x, axis=1)
		div = div[:, np.newaxis] # dito
		
		return e_x / div

	_, logits = model.forward(task_id=task_id, reuse=True, is_training=False, in_expectation=False)
	
	lambd = config['final_lambd']
	for ix in range(ensemble_len):
		feed_dict = {model.X_ph: x_te, model.lambd: lambd}
		np_logits_t = sess.run(logits, feed_dict=feed_dict)
		
		np_probs_t = softmax(np_logits_t)

		sample_entropy = -np.sum(np_probs_t * np.log(np_probs_t + 1e-7), axis=1)
		
		sample_entropy = np.expand_dims(sample_entropy, axis=1)
		
		np_probs_t = np.expand_dims(np_probs_t, axis=1)

		if ix == 0:
			mean_entropy = sample_entropy
			ensemble_out = np_probs_t
		else:
			ensemble_out = np.concatenate([ensemble_out, np_probs_t], axis=1)
			mean_entropy = np.concatenate([mean_entropy, sample_entropy], axis=1)
	
	
	mean_entropy = np.mean(mean_entropy, axis=1, keepdims=True)
	probs_out = np.mean(ensemble_out, axis=1)
	
	pred_entropy = -1 * np.sum(probs_out * np.log(probs_out + 1e-7), axis=1, keepdims=True)
	
	# mi_entropy = pred_entropy - mean_entropy

	return pred_entropy

def predict_task_using_uncertainty(sess, model, config, x_te, task_list=[0]):

	
	for task_id in task_list:
		
		M = get_task_uncertainty(sess, model, config, x_te, task_id, ensemble_len=100)

		if task_id == 0:
			M_score_list = M
		else:
			M_score_list = np.concatenate([M_score_list, M], axis=1)

	return M_score_list

def predict_label(sess, model, config, x_te, y_te, task_latents, task_list, ensemble_len=1):
	
	def softmax(z):
		assert len(z.shape) == 2
		s = np.max(z, axis=1)
		s = s[:, np.newaxis] # necessary step to do broadcasting
		e_x = np.exp(z - s)
		div = np.sum(e_x, axis=1)
		div = div[:, np.newaxis] # dito
		return e_x / div

	np_logits = None
	print (ensemble_len)
	lambd = config['final_lambd']
	fine_tuning = config.get('fine_tuning', False)
	for t in task_list:
		x1_te = x_te[np.where(task_latents == t)]
		y1_te = y_te[np.where(task_latents == t)]
		
		print (x1_te.shape[0])
		if x1_te.shape[0] > 0:
			_, logits_test = model.forward(task_id=t, reuse=True, is_training=False, in_expectation=False, fine_tuning=fine_tuning)
			
			for ix in range(ensemble_len):
				feed_dict = {model.X_ph: x1_te, model.lambd: lambd}
				np_logits_t = sess.run(logits_test, feed_dict=feed_dict)
				
				# check 
				np_logits_t = softmax(np_logits_t)
				
				np_logits_t = np.expand_dims(np_logits_t, axis=1)

				if ix == 0:
					ensemble_logits = np_logits_t
				else:
					ensemble_logits = np.concatenate([ensemble_logits, np_logits_t], axis=1)
			
			np_logits_t = np.mean(ensemble_logits, axis=1)
			
		if t == 0:
			np_logits = np_logits_t
			yy_te = y1_te
		else:
			np_logits = np.concatenate([np_logits, np_logits_t], axis=0)
			yy_te = np.concatenate([yy_te, y1_te], axis=0)
	
	avg_continual_acc = np_accuracy(np_logits, yy_te)
	return avg_continual_acc

def ProbYgivenXt(sess, model, x_te, task_list, ensemble_len=1):
	
	def softmax(z):
		assert len(z.shape) == 2
		s = np.max(z, axis=1)
		s = s[:, np.newaxis] # necessary step to do broadcasting
		e_x = np.exp(z - s)
		div = np.sum(e_x, axis=1)
		div = div[:, np.newaxis] # dito
		return e_x / div

	np_acc = None
	print ('Ensemble: {}'.format(ensemble_len))
	lambd = config['final_lambd']

	for t in task_list:
		x1_te = x_te
		
		if x1_te.shape[0] > 0:
			_, logits_test = model.forward(task_id=t, reuse=True, is_training=False, in_expectation=False)
			
			for ix in range(ensemble_len):
				feed_dict = {model.X_ph: x1_te, model.lambd: lambd}
				np_acc_t = sess.run(logits_test, feed_dict=feed_dict)
				
				# check 
				np_acc_t = softmax(np_acc_t)
				
				np_acc_t = np.expand_dims(np_acc_t, axis=1)

				if ix == 0:
					ensemble_acc = np_acc_t
				else:
					ensemble_acc = np.concatenate([ensemble_acc, np_acc_t], axis=1)
			
			np_acc_t = np.mean(ensemble_acc, axis=1)
		
		np_acc_t = np.expand_dims(np_acc_t, axis=1)

		if t == 0:
			np_acc = np_acc_t
		else:
			np_acc = np.concatenate([np_acc, np_acc_t], axis=1)
			
	return np_acc

def predict_continual_task(sess, model, config, x_te, task_list=[0], est_list=[]):
	"""
	task_list should start with 0 !!
	TODO: Implement for general task_lists as well
	"""
	# from scipy import linalg

	# assert len(est_list) > 0
	
	# final_cov = 0.0
	# N = 0.
	# for ix, est in enumerate(est_list):
	# 	final_cov += est.covariance_ * est.n
	# 	N += est.n

	# final_cov /= N
	# final_precision = linalg.pinvh(final_cov)	

	# from sklearn.utils.extmath import fast_logdet as logdet
	def score(est, h):

		mu, precision, cov, n = est.location_, est.precision_, est.covariance_, est.n
		log_cov = logdet(cov + 1e-5*np.eye(cov.shape[0]))
		print (log_cov)
		# log_prec = logdet(precision)
		score = 0.
		# print (n)
		score += -0.5*np.sum(np.matmul(h - mu, precision) * (h-mu), axis=1) - 0.5*log_cov + np.log(n)
		# score += -0.5*np.sum(np.matmul(h - mu, final_precision) * (h-mu), axis=1) + np.log(n)
		return score

	for task_id in task_list:
		Hi = MODEL[config['model_name']].get_phi(model, sess, 0, x_te)

		# Hi = get_phi(task_id, X_valid, x_te, num_repeats=1)
		est = est_list[task_id]
		M = score(est, Hi)
		print ('All Score Shape: ', M.shape)
		M = np.expand_dims(M, axis=1)
		if task_id == 0:
			M_score_list = M
		else:
			M_score_list = np.concatenate([M_score_list, M], axis=1)

	print (M_score_list.shape)

	return M_score_list

def compute_elbo(config, sess, vae_model, x_te, task_id):

	with tf.Graph().as_default():

		feed_dict = {vae_model.X_ph: x_te, vae_model.is_training_ph: False}
		elbo = -vae_model.task_recon_loss[task_id] - vae_model.task_kl_loss[task_id]
	
		average_elbo = None
		for sample_ix in range(40):
			
			np_elbo = sess.run(elbo, feed_dict=feed_dict)
			np_elbo = np_elbo[:, np.newaxis]
			if sample_ix == 0:
				average_elbo = np_elbo	
			else:
				average_elbo = np.concatenate([average_elbo, np_elbo], axis=1)
		
		average_elbo = np.mean(average_elbo, axis=1)
		# print ('Task {0:d} \tVal LogELBO {1:.4f}'.format(task_id+1, np.mean(average_elbo)))

	print (average_elbo.shape)
	return average_elbo

def predict_continual_task_using_vae(config, x_te, task_list):

	result_dir = os.path.join(config['parent_dir'], 'vae_ckpts')
	save_path = os.path.join(result_dir, "g_{:d}.ckpt".format(4))

	with tf.Graph().as_default():

		vae_model = vae.CIFAR_VAE(config)
		for task_id in task_list:
			reuse = tf.AUTO_REUSE if task_id > 0 else False
			vae_model.forward(vae_model.X_ph, task_id, reuse=reuse)

		sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(sess, save_path)
		print ('Loaded VAE Checkpoint {}'.format(save_path))

		for task_id in task_list:
			M = compute_elbo(config, sess, vae_model, x_te, task_id)
			M = np.expand_dims(M, axis=1)
			if task_id == 0:
				M_score_list = M
			else:
				M_score_list = np.concatenate([M_score_list, M], axis=1)
		
		sess.close()

	return M_score_list


def continual_test_using_pretrained_vae(sess, model, config, x_te, y_te, t_gt, task_list, ensemble_len=100):

	plot_uncertainty = False
	M_score_list = predict_continual_task_using_vae(config, x_te, task_list)
	task_latents = np.argmax(M_score_list, axis=1)
	task_acc = np.mean(t_gt == task_latents)
	print (task_acc)
	# sys.exit()
	avg_continual_acc = predict_label(sess, model, config, x_te, y_te, task_latents, task_list, ensemble_len=ensemble_len)

	return task_acc, avg_continual_acc

def continual_test(sess, model, config, x_te, y_te, t_gt, task_list, est_list, version='task', setting='continual', ensemble_len=100):

	plot_uncertainty = False

	M_score_list = predict_continual_task(sess, model, config, x_te, task_list, est_list)
	task_latents = np.argmax(M_score_list, axis=1)
	task_acc = np.mean(t_gt == task_latents)
	print (task_acc)
	if plot_uncertainty:
		print ('Task Acc: {}'.format(task_acc))
		np_acc = ProbYgivenXt(sess, model, x_te, task_list, ensemble_len=100)
		task_probs = M_score_list
		print (np_acc.shape)
		print (task_probs.shape)
		np.savez('uncertainty_plot', t_gt=t_gt, np_acc=np_acc, task_probs=task_probs)
		sys.exit(0)

	avg_continual_acc = predict_label(sess, model, config, x_te, y_te, task_latents, task_list, ensemble_len=ensemble_len)

	return task_acc, avg_continual_acc

def continual_test_using_uncertainty (sess, model, config, x_te, y_te, t_gt, task_list, est_list, version='task', setting='continual', ensemble_len=100):

	# M_score_list = predict_continual_task(sess, model, config, x_te, task_list, est_list)
	# predict_task_using_uncertainty(sess, model, x_te, task_list=[0]):
	M_score_list = predict_task_using_uncertainty(sess, model, config, x_te, task_list)
	print (M_score_list.shape)
	task_latents = np.argmin(M_score_list, axis=1)
	task_acc = np.mean(t_gt == task_latents)
	print (task_acc)
	np.savez('MI_uncertainty_plot', t_gt=t_gt, H=M_score_list)
	sys.exit()
	avg_continual_acc = predict_label(sess, model, config, x_te, y_te, task_latents, task_list, ensemble_len=ensemble_len)

	return task_acc, avg_continual_acc

def multi_task_test(sess, model, config, x_te, y_te, t_gt, task_list, batch_size=256, ensemble_len=100):

	def softmax(z):
		assert len(z.shape) == 2
		s = np.max(z, axis=1)
		s = s[:, np.newaxis] # necessary step to do broadcasting
		e_x = np.exp(z - s)
		div = np.sum(e_x, axis=1)
		div = div[:, np.newaxis] # dito
		return e_x / div

	print ('\nSetting: MultiTask')
	result_str = ''
	all_acc = []
	lambd = config['final_lambd']
	fine_tuning = config.get('fine_tuning', False)
	for t in task_list:
		x1_te = x_te[np.where(t_gt == t)]
		y1_te = y_te[np.where(t_gt == t)]
		
		if x1_te.shape[0] > 0:
			_, logits_test = model.forward(task_id=t, reuse=True, is_training=False, in_expectation=False, fine_tuning=fine_tuning)
			
			for ix in range(ensemble_len):
				feed_dict = {model.X_ph: x1_te, model.lambd: lambd}
				np_logits_t = sess.run(logits_test, feed_dict=feed_dict)
				
				#TODO: check 
				np_logits_t = softmax(np_logits_t)

				np_logits_t = np.expand_dims(np_logits_t, axis=1)
								
				if ix == 0:
					ensemble_logits = np_logits_t
				else:
					ensemble_logits = np.concatenate([ensemble_logits, np_logits_t], axis=1)
			
			np_logits_t = np.mean(ensemble_logits, axis=1)
		
		np_acc = np_accuracy(np_logits_t, y1_te)
		all_acc.append(np_acc)

	return all_acc

def test(dataset, get_task, config, setting='continual'):
	num_tasks = dataset[4]
	net_out = dataset[-2]
	config['num_classes'] = net_out

	# create model
	model = MODEL[config['model_name']](config)
	sess = None
	kwargs = dict()
	est_list = []
	task_ground_truth = []
	tf_phis = []

	kwargs['fine_tuning'] = config.get('fine_tuning', False)
	for task_id in range(num_tasks):
		
		_, _, X_test, Y_test = get_task(task_id, dataset)
		
		if task_id == 0:
			x_te = X_test
			y_te = Y_test
			task_ground_truth = np.zeros((X_test.shape[0], 1))
		else:
			x_te = np.concatenate([x_te, X_test], axis=0)
			y_te = np.concatenate([y_te, Y_test], axis=0)
			task_ground_truth = np.concatenate([task_ground_truth, task_id*np.ones((X_test.shape[0], 1))], axis=0)

		reuse = tf.AUTO_REUSE if task_id > 0 else False
		 # check if this works for task_id > 0
		NLL_loss, logits = model.forward(task_id, reuse, is_training=False, **kwargs)

		# H = model.pre_x
		# tf_phis.append(H)
				
	task_ground_truth = task_ground_truth.squeeze()
	# save_estimators(config, est_list)
	
	save_path = config['result_dir'] + "continual_model_{:d}.ckpt".format(config['seed'])
	saver = tf.train.Saver()
	sess = tf.Session()
	saver.restore(sess, save_path)
	print ('\nModel Restored')

	# Count the number of parameters.
	# number_parameters = model.compute_number_of_parameters(sess, num_tasks, threshold=0.1)
	# print ('\nNumber parameters: {:d}'.format(number_parameters))
	# sys.exit()

	setting = config['setting']
	ensemble_len = config.get('pred_ensemble_len', 100)
	
	if setting == 'continual':

		task_infer_method = config.get('task_infer_method', 'phi_gaussian')
		
		if task_infer_method == 'vae':
			task_acc, avg_continual_acc = continual_test_using_pretrained_vae(sess, model, config, x_te, y_te, task_ground_truth, np.arange(num_tasks), ensemble_len=ensemble_len)
		elif task_infer_method == 'uncertainty':
			task_acc, avg_continual_acc = continual_test_using_uncertainty(sess, model, config, x_te, y_te, task_ground_truth, np.arange(num_tasks), est_list, ensemble_len=ensemble_len)
		else:
			est_list = load_estimators(config)
			print ('\nEstimators loaded')
			task_acc, avg_continual_acc = continual_test(sess, model, config, x_te, y_te, task_ground_truth, np.arange(num_tasks), est_list, ensemble_len=ensemble_len)
		
		print ('\nSetting: Continual')
		result_str = '\nSetting: Continual using {}'.format(task_infer_method)
		result_str += '\nTask Acc: {:.6f}\n'.format(task_acc)
		result_str += 'Testing Task: Avg Accuracy {:.6f}\n'.format(avg_continual_acc)
	
	else:
		
		all_acc = multi_task_test(sess, model, config, x_te, y_te, task_ground_truth, np.arange(num_tasks), ensemble_len=ensemble_len)
		print ('\nSetting: Multi-Task')

		result_str = '\nSetting: MultiTask'
		for t, np_acc in enumerate(all_acc):
			result_str += '\nTask: {:d} \tTesting - \tAvg Accuracy {:.6f}\n'.format(t+1, np_acc)
		
		result_str += '\nAvg Accuracy {:.6f}\n'.format(np.mean(all_acc))
	
	print (result_str)

	return result_str

def main():
	args = parser.parse_args()

	# Load config
	config_path = args.config
	config = yaml.load(open(config_path), Loader=yaml.FullLoader)
	os.environ['CUDA_VISIBLE_DEVICES'] = config['gpuid']

	if not os.path.exists(config['result_dir']):
		raise ValueError('The dir: ' + config['result_dir'] + ' does not exist.')

	print ('\n')
	pprint.pprint (config)
	print ('\n')
	# sys.exit()

	config['setting'] = args.setting
	if config['dataset'] == 'splitMNIST':
		dataset, get_task = dataloader.load_splitMNIST(config['dataset_path'])
		result_str = test(dataset, get_task, config)
	elif config['dataset'] == 'splitFashionMNIST':
		dataset, get_task = dataloader.load_splitFashionMNIST(config['dataset_path'])
		result_str = test(dataset, get_task, config)
	elif config['dataset'] == 'splitCIFAR10':
		dataset, get_task = dataloader.load_splitCIFAR10(config['dataset_path'])
		result_str = test(dataset, get_task, config)
	elif config['dataset'] == 'splitCIFAR100':
		dataset, get_task = dataloader.load_splitCIFAR10(config['dataset_path'])
		result_str = test(dataset, get_task, config)
	elif config['dataset'] == 'permutedMNIST':
		dataset, get_task = dataloader.load_permutedMNIST(config['dataset_path'], config['num_tasks'])
		result_str = test(dataset, get_task, config)
	else:
		raise ValueError('Dataset: {} not found.'.format(config['dataset']))
	
	result_fname = config['setting']+ '_alpha' + str(config['prior_alpha']) + '_kappa'+str(config['kappa'])+'.txt'
	task_infer_method = config.get('task_infer_method', 'phi_gaussian')

	with io.open(os.path.join(config['parent_dir'], result_fname), 'a+') as fout:
		fout.write('\n********Seed: {} & Alpha: {} & kappa: {} & Task_Infer: {}********\n'.format(config['seed'], str(config['prior_alpha']), str(config['kappa']), task_infer_method))
		fout.write(result_str)

if __name__ == '__main__':
	main()