import sys
import yaml
import os
from argparse import ArgumentParser
import pprint
import time
import io
import pickle as pkl

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import dataloader
from models import MODEL

from utilities import inference_utils
from utilities.inference_utils import *
from utilities.estimator_utils import *

parser = ArgumentParser()
parser.add_argument('--config', '-c', default='configs/permutedMNIST10_ibpWF.yaml')
parser.add_argument('--load_ckpt', default=0)
parser.add_argument('--seed', default=-1, type=int)
parser.add_argument('--kappa', default=-1.0, type=float)
parser.add_argument('--save_phi', default=False, type=bool)
parser.add_argument('--alpha', default=0., type=float)

def train_on_task(sess, model, task_id, config, data_train, data_test, fine_tuning=False, **kwargs):
	
	X, Y = data_train
	X_te, Y_te = data_test

	# Training loop same for all tasks
	batch_size = config['batch_size']
	num_epochs = config['epochs']
	is_deterministic = config['is_deterministic']

	num_batches = X.shape[0] // batch_size
	if num_batches*batch_size < X.shape[0]:
		num_batches += 1

	if task_id > 0 or fine_tuning:
		old_global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

	reuse = tf.AUTO_REUSE if task_id > 0 or fine_tuning else False
	
	# Update graph for the task
	l2_loss = tf.constant(0.)
	l2_global_W = tf.constant(0.)
	klb = tf.constant(0.)
	klr = tf.constant(0.)
	klv = tf.constant(0.)
	optim_name = ''

	if fine_tuning:
		print ('Finetuning on task {}'.format(task_id+1))
		kwargs['fine_tuning'] = True
		optim_name = '_fine_tune'
	else:
		print ('\nTrain model on task {}'.format(task_id+1))

	NLL_loss, logits = model.forward(task_id, reuse, is_training=True, **kwargs)
	if not fine_tuning and not is_deterministic:
		# Get KL Distance
		klb = tf.math.add_n(tf.get_collection(TF_KLB))
		klv = tf.math.add_n(tf.get_collection(TF_KLV))
	
	if config['weight_decay'] > 0:
		print ('*******weight_decay*********')
		if (not fine_tuning) or (config.get('fine_tuning_l2', False)):
			
			l2_loss = tf.math.add_n(tf.get_collection('L2_LOSS'))

	# else:
	# 	klr = tf.math.add_n(tf.get_collection(TF_KLR))
	elif not (is_deterministic or fine_tuning):
		klr = tf.math.add_n(tf.get_collection(TF_KLR))
		
	kl_loss = (klb + klr + klv) / X.shape[0]
	loss = tf.reduce_mean(NLL_loss)  + kl_loss

	if config.get('MAP_W', False) and task_id > 0:
		w_norm = tf.math.add_n(tf.get_collection(TF_W_NORM))
		l2_global_W = w_norm
		loss += l2_global_W
		
	global_step = tf.get_variable('global_step_{}{}'.format(task_id, optim_name), trainable= False, initializer=0)
	if config['optimizer'] == 'SGD':

		if fine_tuning:
			lr = config.get('fine_tuning_lr', config['lr_values'][-1])
			num_epochs = config.get('fine_tuning_epochs', 5)
			solver = tf.train.MomentumOptimizer(learning_rate=lr, momentum=config['optim_momentum'], name='SGD_task_{}{}'.format(task_id, optim_name))
		else:	
			boundaries = [epoch*num_batches for epoch in config['epoch_boundaries']]
			lr_values = config['lr_values']
			lr = tf.train.piecewise_constant(global_step, boundaries, lr_values)
			solver = tf.train.MomentumOptimizer(learning_rate=lr, momentum=config['optim_momentum'], name='SGD_task_{}{}'.format(task_id, optim_name))
	else:
		
		if (not fine_tuning) and 'lr_values' in config and 'epoch_boundaries' in config:
			print ('Applying lr decay.')
			boundaries = [epoch*num_batches for epoch in config['epoch_boundaries']]
			lr_values = config['lr_values']
			lr = tf.train.piecewise_constant(global_step, boundaries, lr_values)
		else:
			# lr = 0.0001 if fine_tuning else config['lr']
			lr = config['lr']
			if fine_tuning:
				lr = config.get('fine_tuning_lr', lr)
				num_epochs = config.get('fine_tuning_epochs', 5)

		solver = tf.train.AdamOptimizer(learning_rate=lr, name='Adam_task{}{}'.format(task_id, optim_name))
	
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):

		gvs = solver.compute_gradients(loss+config['weight_decay']*l2_loss)
		capped_gvs = []
		for grad, var in gvs:
			if grad is not None:
				capped_gvs.append((grad, var))

		# TODO: Check if need to clip this! Not needed for split mnist
		if config.get('clip_gradients', False):
			capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in capped_gvs]
		
		step = solver.apply_gradients(capped_gvs, global_step)

	if sess is None:
		# when task_id = 0
		configProto = tf.ConfigProto()
		configProto.gpu_options.allow_growth=True
		sess = tf.Session(config=configProto)
		sess.run(tf.global_variables_initializer())
	else:
		# when task_id > 0
		init_start = time.time()
		new_global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		delta_global_vars = []

		for var in new_global_vars:
			if (var in old_global_vars):
				continue
			else:
				delta_global_vars.append(var)

		# print ('Delta Global Vars...')
		# print (delta_global_vars)
		init = tf.initializers.variables(delta_global_vars)
		sess.run(init)
		
	initial_lambd = config['final_lambd'] if fine_tuning else config['initial_lambd']
	decay_gamma = config['lambd_decay_gamma']
	final_lambd = config['final_lambd']

	assert decay_gamma < 1. and decay_gamma > 0.

	for epoch in range(1, num_epochs+1):
		
		# local_seed = 1234 # training data seeed remains the same but global tf,np seeds change
		# X, Y = shuffle(X, Y, random_state=local_seed+epoch)

		start = time.time()
		epoch_loss = []
		epoch_logits = []
		epoch_klb = []
		epoch_klv = []
		epoch_klr = []

		for batch_idx in range(num_batches):

			start_idx = batch_idx*batch_size
			end_idx = min(start_idx+batch_size, X.shape[0])

			X_batch = X[start_idx:end_idx]
			Y_batch = Y[start_idx:end_idx]

			step_count = (epoch-1)*num_batches + batch_idx
			lambd = inference_utils.get_lambda(initial_lambd, final_lambd, step_count, decay_gamma, fine_tuning)
			
			feed_dict = {
										model.X_ph: X_batch, 
										model.Y_ph: Y_batch, 
										model.lambd: lambd,
									}

			np_loss, np_klb, np_klv, np_klr, np_logits, _, np_l2, np_l2_global_W = sess.run([loss, klb, klv, klr, logits, step, l2_loss, l2_global_W], feed_dict=feed_dict)

			epoch_loss.append(np_loss)
			epoch_klb.append(np_klb)
			epoch_klv.append(np_klv)
			epoch_klr.append(np_klr)
			epoch_logits.extend(np_logits)
			
		epoch_acc = np_accuracy(epoch_logits, Y)
		
		stop = time.time()
		delt = stop-start
		
		if epoch % 5 == 0:
			
			if (config['use_validation']):
				print ('Validation...')
				X_va, Y_va = kwargs['data_va']
				feed_val = {model.X_ph: X_va, model.lambd: final_lambd}
				logits_val = sess.run(logits, feed_dict=feed_val)
				val_acc = np_accuracy(logits_val, Y_va)
				print ('Task {0:d} \tValidation Accuracy {1:.4}'.format(task_id+1, val_acc))

			print ("Task {0:d} | Epoch {1:d} | Avg Loss: {2:.4f} | KLD V: {3:.2f} | KLD B: {4:.2f} | KLD R: {5:.2f} | L2: {8:.2f} | Avg Accuracy: {6:.4f} | time/epoch: {7:.2f}"
					.format(task_id+1, epoch, np.mean(epoch_loss),
							np.mean(epoch_klv), np.mean(epoch_klb), 
							np.mean(epoch_klr), epoch_acc, delt, np_l2))

	print ('Testing...')
	# if 'fine_tuning' in kwargs:
	# 	kwargs.pop('fine_tuning')
	_, logits = model.forward(task_id, tf.AUTO_REUSE, is_training=False, **kwargs)
	feed_test = {model.X_ph: X_te, model.lambd: lambd}
	logits_test = sess.run(logits, feed_dict=feed_test)
	test_acc = np_accuracy(logits_test, Y_te)
	print ('Task {0:d} \tAccuracy {1:.4}'.format(task_id+1, test_acc))

	return sess

def get_MLE_estimator(phi):
	estimator = MyEstimator(assume_centered=False)
	estimator.fit(phi)
	return estimator

def return_estimators(config, model, sess, task_id, X, save_task = 0):
	
	print ('\nGetting task estimator for task {}'.format(task_id))

	# set expectation to be true since we want deterministic features
	H = MODEL[config['model_name']].get_phi(model, sess, task_id, X)
	print (H.shape)
	
	if config['save_phi']:
		np.savez(config['result_dir']+config['dataset']+'_phi_'+str(save_task), H=H)

	estimator = get_MLE_estimator(H)
	return estimator

def get_kumar_vars(sess, task_id):

	print ('\nTask: {} - Getting Kumaraswamy parameters'.format(task_id+1))
	kumar_vars = tf.get_collection(KUMAR_VARS)

	t_names = []
	t_values = []
	for dic in kumar_vars:
		for d in dic:
			t_names.append(d)
			t_values.append(dic[d])

	np_cd = sess.run(t_values)

	prior_cd = {}
	for ix, name in enumerate(t_names):
		prior_cd[name] = np_cd[ix]
		
	return prior_cd

def get_global_W(sess, task_id):
	
	print ('\nGetting Global parameters'.format(task_id+1))
	global_W = tf.get_collection(GLOBAL_W_VARS)

	t_names = []
	t_values = []
	for dic in global_W:
		for d in dic:
			t_names.append(d)
			t_values.append(dic[d])

	np_Ws = sess.run(t_values)
	
	prior_Ws = {}
	for ix, name in enumerate(t_names):
		prior_Ws[name] = np_Ws[ix]

	return prior_Ws

def train(dataset, get_task, config):
	
	num_tasks = dataset[4]
	net_out = dataset[-2]
	config['num_classes'] = net_out
	
	get_valid = config['use_validation']

	# create model
	model = MODEL[config['model_name']](config)
	sess = None
	kwargs = dict()
	est_list = []

	for task_id in range(num_tasks):
		
		if get_valid:
			Xtr, Ytr, Xva, Yva, Xte, Yte = get_task(task_id, dataset, get_valid=get_valid)
			data_va = (Xva, Yva)
			kwargs['data_va'] = data_va
		else:
			Xtr, Ytr, Xte, Yte = get_task(task_id, dataset, get_valid=get_valid)

		data_tr = (Xtr, Ytr)
		data_te = (Xte, Yte)

		sess = train_on_task(sess, model, task_id, config, data_tr, data_te, fine_tuning=False, **kwargs)
					
		if config['fine_tuning']:
			sess = train_on_task(sess, model, task_id, config, data_tr, data_te, fine_tuning=True, **kwargs)
		
		if config.get('save_params', False):
			save_parameters(config, sess, task_id)

		est_t = return_estimators(config, model, sess, 0, Xtr, save_task=task_id)	
		kwargs = get_kumar_vars(sess, task_id)
		kwargs.update(get_global_W(sess, task_id))

		est_list.append(est_t)
		
	save_estimators(config, est_list)

	print ('See config here:' + str(config['result_dir']))
	save_path = os.path.join(config['result_dir'], "continual_model_{:d}.ckpt".format(config['seed']))
	saver = tf.train.Saver()
	save_path = saver.save(sess, save_path)
	print("Model saved in path: %s" % save_path)

def save_parameters(config, sess, task_id):

	all_vars = {}
	
	var_names = []    
	for var in tf.trainable_variables():    
		var_names.append(var.name)

	print (var_names)
	values = sess.run(var_names)

	for name, val in zip(var_names, values):
		all_vars[name] = val
	
	fname = config['result_dir']
	fname += ("weights_{:d}".format(task_id))

	with open(fname, 'wb') as outfile:
		pkl.dump(all_vars, outfile)

	# sys.exit()

def main():
	args = parser.parse_args()

	# Load config
	config_path = args.config
	config = yaml.load(open(config_path), Loader=yaml.FullLoader)
	os.environ['CUDA_VISIBLE_DEVICES'] = config['gpuid']

	if int(args.seed) > 0:
		config['seed'] = args.seed
	if args.kappa >= 0.0 and args.kappa <= 1.0:
		config['kappa'] = args.kappa

	if args.alpha > 0:
		
		if (config['dataset'] == 'splitCIFAR'):
			raise NotImplementedError('splitCIFAR implementation does not support alpha cli input. Set alpha in config.')

		config['prior_alpha'] = args.alpha

	config['save_phi'] = args.save_phi
	alpha_str = '/'

	if 'splitCIFAR' not in config['dataset']:
		alpha_str = '_alpha{}/'.format(str(config['prior_alpha']))

	config['parent_dir'] = config['result_dir']
	if config['is_deterministic']:
		config['result_dir'] = os.path.join(config['result_dir'], 'deterministic', 'seed{}/'.format(str(config['seed'])))
	else:	
		config['result_dir'] = os.path.join(config['result_dir'],'seed{}_kappa{}'.format(str(config['seed']), str(config['kappa']))+alpha_str)

	if not os.path.exists(config['result_dir']):
		os.makedirs(config['result_dir'])

	print ('\n')
	pprint.pprint (config)
	print ('\n')

	# Write Config YAML file
	with io.open(config['result_dir']+'config.yaml', 'w', encoding='utf8') as outf:
		yaml.dump(config, outf, default_flow_style=False, allow_unicode=True)

	tf.set_random_seed(config['seed'])
	np.random.seed(config['seed'])

	if config['dataset'] == 'splitMNIST':
		dataset, get_task = dataloader.load_splitMNIST(config['dataset_path'])
		train(dataset, get_task, config)
	
	elif config['dataset'] == 'splitCIFAR10':
		dataset, get_task = dataloader.load_splitCIFAR10(config['dataset_path'])
		train(dataset, get_task, config)
	
	elif config['dataset'] == 'permutedMNIST':
		dataset, get_task = dataloader.load_permutedMNIST(config['dataset_path'], config['num_tasks'])
		train(dataset, get_task, config)
	else:
		dname = config["dataset"]
		raise ValueError('Dataset name not valid. Found: {}'.format(dname))

if __name__ == '__main__':
	main()