import tensorflow as tf

import sklearn.covariance
from sklearn.utils.extmath import fast_logdet as logdet
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import ShrunkCovariance

import pickle as pkl

class MyEstimator(EmpiricalCovariance):
	 
	def __init__(self, store_precision=True, assume_centered=False):
		self.logdet_prec = None
		EmpiricalCovariance.__init__(self, store_precision=True, assume_centered=assume_centered)
						
	def my_score(self, h):
		
		precision = self.precision_
		mu = self.location_
		
		if self.logdet_prec is None:
			self.logdet_prec = logdet(precision + np.eye((precision.shape[0]))*0.001 ) 
		
		print (self.logdet_prec)
		
		#SMALL = 1e-5
		return 0.5*self.logdet_prec - 0.5*np.sum(np.matmul(h - mu, precision) * (h-mu), axis=1)    
#         return 0.5*np.log(linalg.det(sigma_inv)+SMALL) - 0.5*np.sum(np.matmul(h - mu, sigma_inv) * (h-mu), axis=1)
#         return 0.5* - 0.5*np.sum(np.matmul(h - mu, precision) * (h-mu), axis=1)

	def fit(self, X):
		self.n = X.shape[0]
		EmpiricalCovariance.fit(self, X)

# class MyEstimator(ShrunkCovariance):
 
# 	def __init__(self, store_precision=True, assume_centered=False):
# 		self.logdet_prec = None
# 		ShrunkCovariance.__init__(self, store_precision=True, assume_centered=assume_centered, shrinkage=0.005)
						
# 	def my_score(self, h):
		
# 		precision = self.precision_
# 		mu = self.location_
		
# 		if self.logdet_prec is None:
# 			self.logdet_prec = logdet(precision) 
# 			print (self.logdet_prec)
		
# 		#SMALL = 1e-5
# 		return 0.5*self.logdet_prec - 0.5*np.sum(np.matmul(h - mu, precision) * (h-mu), axis=1)    
# #         return 0.5*np.log(linalg.det(sigma_inv)+SMALL) - 0.5*np.sum(np.matmul(h - mu, sigma_inv) * (h-mu), axis=1)
# #         return 0.5* - 0.5*np.sum(np.matmul(h - mu, precision) * (h-mu), axis=1)

# 	def fit(self, X):
# 		self.n = X.shape[0]
# 		ShrunkCovariance.fit(self, X)

def save_estimators(config, estimators):

	est_data = {}

	for ix, est in enumerate(estimators):
		est_data['est_t{}'.format(ix)] = est

	est_data['size'] = len(estimators)
	
	fname = config['result_dir']
	fname += ("est.dictionary_{:d}".format(config['seed']))

	with open(fname, 'wb') as est_file:
		pkl.dump(est_data, est_file)

def load_estimators(config):

	est_list = []

	fname = config['result_dir']
	fname += ("est.dictionary_{:d}".format(config['seed']))

	with open(fname, 'rb') as est_file:
		est_data = pkl.load(est_file)

		size = est_data.get('size')
	
		for i in range(size):
			est_list.append(est_data['est_t{}'.format(i)])


	return est_list