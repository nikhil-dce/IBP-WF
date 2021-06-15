import tensorflow as tf
import numpy as np
# from input_data import load_dataset, load_permuted_dataset, get_next_batch

SMALL = 1e-7

TF_KLB = "TF_KLB"
TF_KLV = "TF_KLV"
TF_KLR = "TF_KLR"
KUMAR_VARS = "TF_KUMAR"
GLOBAL_W_VARS = "TF_GLOBAL_W"
TF_W_NORM = "TF_W_NORM"

LAMBD = 2.0/3
# LAMBD_PRIOR = 2.0/3
# LAMBD_POST = 2.0/3
# LAMBD_POST = 10.
# LAMBD_PRIOR = 10.


def stop_gradients(target, mask):
    """Stop gradient where mask==0."""
    
    mask_h = tf.abs(mask-1)
    return tf.stop_gradient(mask_h * target) + mask * target

def get_tf_variable(name='W', shape=(10, 10), init=None, var_list=None, init_std=0.1):
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if init is None:
            init = tf.random_normal(shape, mean=0.0, stddev=init_std, dtype=tf.float32)
                    
        W = tf.get_variable(name, initializer=init, dtype=tf.float32)
    
    if (var_list is not None):
        var_list.append(W)
        return W, var_list
    else:
        return W

def get_tf_pos_variable(name='POS_VARIABLE', init_value=0.1, shape=(10,10)):
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        init = tf_logit(tf_inv_odds(tf.constant(init_value, shape=shape)))
        alpha_logit=tf.get_variable('logit', initializer=init) # initializing the variable as a logit
        out_tensor = tf_odds(tf.math.sigmoid(alpha_logit))     
    # convert logit to prob and then to positive real-number using tf_odds
    return out_tensor


# Load data
def get_task(task_no, continual=False):
    X, Y, X_test, Y_test = load_permuted_dataset(task_no, continual=continual)
    return X, Y, X_test, Y_test

def np_accuracy(logit, Y_gt):
    return np.mean(np.argmax(logit, axis=1) == np.argmax(Y_gt, axis=1))

def IBP( N,alpha ):
   #Generate initial number of dishes from a Poisson
    n_init = np.random.poisson(alpha,1)[0]
    Z = np.zeros(shape=(N,n_init))
    Z[0,:] = 1
    m = np.sum(Z,0)
    K = n_init

    for i in range(1,N):
        #Calculate probability of visiting past dishes
        prob = m/(i+1)
        index = np.greater(prob,np.random.rand(1,K))
        Z[i,:] = index.astype(int);
        #Calculate the number of new dishes visited by customer i
        knew = np.random.poisson(alpha/(i+1),1)[0]
        Z=np.concatenate((Z,np.zeros(shape=(N,knew))), axis=1)
        Z[i,K:K+knew:1] = 1
        #Update matrix size and dish popularity count
        K = K+knew
        m = sum(Z,0)
    
    return Z

def stick_breaking (N, alpha, kumaraswamy=False, truncation=100, probs=False):
    
    if kumaraswamy:
        v = np_sample_kumaraswamy(alpha, 1, size=(N, truncation))
    else:
        v = np.random.beta(alpha, 1, size=(N, truncation))
    
    print (v.shape)
    pi = np.cumprod(v, axis=1)
    
    if probs:
        return pi
    else:
        Z = np.greater(pi, np.random.rand(N, truncation))
        return Z.astype(int)

def tf_stick_breaking_weights(a, b, size=None):
    
    """
    Args: 
    a: Parameter
    b: parameter
    size: Shape of 2D tensor
    
    Returns: Log probabilities for the binary vector
    """
    if size is None:
        size = a.get_shape()
        
    v = tf_sample_kumaraswamy(a, b, size)
    v_term = tf.log(v + SMALL)
    log_prior = tf.cumsum(v_term, axis=0)
    
    return log_prior

def np_sample_kumaraswamy(a, b, size):
    """
    Numpy function to sample k ~ Kumaraswamy(a, b)
    Args:
    a: shape parameter 1
    b: shape parameter 2
    size: Return shape of np array
    """
    assert a>0 and b>0, "Parameters can not be zero"
    
    U = np.random.uniform(size=size)
    K = (1 - (1 - U)**(1.0/b))**(1.0/a)
    return K

def tf_sample_kumaraswamy(a, b, size=None):
    """
    TF function to sample k ~ Kumaraswamy(a, b)
    Args:
    a: shape parameter 1
    b: shape parameter 2
    size: Return shape of tf tensor
    """
    U = tf.random.uniform(minval=0.0001, maxval=0.9999, shape=size)
    K = (1 - (1 - U)**(1.0/b))**(1.0/a)
    return K

def tf_kullback_distance(a1, b1, a2, b2):
    """
    TF function to compute the Kullback Leibler Distance
    between Kumar(a1, b1) and Kumar(a2, b2)
    D = KL(Kumar(a1, b1) || Kumar(a2, b2))
    Important note: Kumar (a1, 1) and Beta (a2, 1) are the same.
    """
    
    pass

def Beta_fn(a, b):
    return tf.exp(tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a+b))

def kullback_normal_normal(mu_1, sigma_1, mu_2, sigma_2):
    """
    mu_1: mean of posterior
    sigma_1: std. deviation of posterior
    mu_2: mean of prior
    sigma_2: std. deviation of prior
    """
    kl = tf.log(sigma_2 / sigma_1)
    kl += (sigma_1**2 + ((mu_1 - mu_2)**2)) / (2*(sigma_2**2))
    kl += -0.5
    
    return kl
    
def kullback_kumar_beta(a1, b1, prior_alpha, prior_beta=1):
    
    """
    Credit: Nalisnick et al. "SBP DGMS" ICLR 2017
    
    TF function to approximate the Kullback Leibler Distance
    between Kumar(a1, b1) and Beta(prior_alpha, prior_beta)
    D = KL(Kumar(a1, b1) || Beta(prior_alpha, prior_beta))
    Important note: Kumar (a, 1) and Beta (a, 1) are the same.
    """  
    
    # compute taylor expansion for E[log (1-v)] term                                                                                                                                             
    # hard-code so we don't have to use Scan()                                                                                                                                                   
    kl = 1./(1+a1*b1) * Beta_fn(1./a1, b1)
    kl += 1./(2+a1*b1) * Beta_fn(2./a1, b1)
    kl += 1./(3+a1*b1) * Beta_fn(3./a1, b1)
    kl += 1./(4+a1*b1) * Beta_fn(4./a1, b1)
    kl += 1./(5+a1*b1) * Beta_fn(5./a1, b1)
    kl += 1./(6+a1*b1) * Beta_fn(6./a1, b1)
    kl += 1./(7+a1*b1) * Beta_fn(7./a1, b1)
    kl += 1./(8+a1*b1) * Beta_fn(8./a1, b1)
    kl += 1./(9+a1*b1) * Beta_fn(9./a1, b1)
    kl += 1./(10+a1*b1) * Beta_fn(10./a1, b1)
    kl *= (prior_beta-1)*b1

    # use another taylor approx for Digamma function                                                                                                                                             
    psi_b_taylor_approx = tf.log(b1) - 1./(2 * b1) - 1./(12 * b1**2)
    
    #     psi_b_taylor_approx = tf.math.digamma(b1)
    kl += (a1-prior_alpha)/a1 * (-0.57721 - psi_b_taylor_approx - 1/b1) # T.psi(self.posterior_b)                                                                                        

    # add normalization constants                                                                                                                                                                
    kl += tf.log(a1*b1) + tf.log(Beta_fn(prior_alpha, prior_beta))

    # final term                                                                                                                                                                                 
    kl += -(b1-1)/b1

    return kl

def kullback_kumar_kumar(a1, b1, prior_alpha, prior_beta):
    
    """   
    TF function to approximate the Kullback Leibler Distance
    between Kumar(a1, b1) and Kumar(prior_alpha, prior_beta)
    D = KL(Kumar(a1, b1) || Kumar(prior_alpha, prior_beta))
    """  
        
    # compute taylor expansion for E[log (1-v^{prior_alpha})] term                                                                                                                                             
    # hard-code so we don't have to use Scan()                                                                                                                                                   
    kl = 1./(1*prior_alpha+a1*b1) * Beta_fn(1.*prior_alpha/a1, b1)
    kl += 1./(2*prior_alpha+a1*b1) * Beta_fn(2.*prior_alpha/a1, b1)
    kl += 1./(3*prior_alpha+a1*b1) * Beta_fn(3.*prior_alpha/a1, b1)
    kl += 1./(4*prior_alpha+a1*b1) * Beta_fn(4.*prior_alpha/a1, b1)
    kl += 1./(5*prior_alpha+a1*b1) * Beta_fn(5.*prior_alpha/a1, b1)
    kl += 1./(6*prior_alpha+a1*b1) * Beta_fn(6.*prior_alpha/a1, b1)
    kl += 1./(7*prior_alpha+a1*b1) * Beta_fn(7.*prior_alpha/a1, b1)
    kl += 1./(8*prior_alpha+a1*b1) * Beta_fn(8.*prior_alpha/a1, b1)
    kl += 1./(9*prior_alpha+a1*b1) * Beta_fn(9.*prior_alpha/a1, b1)
    kl += 1./(10*prior_alpha+a1*b1) * Beta_fn(10.*prior_alpha/a1, b1)
    kl += 1./(11*prior_alpha+a1*b1) * Beta_fn(11.*prior_alpha/a1, b1)
    kl += 1./(12*prior_alpha+a1*b1) * Beta_fn(12.*prior_alpha/a1, b1)
    kl += 1./(13*prior_alpha+a1*b1) * Beta_fn(13.*prior_alpha/a1, b1)
    kl += 1./(14*prior_alpha+a1*b1) * Beta_fn(14.*prior_alpha/a1, b1)
    kl += 1./(15*prior_alpha+a1*b1) * Beta_fn(15.*prior_alpha/a1, b1)
    kl *= (prior_beta-1)*prior_alpha*b1
    
#     psi_b = tf.math.digamma(b1)
    # use another taylor approx for Digamma function                                                                                                                                             
    psi_b_taylor_approx = tf.log(b1) - 1./(2 * b1) - 1./(12 * b1**2)
    
    kl += tf.log(a1*b1) + (a1-prior_alpha)/a1 * (-0.57721 - psi_b_taylor_approx - 1/b1)
    kl += -(b1-1)/b1
    kl += -tf.log(prior_alpha*prior_beta)

    return kl

def tf_sample_logistic_Y(p, lambd):
    """
    Y = (log (p / (1-p)) + L) / lambd
    
    Args: 
    p: Bernoulli parameter used to construct \alpha = p / (1-p) for BinConcrete(\alpha, temperature)
    lambd: Temp parameter for BinConcrete
    """
    
    # assert (lambd > 0 and lambd <= 1), "Temperature not in (0,1]"
    
    alpha = p / (1 - p)
    U = tf.random.uniform(minval=0.0001, maxval=0.9999, shape=tf.shape(p))
    L = tf.log(U) - tf.log(1-U)
    
    Y = (tf.log(alpha) + L)/lambd
    return Y
    
def tf_sample_BernConcrete(p, lambd):
    """
    TF function to sample from Concrete equivalent of 
    Bernoulli(p).
    
    Args:
    p: bernoulli parameter \in (0, 1)
    
    Returns:
    Samples with shape same as p
    """
    
    Y = tf_sample_logistic_Y(p, lambd)
    return Y, tf.math.sigmoid(Y)

def tf_log_density_logistic(p, lambd, y_sample):
    """
    MC estimate of log(p_{lambd}(Y)) where Y =  (log (p / (1-p)) + L) / lambd
    
    Args:
    p: Bern parameter
    lambd: Concrete temperature parameter
    y_sample: For MC approximation
    """
    
    p = tf.clip_by_value(p, clip_value_min=0.001, clip_value_max=0.999)
    lAlpha = tf.log(p / (1-p))
    
    log_density = lAlpha - lambd*y_sample + tf.log(lambd) - 2 * tf.math.softplus(lAlpha-lambd*y_sample)
    
    return log_density

def tf_kl_logistic (y_sample, p_post, lambd_post, p_prior, lambd_prior):
    """
    MC estimate for KL (q(Y) || p(Y))
    where Y = (log(alpha_post) + L) / lambd_post => Y ~ q(Y)
    and Y = (log(alpha_prior) + L) / lambd_prior => Y ~ p(Y)
    
    Args:
    y_sample: Y ~ q(Y)
    p_post: for \alpha_post
    p_prior: for \alpha_prior
    lambd_post: temperature posterior
    lambd_prior: temperature prior
    
    Returns:
    KL estimate 
    """
    
    log_q = tf_log_density_logistic(p_post, lambd_post, y_sample)
    log_p = tf_log_density_logistic(p_prior, lambd_prior, y_sample)
    
    return log_q - log_p

def tf_odds(p):
    """
    Args: p \in (0.001, 0.999)
    returns logit for p \in (0,1)
    """
    p = tf.clip_by_value(p, clip_value_min=0.001, clip_value_max=0.999)
    return p/(1-p)

def tf_inv_odds(odds):
    """
    inverse of tf_odd function
    """
    return (odds / (odds+1))

def tf_logit(p):
    """
    return log(p / (1-p))
    """
    p = tf.clip_by_value(p, clip_value_min=0.001, clip_value_max=0.999)
    return tf.log(p / (1-p))

def np_logit(p):
    """
    return log(p / (1-p))
    """
    p = tf.clip_by_value(p, clip_value_min=0.001, clip_value_max=0.999)
    return tf.log(p / (1-p))

def get_lambda(initial_lambd, final_lambd, step_count, decay_gamma, fine_tuning=False):
    
    if fine_tuning:
        return final_lambd
        
    lambd = initial_lambd*(decay_gamma**step_count)
    lambd = max(lambd, final_lambd)
    return lambd