# coding: utf-8

"""
Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
chapter1: changepoint detection

ref: https://discourse.edwardlib.org/t/bayesian-hacker-chapter1-introduction-imp-use-edward/299  
"""
import numpy as np 
import tensorflow as tf 
import edward as ed
from tensorflow.python import debug as tf_debug
from edward.models import Normal, Poisson, PointMass, Exponential, Uniform, Empirical 

count_data = np.loadtxt("data/txtdata.csv")
n_count_data = len(count_data)


sess = tf.Session()

alpha_f = 1.0/count_data.mean()

with tf.name_scope('model'):
    alpha = tf.Variable(alpha_f, name="alpha", dtype=tf.float32)

    # init 
    lambda_1 = Exponential(alpha, name="lambda1")
    lambda_2 = Exponential(alpha, name="lambda2")
    tau = Uniform(low=0.0,high=float(n_count_data - 1),name="tau")
    idx = np.arange(n_count_data)
    lambda_ = tf.where(tau>=idx,tf.ones(shape=[n_count_data,],dtype=tf.float32)*lambda_1,
            tf.ones(shape=[n_count_data,],dtype=tf.float32)*lambda_2)

    # error
    z = Poisson(lambda_,value=tf.Variable(tf.ones(n_count_data)), name="poi")


# model
T = 5000  # number of posterior samples
with tf.name_scope("posterior"):

    qlambda_1 =  Empirical(params=tf.Variable(tf.zeros([T])), name="qlambda1")
    qlambda_2 =  Empirical(params=tf.Variable(tf.zeros([T])),name="qlambda2")
    qtau = Empirical(params=tf.Variable(tf.zeros([T])),name="qtau")
"""
qlambda_1 =  Empirical(params=tf.Variable(tf.zeros([n_count_data])))
qlambda_2 =  Empirical(params=tf.Variable(tf.zeros([n_count_data])))
"""
# qlambda_  =  Empirical(params=tf.Variable(tf.zeros([T,n_count_data,1])))
# qz = Empirical(params=tf.Variable(tf.random_normal([n_count_data,1])))

inference = ed.HMC({lambda_1:qlambda_1,lambda_2:qlambda_2,tau:qtau},data={z:count_data})
inference.run(logdir='log/cp_detection')