# coding: utf-8

import numpy as np 
import pandas as pd 
import tensorflow as tf 
import edward as ed
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt 
import seaborn as sns
from edward.models import Normal, Empirical, Uniform

ed.set_seed(42)
data = pd.read_csv("data/data-salary-2.txt")
# T=5000
N = len(data)

# model
X = tf.placeholder(tf.float32, [N])
KID = Uniform(low=0.0, high=4.0, sample_shape=N)
a = Uniform(low=-1000.0, high=1000.0, sample_shape=N)
b = Uniform(low=-1000.0, high=1000.0, sample_shape=N)
sd = Uniform(low=0.0, high=1000.0, sample_shape=N)
Y = Normal(loc=a*KID+b*KID*X, scale=sd, sample_shape=N)

# inference
qa = Empirical(params=tf.Variable(tf.random_normal([N])))
qb = Empirical(params=tf.Variable(tf.random_normal([N])))
qsd = Empirical(params=tf.Variable(tf.random_normal([N])))

# preprocess
x = np.array(data['X']).reshape((N,1))
kid = np.array(data['KID']).reshape((N, 1))
y = np.array(data['Y']).reshape((N, 1))

# variational inference
inference = ed.KLqp({a:qa, b:qb, sd:qsd}, data={X:x, KID:kid, Y:y})
inference = tf_debug.LocalCLIDebugWrapperSession(inference)
inference.run()

