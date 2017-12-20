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
T = 3000
N = len(data)

# model
X = tf.placeholder(tf.float32, [N])
KID = tf.placeholder(tf.float32, [N])
a = Uniform(low=-1000.0, high=1000.0)
b = Uniform(low=-1000.0, high=1000.0)
sd = Uniform(low=0.0, high=1000.0)
Y = Normal(loc=a*KID+b*KID*X, scale=sd)

# inference
qa = Empirical(params=tf.Variable(tf.random_normal([T])))
qb = Empirical(params=tf.Variable(tf.random_normal([T])))
qsd = Empirical(params=tf.Variable(tf.random_normal([T])))

# preprocess
x = np.array(data['X'])
kid = np.array(data['KID'])
y = np.array(data['Y'])
# Hamiltonian Monte Carlo
inference = ed.HMC({a:qa, b:qb, sd:qsd}, data={X:x, KID:kid, Y:y})
inference.run()