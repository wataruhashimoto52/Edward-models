# coding: utf-8

import numpy as np 
import pandas as pd 
import scipy as sp 
import tensorflow as tf 
import edward as ed
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt 
import seaborn as sns
from edward.models import Normal, Empirical, PointMass, Exponential, Poisson

ed.set_seed(42)

# preparation of data
def build_toy_dataset(N=50, noise_std=0.1):
    x = np.linspace(-3, 3, num=N)
    y = np.cos(x) + np.random.normal(0, noise_std, size=N)
    x = x.astype(np.float32).reshape((N, 1))
    y = y.astype(np.float32)
    return x, y

def neural_network(x, W_0, W_1, b_0, b_1):
    h = tf.matmul(x, W_0) + b_0
    h = tf.tanh(h)
    h = tf.matmul(h, W_1) + b_1
    return tf.reshape(h, [-1])


# preprocess

# model

# inference

# criticism