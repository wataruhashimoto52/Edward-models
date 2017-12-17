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
data = pd.read_csv("../data/data-salary-2.txt")

X = tf.placeholder(tf.float32, [None])
KID = tf.placeholder(tf.float32, [None])
a = Uniform(low=-1000, high=1000)
b = Uniform(low=-1000, high=1000)
sd = Uniform(low=0, high=1000)
y = Normal(loc=a*KID+b*KID*X, scale=sd)

inference = ed.KLqp({a:qa, b:qb, }, data={X:data['X'], KID:data['KID'], y:data['Y']})
inference.run()

