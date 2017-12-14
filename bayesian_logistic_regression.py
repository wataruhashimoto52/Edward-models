# coding: utf-8

import numpy as np 
import tensorflow as tf 
import edward as ed
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt 
import seaborn as sns
from edward.models import Normal, Empirical, Bernoulli

def build_toy_dataset(N, noise_std=0.1):
    pass

ed.set_seed(42)

N = 40  # number of data points
D = 1  # number of features
