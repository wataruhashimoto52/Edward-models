# coding: utf-8

import numpy as np 
import edward as ed 
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from edward.models import Normal, MultivariateNormalTriL, Empirical

ed.set_seed(42)
T = 5000 # number of sampling
D = 3 # dimension
cov = [[ 1.36, 0.62, 0.93],
       [ 0.80, 1.19, 0.43],
       [ 0.57, 0.73, 1.06]]

# model
z = MultivariateNormalTriL(loc=tf.ones(D),
                scale_tril=tf.cholesky(cov))

# inference 
qz = MultivariateNormalTriL(loc=tf.Variable(tf.zeros(D)),
                            scale_tril=tf.nn.softplus(tf.Variable(tf.zeros((D, D)))))
inference = ed.KLqp({z:qz})

# qz = Empirical(tf.Variable(tf.random_normal([T,D])))
# inference = ed.HMC({z:qz}) 

inference.run()

# criticism
sess = ed.get_session()
mean, stddev = sess.run([qz.mean(), qz.stddev()])
print("Inferred posterior mean: ", mean)
print("Inferred post erior stddev: ", stddev)
a = sess.run(qz.sample(5000))

# plot
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(a[:, 0], a[:, 1], a[:, 2], "o")
plt.show()