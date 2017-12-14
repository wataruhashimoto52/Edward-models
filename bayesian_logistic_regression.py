# coding: utf-8

import tensorflow as tf 
import edward as ed
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt 
import seaborn as sns
from edward.models import Normal, Empirical 

def build_toy_dataset(N, noise_std=0.5):
    X = np.concatenate([np.linspace(0, 2, num=N/2),
                        np.linspace(6, 8, num=N/2)])
    Y = 2.0*X + 10 * np.random.normal(0, noise_std, size=N)
    X = X.reshape((N, 1))
    return X, Y

ed.set_seed(42)

N = 40 # number of data points
D = 1 # number of features 

# data
X_train, y_train = build_toy_dataset(N)
X_test, y_test = build_toy_dataset(N)

# model
X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X, w), scale=tf.ones(N))

# inference
T=5000              # number of samples
nburn = 100         # number of burn-in samples
stride = 10         # frequency with which to plot samples
qw = Empirical(params=tf.Variable(tf.random_normal([T, D])))
qb = Empirical(params=tf.Variable(tf.random_normal([T, 1])))

inference = ed.SGHMC({w:qw, b:qb}, data={X:X_train, y: y_train})
inference.run(step_size=1e-3)

# criticism

# plot posterior samples
sns.jointplot(qb.params.eval()[nburn:T:stride],
              qw.params.eval()[nburn:T:stride])
plt.show()

