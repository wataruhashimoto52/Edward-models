# coding: utf-8

import numpy as np 
import tensorflow as tf 
import edward as ed
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt 
import seaborn as sns
from edward.models import Normal, Empirical, Bernoulli

def build_toy_dataset(N, noise_std=0.1):
    D = 1
    X = np.linspace(-6, 6, N)
    y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
    y[y<0.5] = 0
    y[y>=0.5] = 1
    X = (X-4.0) / 4.0
    X = X.reshape((N, D))
    return X, y

ed.set_seed(42)

N = 40  # number of data points
D = 1  # number of features

X_train, y_train = build_toy_dataset(N)

X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=1.0*tf.ones(D))
b = Normal(loc=tf.zeros([]), scale=1.0*tf.ones([])) 
y = Bernoulli(logits=ed.dot(X, w)+b)


# inference
T=5000
qw = Empirical(params=tf.Variable(tf.random_normal([T, D])))
qb = Empirical(params=tf.Variable(tf.random_normal([T])))

inference = ed.HMC({w:qw, b:qb}, data={X:X_train, y:y_train})
inference.initialize(n_print=10, step_size=0.6)

tf.global_variables_initializer().run()

# criticism & set up figure
fig = plt.figure(figsize=(8,8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

n_samples = 50
inputs = np.linspace(-5, 3, num=400, dtype=np.float32).reshape((400, 1))
probs = tf.stack([tf.sigmoid(ed.dot(inputs, qw.sample()) + qb.sample())
                for _ in range(n_samples)])

for t in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)

    if t % inference.n_print == 0:
        outputs = probs.eval()

        # plot data and functions
        plt.cla()
        ax.plot(X_train[:], y_train, 'bx')
        for s in range(n_samples):
            ax.plot(inputs[:], outputs[s], alpha=0.2)
        ax.set_xlim([-5, 3])
        ax.set_ylim([-0.5, 1.5])
        plt.draw()
        plt.pause(1.0/60.0)

plt.show()

