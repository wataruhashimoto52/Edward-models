# coding: utf-8

import numpy as np
import pandas as pd 
import scipy as sp 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(context='paper', style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.2)
import tensorflow as tf 
import edward as ed
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt 
import seaborn as sns
from edward.models import Normal, Empirical, Uniform, Poisson, PointMass,\
                        Gamma, Exponential


np.random.seed(39)
def build_toy_dataset(N, w, b, noise_std=2.0):
    D = len(w)
    x = np.sort(np.random.randn(N))
    xx = np.array([[r ** i for i in range(1, D+1)] for r in x])
    y = np.dot(xx, w) + b + np.random.normal(0, noise_std, size=N)
    return x, y

N = 30 # データ数
D = 3 # データの次元数
MD=3 # モデルの次元

# 真のパラメータの設定
w_true = 5*np.random.randn(D)
b_true = 0.5
print("True parameters")
print("w={}".format(w_true))
print("b={}".format(b_true))
X_train, y_train = build_toy_dataset(N, w_true, b_true)
XX_train = np.array([X_train**i for i in range(1, MD+1)])
XX_train = XX_train.T

# それぞれのsigmaは既知であるとする
# model
X = tf.placeholder(tf.float32, [N, MD])
w = Normal(loc=tf.zeros(MD), scale=tf.ones(MD))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X,w)+b, scale=tf.ones(N)*2.0)


# inference
qw = PointMass(params=tf.Variable(tf.zeros([MD])))
qb = PointMass(params=tf.Variable(tf.zeros(1)))

inference = ed.MAP({w:qw, b:qb}, {X:XX_train, y:y_train})
inference.run(n_iter=10000)

# criticism
def visualize(X_data, y_data, w, b, n_samples=10):
    w_samples = w.sample(n_samples)[:].eval()
    D = len(w_samples[0]) #次元数
    b_samples = b.sample(n_samples).eval()
    plt.scatter(X_data, y_data)
    
    inputs = np.linspace(-8, 8, num=1000)
    inputs_list = []
    outputs = []
    # サンプリング
    for ns in range(n_samples):
        # 範囲内のxに対するyの値を計算
        output = np.random.normal(sum([inputs**i * w_samples[ns][i-1] for i in range(1, D+1)]) + b_samples[ns], 2.0)
        # 得られた結果をリストに追加
        for x in inputs:
            inputs_list.append(x)
        for y in output:
            outputs.append(y)
            
    plt.hist2d(inputs_list, outputs, bins=[100, 100], range=np.array([(-8, 8), (-80, 80)]), normed=True, cmap=cm.Reds)
    plt.xlim([-8.0, 8.0])
    plt.ylim([-80.0, 80.0])

visualize(X_train, y_train, qw, qb, 10000)