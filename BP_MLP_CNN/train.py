from mlp import Model
from mlp import BasicRoutinizer
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import rc


test = np.random.random([9,20])
# test  =[-3,-5,-5,-6,-1,-64,-12,-123,-3]
# test = np.array(test).T

"""Create toy data"""
import math

# sin
def sin(x):
    y = np.sin(2 * math.pi * x)
    return y

def create_toy_data(func, interval, sample_num, noise = 0.0, add_outlier = False, outlier_ratio = 0.001):
    """
    generate data with the given function
    
    input:
       - func: the input function
       - interval: the range of values of x, a tuple (start, end)
       - sample_num: number of samples
       - noise: the standard deviation of Gaussian noise
       - add_outlier: whether to generate outliers
       - outlier_ratio: proportion of outliers
       
    output:
       - X: samples, shape = [n_samples,1]
       - y: labels, shape = [n_samples,1]
    """
    
    X = np.random.rand(sample_num,1) * (interval[1]-interval[0]) + interval[0]
    y = func(X)

    # add Gaussian noise
    epsilon = np.random.normal(0, noise, (sample_num,1))
    y = y + epsilon
    
    # add outlier
    if add_outlier:
        outlier_num = int(sample_num * outlier_ratio)
        if outlier_num != 0:
            outlier_idx = np.random.randint(sample_num, size = [outlier_num,1])
            y[outlier_idx] = y[outlier_idx] * 5
            
    return X, y

func = sin
interval = (0,1)
train_num = 64
test_num = 100
noise = 0
X_train, y_train = create_toy_data(func=func, interval=interval, sample_num=train_num, noise=noise)
X_test, y_test = create_toy_data(func=func, interval=interval, sample_num=test_num, noise=noise)

X_underlying = np.linspace(interval[0],interval[1],num=100)
y_underlying = sin(X_underlying)


# One good setting for sin
# max_iter = 250000
# lr = 5e-6
# model = 1,100,100,1

max_iter = 10000

m = Model(layer_size=[1,100,100,1],debug=False)
m.initialize()
run = BasicRoutinizer(max_iter=max_iter,feature_data=X_train.T,label_data=y_train.T,lr=3e-4,dynamic_show_res=50,dynamic_show=True)
run.run(m=m)
# 

predict = m.forward(X_test.T)


# plot
"""_summary_ draw the plot of the underlying function and the training and test data"""
rc('font', family='times new roman')
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.plot(X_underlying, y_underlying, c='#000000', label=r"$\sin(2\pi x)$")
plt.scatter(X_train, y_train, facecolor="none", edgecolor='#e4007f', s=50, label="train data")
plt.scatter(X_train, run.final_ans[-1], facecolor="none", edgecolor="#61f43f", marker = '^', s=50, label="mlp data")
plt.legend(fontsize='x-large')
plt.show()

# At last we look at test set

# plot
"""_summary_ draw the plot of the underlying function and the training and test data"""
rc('font', family='times new roman')
plt.rcParams['figure.figsize'] = (8.0, 6.0)
plt.plot(X_underlying, y_underlying, c='#000000', label=r"$\sin(2\pi x)$")
plt.scatter(X_test, y_test, facecolor="none", edgecolor="r", marker = '^', s=50, label="test data")
plt.scatter(X_test, predict, facecolor="none", edgecolor="#61f43f", marker = '^', s=50, label="mlp data")
plt.legend(fontsize='x-large')
plt.show()