import numpy as np
import pickle
import os
from activation_function import sigmoid
from output_layer import softmax
from mnist import load_mnist

def get_data():
    (x_train,t_train),(x_test,t_test) = \
    load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test


def init_network():
    # 1. 先动态拿到当前 neuralnet_mnist.py 所在的绝对路径（即 ch-03 文件夹的路径）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 把 ch-03 的路径和文件名拼接成一个绝对路径
    weight_path = os.path.join(current_dir, "sample_weight.pkl")
    
    # 3. 用绝对路径打开它，不管工作目录在哪，都能 100% 精准狙击
    with open(weight_path, 'rb') as f:
        network = pickle.load(f)
        
    return network

def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y

# 评价 识别精度（accuracy
x,t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y) # 获得概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

   
