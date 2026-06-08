import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取 ch_04 的绝对路径
parent_dir = os.path.dirname(current_dir)                 # 获取项目总根目录
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ch_03.activation_function import sigmoid
from ch_03.output_layer import softmax
from loss_func import cross_entropy_error
from math_basics import numerical_gradient as g_evaluator   

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, x, t):
        '''x:输入数据 t:监督数据'''
        y = self.predict(x)

        return cross_entropy_error(y,t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        '''x:输入数据 t:监督数据'''
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = g_evaluator(loss_W, self.params['W1'])
        grads['b1'] = g_evaluator(loss_W, self.params['b1'])
        grads['W2'] = g_evaluator(loss_W, self.params['W2'])
        grads['b2'] = g_evaluator(loss_W, self.params['b2'])

        return grads

    