import sys
import os
import numpy as np

# 利用高鲁棒性打法，动态抓取项目总根目录并强注到系统寻路最前端
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取 ch_04 的绝对路径
parent_dir = os.path.dirname(current_dir)                 # 获取项目总根目录
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ch_03.output_layer import softmax
from loss_func import cross_entropy_error
from math_basics import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) 
    
    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)
    
        return loss


if __name__ == '__main__':
    net = simpleNet()
    print(net.W)
    x = np.array([0.6,0.9])
    p = net.predict(x)
    print(p)
    t = np.array([0,0,1])
    loss = net.loss(x,t)
    print(loss)
    
    # def f(W):
    #     return net.loss(x,t)
    f = lambda w: net.loss(x,t)
    
    dW = numerical_gradient(f,net.W)
    print(dW)