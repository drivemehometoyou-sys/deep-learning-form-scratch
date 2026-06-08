import numpy as np
import matplotlib.pyplot as plt

# 不好的实现
# def numerical_diff(f,x):
#     '''前向差分'''
#     h = 10e-50
#     return (f(x+h) - f(x)) / h

def numerical_diff(f,x):
    '''中心差分'''
    h = 10e-4
    return (f(x+h) - f(x-h)) / (2*h)


def _numerical_gradient(f,x):

    h = 1e-4
    grad = np.zeros_like(x) # 生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx] #记下这个位置原本的数学值
        
        # f(x+h) 的计算
        x[idx] = tmp_val + h # 只让这一个位置增加 h
        fxh1 = f(x)
        # f(x-h) 的计算
        x[idx] = tmp_val - h # 只让这一个位置减去 h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h) # 算出该位置的偏导数
        
        x[idx] = tmp_val
    
    return grad

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=[['readwrite']])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad


def function_1(x):
    return 0.01 * x**2 + 0.1 * x

def function_2(x):
    # return x[0]**2 + x[1]**2
    return np.sum(x**2)


def get_tangent_line_function(f, x_point):
    '''用数值微分的值作为斜率在对应点处画直线'''
    # Calculate the localized derivative (slope, k)
    slope = numerical_diff(f, x_point)
    
    # Calculate the Y-intercept (b = f(x) - k*x)
    intercept = f(x_point) - slope * x_point
    
    # Return the dynamic linear function g(t)
    return lambda t: slope * t + intercept

if __name__ == '__main__':
    # print(np.float32(10e-50))
    
    # x = np.arange(0.0,20.0,0.1)
    # y1 = function_1(x)
    
    # tf = get_tangent_line_function(function_1, 5)
    # y2 = tf(x)
    
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.plot(x,y1)
    # plt.plot(x,y2)
    # plt.show()

    # diff1 = numerical_diff(function_1,5)
    # diff2 = numerical_diff(function_1,10)
    # print(diff1)
    # print(diff2)

    print(numerical_gradient(function_2,np.array([3.0,4.0])))
    print(numerical_gradient(function_2,np.array([3.0,4.0])))
                             