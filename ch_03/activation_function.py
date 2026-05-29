import numpy as np
import matplotlib.pyplot as plt

# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0
# 这个函数无法接收Numpy数组

# def step_function(x):
#     y = x>0
#     return y.astype(int)


# 我自己的写法
def step_function(x):
    return (x>0).astype(int)

# 鱼书写法
# def step_function(x):
#     return np.array(x>0,dtype =int)

# x = np.arange(-5.0,5.0,0.1)
# y = step_function(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()





def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# x = np.array([-1.0,1.0,2.0])
# print(sigmoid(x))

# x = np.arange(-5.0,5.0,0.1)
# y = sigmoid(x)
# plt.plot(x,y)
# plt.ylim(-0.1,1.1)
# plt.show()



def relu(x):
    return np.maximum(0,x)

# x = np.array([-1.0,1.0,2.0])
# print(relu(x))

# x = np.arange(-5.0,5.0,0.1)
# y = relu(x)
# plt.plot(x,y)
# plt.show()
