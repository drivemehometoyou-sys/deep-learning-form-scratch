import numpy as np

# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y
# # 存在溢出问题

# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a-c)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
    
#     return y

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

# 只有当你直接右键运行 output_layer.py 本身时，下面三行才会执行
# 如果是被别人 import 导入，下面三行会自动闭嘴（隐形）
if __name__ == '__main__':
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    print(y)
