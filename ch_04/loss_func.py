import numpy as np
y_2 = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
y_7 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
t = [0,0,1,0,0,0,0,0,0,0]

# 均方误差（ mean squared error
def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

# 交叉熵误差（ cross_entropy_error
# def cross_entropy_error(y,t):
#     delta = 1e-7
    # return -np.sum(t*np.log(y+delta))

# mini-batch版 交叉熵误差
def cross_entropy_error(y,t):
    # y: 模型的预测矩阵
    # t: 监督数据矩阵
    if y.ndim == 1:   # 数据的维度为 1
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)   # 变成 二维方阵表格

    batch_size = y.shape[0]    # 如果是单张照片，不进行刚才的变维操作：此处就会测得size为10
    return -np.sum(t*np.log(y+1e-7)) / batch_size

# 该函数针对 监督数据 是标签格式
def cross_entropy_error_tag(y,t):
    # y: 模型的预测矩阵
    # t: 监督数据矩阵
    if y.ndim == 1:   # 数据的维度为 1
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)   

    batch_size = y.shape[0]    
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7)) / batch_size
# y[np.arange(batch_size),t]  花式索引 ： arnage生成的是0~9，监督数据 t 刚好对应猜测的下标
# 靠 花式索引 把预测矩阵中的对应监督数据猜测概率取出来了

if __name__ == '__main__':
    print("mean squared error:")
    print(mean_squared_error(np.array(y_2),np.array(t)))                                                                                                                                                                                                                                                                                                
    print(mean_squared_error(np.array(y_7),np.array(t)))

    print("cross_entropy_error:")
    print(cross_entropy_error(np.array(y_2),np.array(t)))                                                                                                                                                                                                                                                                                                
    print(cross_entropy_error(np.array(y_7),np.array(t)))

    print("mini-batch:cross_entropy_error:")
    print(cross_entropy_error(np.array(y_2),np.array(t)))                                                                                                                                                                                                                                                                                                
    print(cross_entropy_error(np.array(y_7),np.array(t)))