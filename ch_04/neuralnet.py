import sys
import os
import numpy as np
import matplotlib.pyplot as plt  # 引入现代工业标准的绘图正统通道

current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取 ch_04 的绝对路径
parent_dir = os.path.dirname(current_dir)                 # 获取项目总根目录
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ch_03.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train,t_train), (x_test,t_test) = load_mnist(normalize = True, one_hot_label= True)

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size/batch_size,1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch,t_batch)

    # 更新参数
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        

# =====================================================================
# 核心渲染闭环：绘制损失函数下降轨迹
# =====================================================================
# 1. 创建画布
plt.figure(figsize=(8, 5))

# 2. 绘制折线图
# 横坐标是迭代步数（0 到 len-1），纵坐标是对应的 Loss 值
plt.plot(train_loss_list, label='Training Loss', color='#1f77b4', linewidth=1.5)

# 3. 润饰图表元素
plt.title('Neural Network Training Loss Curve', fontsize=14, fontweight='bold')
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)  # 开启网格线，方便定量分析
plt.legend(fontsize=11)

# 4. 强制渲染并弹出窗口
plt.tight_layout()  # 自动调整布局，防止标签切边
plt.show()