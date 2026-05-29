import sys, os
import numpy as np

# 1. 动态拿到当前 mini_batch.py 所在的绝对路径（即 ch_04 文件夹）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 动态拼出它兄弟文件夹 ch_03 的绝对路径
ch03_dir = os.path.join(os.path.dirname(current_dir), "ch_03")
# 3. 把 ch_03 的绝对路径加入 Python 的搜寻地图中
sys.path.append(ch03_dir)

# 4. 既然 ch_03 已经在地图里了，直接 from mnist 即可，不需要再带 ch_03 前缀了！
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True,one_hot_label=True)

# print(x_train.shape)  # (60000, 784) 
# print(t_train.shape)  # (60000, 10)

train_size = x_train.shape[0]
# print(train_size) # 60000
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

                                                                                                                                                                                                                                                                                             

