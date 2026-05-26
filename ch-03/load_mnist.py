import sys,os
# sys.path.append(os.pardir)   # 其实用不上了，因为我把mnist.py已经移进ch-03了
# from dataset.minst import load_mnist

# from minst import load_mnist  #Pylance 插件来说，它们默认只会把最外层的根目录加入到搜索路径里。
# 它们在找 mnist 时，会去最外层找，结果最外层只有 ch-01, ch-02, ch-03

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from mnist import load_mnist




(x_train,t_train) ,(x_test,t_test) = load_mnist(flatten=True,normalize=False)

# 输出各个数据的形状
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

