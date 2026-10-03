# Deep Learning from Scratch

一个使用 Python 和 NumPy 从零实现深度学习基础算法的学习工程。

项目按照《深度学习入门：基于 Python 的理论与实现》的知识路线逐章推进，不依赖 TensorFlow、PyTorch 等深度学习框架，重点理解神经网络内部的数据流动、梯度计算和训练过程。

## 学习内容

| 章节 | 主题 | 主要内容 |
| --- | --- | --- |
| `ch_01` | Python / NumPy 基础 | NumPy 数组、矩阵运算、Matplotlib 绘图 |
| `ch_02` | 感知机 | AND、NAND、OR、XOR 逻辑门 |
| `ch_03` | 神经网络 | 激活函数、Softmax、多维数组、MNIST 推理 |
| `ch_04` | 神经网络训练 | 损失函数、数值微分、梯度下降、mini-batch、两层网络 |
| `ch_05` | 误差反向传播 | 计算图、基础层、反向传播、梯度检查 |
| `ch_06` | 训练技巧 | SGD、Momentum、AdaGrad、RMSprop、Adam、权重初始化、Batch Normalization、Dropout、权值衰减、超参数搜索 |
| `ch_07` | 卷积神经网络 | Convolution、Pooling、im2col、简单 CNN 训练 |
| `ch_08` | 深度网络 | 深层 CNN、Dropout、模型参数保存、半精度推理与误分类分析 |

## 环境要求

- Python 3
- NumPy
- Matplotlib
- Pillow（仅显示 MNIST 图片时需要）

安装依赖：

```bash
python -m pip install numpy matplotlib Pillow
```

如果需要运行静态类型检查，还需安装 Pyright：

```bash
npm install -g pyright
```

## 快速开始

克隆项目并进入仓库根目录：

```bash
git clone https://github.com/drivemehometoyou-sys/deep-learning-form-scratch.git
cd deep-learning-form-scratch
```

所有脚本都建议从仓库根目录运行。部分代码会根据脚本位置寻找数据或跨章节导入模块，因此不要先进入章节目录。

```bash
# NumPy 基础
python ch_01/numpy_test.py

# 感知机与逻辑门
python ch_02/perceptron.py

# MNIST 推理
python ch_03/neuralnet_mnist.py

# 两层神经网络
python ch_04/two_layer_net.py

# 反向传播梯度检查
python ch_05/gradient_check.py

# 优化器效果比较
python ch_06/optimizer_compare_naive.py

# 训练简单卷积神经网络
python ch_07/train_convnet.py

# 训练深层卷积神经网络
python ch_08/train_deepnet.py
```

训练 CNN 和深层 CNN 的计算量较大，在只使用 CPU 时可能需要较长时间。学习或调试时，可以在训练脚本中截取一小部分 MNIST 数据，并减少训练轮数。

## MNIST 数据集

首次调用 `ch_03/mnist.py` 中的 `load_mnist()` 时，程序会自动下载 MNIST 数据集，并在 `ch_03/` 中生成缓存文件：

- `*.gz`：MNIST 原始压缩数据
- `mnist.pkl`：转换后的 NumPy 数据缓存

这些文件已经加入 `.gitignore`，不会提交到 Git。

`ch_03/neuralnet_mnist.py` 还需要预训练权重 `ch_03/sample_weight.pkl`。由于 `*.pkl` 被忽略，该文件不包含在仓库中；运行前需要自行放入对应目录。第七、八章训练脚本生成的 `params.pkl` 和 `deep_convnet_params.pkl` 同样只保存在本地。

## 项目结构

```text
deep-learning-form-scratch/
├── ch_01/              # NumPy 与 Matplotlib 基础
├── ch_02/              # 感知机
├── ch_03/              # 神经网络与 MNIST
├── ch_04/              # 损失函数、梯度与训练
├── ch_05/              # 误差反向传播
├── ch_06/              # 优化器与训练技巧
├── ch_07/              # 简单卷积神经网络
├── ch_08/              # 深度卷积神经网络
├── pyrightconfig.json  # Pyright 配置
└── README.md
```

## 使用说明

- 本项目以教学和实验为目的，代码优先展示算法原理，而不是追求生产环境性能。
- 绘图脚本会打开 Matplotlib 图形窗口。
- `ch_08/half_float_network.py` 和 `ch_08/misclassified_mnist.py` 目前保留了原书的 `dataset.mnist` 导入方式和模型权重依赖，直接运行前需要将导入调整为本项目的 `ch_03.mnist`，并准备相应参数文件。
- 可以在仓库根目录运行 `pyright` 进行静态类型检查。

## 核心思路

这个项目刻意不使用成熟深度学习框架封装好的网络层和自动求导，而是亲手实现：

1. 神经元如何进行加权求和与激活；
2. 损失函数如何衡量预测结果；
3. 梯度如何通过反向传播传递；
4. 优化器如何更新网络参数；
5. 卷积和池化如何提取图像特征；
6. 正则化和初始化策略如何影响训练效果。

适合希望理解“神经网络为什么能学习”，而不只是学习框架 API 的初学者。
