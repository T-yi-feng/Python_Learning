'''
                    一、从零实现softmax回归
'''

import torch
import numpy as np
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display
from matplotlib import pyplot as plt


'''首先先读取MINIST数据集'''
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="C:\\Users\\21495\\Desktop\\python\\DeepLearning\\Linear_Neural_Networks_for_Regression\\data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="C:\\Users\\21495\\Desktop\\python\\DeepLearning\\Linear_Neural_Networks_for_Regression\\data", train=False, transform=trans, download=True)

# Fashion-MNIST由10个类别的图像组成，
# 每个类别由训练数据集中的6000张图像和测试数据集中的1000张图像组成。
# 因此，训练集和测试集分别包含60000和10000张图像。
# 测试数据集不会用于训练，只用于评估模型性能.

print("训练集样本数:", len(mnist_train), "测试集样本数:", len(mnist_test))
print("训练集样本形状:", mnist_train[0][0].shape, "训练集标签:", mnist_train[0][1])
print("测试集样本形状:", mnist_test[0][0].shape, "测试集标签:", mnist_test[0][1])

# Fashion-MNIST中包含的10个类别分别为:
# t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、
# coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、
# bag（包）和ankle boot（短靴）。
# 以下函数用于在数字标签索引及其文本名称之间进行转换.

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

'''小批量读取数据集'''
batch_size = 256

def get_dataloader_workers():
    """Windows 上 num_workers>0 会报多进程错误，设为 0"""
    return 0

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers())
'''
初始化模型参数

每个图像都是28×28像素的灰度图像。
我们将把每个图像展平为长度为784的向量，一共有10个类别
因此输出层的输出个数为10。
从而可以得到权重的形状为784×10，偏差的形状为1x10。
'''

num_inputs = 784
num_outputs = 10

w = torch.randn(num_inputs, num_outputs, requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

'''
定义softmax操作

回想一下，实现softmax由三个步骤组成：

    对每个项求幂（使用exp）；

    对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；

    将每一行除以其规范化常数，确保结果的和为1。

'''

def softmax(O):
    O_exp = torch.exp(O)
    partition = O_exp.sum(dim = 1,keepdim = True)
    return O_exp / partition  # 这里应用了广播机制

'''定义模型'''

def net(X):
    return softmax(torch.matmul(X.reshape(-1, w.shape[0]), w) + b)

'''定义交叉熵损失函数'''
def cross_entropy(y_hat,y):
    # y_hat: 2D tensor of shape (batch_size, num_classes), 每一行是预测的类别概率分布
    # y: 1D tensor of length batch_size，包含每个样本的真实标签（0..num_classes-1）
    # y_hat[range(len(y_hat)), y] 会选择每一行中对应真实类别的预测概率
    # 返回的是每个样本的交叉熵损失：-log(p_true)
    return -torch.log(y_hat[range(len(y_hat)), y])

'''计算分类精度'''
def accuracy(y_hat, y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)  # 取预测概率最大的类别作为预测结果
    cmp = y_hat.type(y.dtype) == y  
    # 将预测结果与真实标签进行比较，得到布尔值张量
    return float(cmp.type(y.dtype).sum())  
    # 将布尔值张量转换为数值类型，并求和得到正确预测的样本数

'''评估在任意模型net的精度'''
class Accumulator:
            """累加器，用于统计多个变量的和"""
            def __init__(self, n):
                self.data = [0.0] * n

            def add(self, *args):
                for i, a in enumerate(args):
                    self.data[i] += float(a)

            def reset(self):
                for i in range(len(self.data)):
                    self.data[i] = 0.0

            def __getitem__(self, idx):
                return self.data[idx]

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数，总样本数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]  # 返回精度

'''训练模型'''

def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和，训练准确度总和，样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

'''
在展示训练函数的实现之前
我们定义一个在动画中绘制数据的实用程序类Animator
它能够简化本书其余部分的代码。
'''
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        plt.switch_backend('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self.axes[0].set(
            xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
            xscale=xscale, yscale=yscale)
        self.legend = legend
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        if self.legend:
            self.axes[0].legend(self.legend)
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.15

def updater(batch_size):
    # 使用自定义的随机梯度下降更新参数 w 和 b
    global w, b
    with torch.no_grad():
        # 如果没有梯度，则直接返回
        if w.grad is None or b.grad is None:
            return
        # 在.grad.data上进行原地运算以避免类型检查/广播问题
        w.grad.data /= batch_size
        b.grad.data /= batch_size
        # 在.data上更新参数，保持requires_grad=True不变
        w.data -= lr * w.grad.data
        b.data -= lr * b.grad.data
        # 清零梯度（grad不为None，此时可安全调用）
        w.grad.zero_()
        b.grad.zero_()


num_epochs = 50
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

'''预测'''
def predict_ch3(net, test_iter, n=6):  #@save
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    plt.show(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()

predict_ch3(net, test_iter)










