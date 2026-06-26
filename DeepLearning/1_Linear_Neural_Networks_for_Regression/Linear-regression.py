#minibatch-gradient-descent:需要将数据集分成小批量，每次使用一个小批量来更新模型参数。这样可以在每次迭代中减少计算量，同时也能更快地收敛。
#同时需要进行矢量化，同时处理多批次batch而不是for循环处理每个样本
# 这样可以充分利用矩阵运算的优势，提高计算效率。
import math
import time
import numpy as np
import torch
from matplotlib import pyplot as plt


#定义一个计数器：
class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def __call__(self, func):
        """允许作为装饰器 @timer 使用"""
        def wrapped(*args, **kwargs):
            self.start()
            result = func(*args, **kwargs)
            print(f"{func.__name__} took {self.stop():.6f} sec")
            return result
        return wrapped

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


timer = Timer()

'''
                        从零实现线性回归
'''

'''
第一步：生成数据集：
生成一个包含1000个样本的数据集， 每个样本包含从标准正态分布中采样的2个特征。
我们使用线性模型参数和噪声项生成数据集及其标签,
噪声项服从均值为0的正态分布
'''
def synthetic_data(w, b, num_examples):  #@save
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b

    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features,labels = synthetic_data(true_w, true_b, 1000)

# 展示生成的 features 和 labels
plt.figure(figsize=(8, 4))
plt.scatter(features[:, 0].numpy(), labels.numpy(), s=10, alpha=0.6, label='labels vs feature1')
plt.xlabel('feature1')
plt.ylabel('labels')
plt.title('Synthetic Data: features and labels')
plt.legend()
plt.tight_layout()
plt.show()

'''
第二步：读取数据集：
训练模型时要对数据集进行遍历，每次抽取一小批量样本
并使用它们来更新我们的模型。
由于这个过程是训练机器学习算法的基础，所以有必要定义一个函数，
该函数能打乱数据集中的样本并以小批量方式获取数据。

在下面的代码中，我们定义一个data_iter函数，：
    该函数接收批量大小、特征矩阵和标签向量作为输入
    生成大小为batch_size的小批量。
    每个小批量包含一组特征和标签。
'''

def data_iter(batch_size,features,labels):
    # 计算样本总数
    num_examples = len(features)
    # 生成从0到num_examples-1的索引列表（用于表示每个样本的位置）
    indices = list(range(num_examples))
    # 将索引顺序打乱，以便每个epoch随机抽取样本，打破顺序偏差
    np.random.shuffle(indices)  # 样本的读取顺序是随机的
    # 按照batch_size步长遍历打乱后的索引，生成每个小批量的索引范围
    for i in range(0, num_examples, batch_size):
        # 选择当前批次的索引（注意处理最后一个批次可能不足batch_size的情况）
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        # 使用这些索引从特征和标签张量中切片出当前小批量并通过yield返回
        yield features[batch_indices], labels[batch_indices]

#试试：
# batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break

'''
第三步：初始化模型参数
在我们开始用小批量随机梯度下降优化我们的模型参数之前， 我们需要先有一些参数。
通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重
并将偏置初始化为0。
'''
w = torch.normal(0, 0.01, size = (2,1), requires_grad = True)  # 设置requires_grad=True以便在反向传播时计算梯度
b = torch.zeros(1,requires_grad = True)  # 设置requires_grad=True以便在反向传播时计算梯度

'''
第四步：定义模型
'''
def LinearRegression(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

'''
第五步：定义损失函数
'''
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

'''
第六步：定义优化算法
'''
def sgd(param, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    for param in param:
        if param.grad is not None:
            param.data -= lr * param.grad / batch_size
            param.grad.zero_()

'''
训练：
'''

lr = 0.03  # 学习率
num_epochs = 1  # 训练轮数
batch_size = 10  # 批量大小

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        # 计算梯度并更新参数
        l = squared_loss(LinearRegression(X, w, b), y)  # 计算损失
        l.sum().backward()  # 反向传播计算梯度
        sgd([w, b], lr, batch_size)  # 更新参数

        with torch.no_grad():
            # 计算当前批次的平均损失
            train_l = squared_loss(LinearRegression(X, w, b), y)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


# 打印线性回归学习到的参数
print(f'估计的权重 w: {w.reshape(true_w.shape)}')
print(f'估计的偏置 b: {b}')

# 展示样本散点图和拟合直线
with torch.no_grad():
    # 这里有两个特征，模型本质上对应一个平面；为了画成一条直线，固定第二个特征为均值。
    feature2_mean = features[:, 1].mean()
    # feature2_mean is a 0-d tensor; use .item() to pass a Python scalar to full_like
    plot_features = torch.stack(
        (features[:, 0], torch.full_like(features[:, 0], feature2_mean.item())), dim=1
    )
    y_hat = LinearRegression(plot_features, w, b)
    true_y = LinearRegression(plot_features, true_w.reshape(-1, 1), true_b)
    sorted_idx = torch.argsort(features[:, 0])

    plt.figure(figsize=(8, 4))
    plt.scatter(features[:, 0].numpy(), labels.numpy(), s=10, alpha=0.6, label='samples')
    plt.plot(
        features[sorted_idx, 0].numpy(),
        true_y[sorted_idx].numpy(),
        color='green',
        linewidth=2,
        label='true line'
    )
    plt.plot(
        features[sorted_idx, 0].numpy(),
        y_hat[sorted_idx].numpy(),
        color='red',
        linewidth=2,
        label='linear regression'
    )
    plt.xlabel('feature1')
    plt.ylabel('labels')
    plt.title('Samples and Linear Regression Result')
    plt.legend()
    plt.tight_layout()


    plt.show()
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')


'''
                        简洁实现线性回归
'''

'''
在过去的几年里，出于对深度学习强烈的兴趣，
许多公司、学者和业余爱好者开发了各种成熟的开源框架。
这些框架可以自动化基于梯度的学习算法中重复性的工作。
'''

'''
1生成数据集是一样的
'''
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

'''
2读取数据集可以直接使用现有的API框架
'''

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10

data_loader = load_array((features, labels), batch_size)

'''
3定义模型可以直接使用现有的

首先定义一个模型变量net，它是一个Sequential类的实例。
Sequential类将多个层串联在一起
当给定输入数据时，Sequential实例将数据传入到第一层
然后将第一层的输出作为第二层的输入，以此类推。 
在下面的例子中，我们的模型只包含一个层，因此实际上不需要Sequential。
但是由于以后几乎所有的模型都是多层的在这里使用Sequential熟悉“标准的流水线”。
'''
from torch import nn
#nn是神经网络的缩写

net = nn.Sequential(nn.Linear(2, 1)) 
  # nn.Linear(2, 1)：输入特征数为2，输出特征数为1；Sequential类将多个层串联在一起

'''
4初始化模型参数
我们通过net[0]选择网络中的第一个图层
然后使用weight.data和bias.data方法访问参数。
我们还可以使用替换方法normal_和fill_来重写参数值。
'''

net[0].weight.data.normal_(0, 0.01)  # 将权重初始化为均值为0、标准差为0.01的正态分布
net[0].bias.data.fill_(0)  # 将偏置初始化为0

'''
5定义损失函数
计算均方误差使用的是MSELoss类，也称为平方范数。
默认情况下，它返回所有样本损失的平均值。
'''
loss = nn.MSELoss()


'''
6定义优化算法
我们使用PyTorch提供的优化器来实现小批量随机梯度下降。
PyTorch在optim模块中实现了该算法的许多变种。
当我们实例化一个SGD实例时
    我们要指定优化的参数以及优化算法所需的超参数字典。
（可通过net.parameters()从我们的模型中获得）

小批量随机梯度下降只需要设置lr值，这里设置为0.03。
'''
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

'''
训练
'''
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_loader:
        l = loss(net(X), y)  # 计算损失
        trainer.zero_grad()  # 梯度清零
        l.backward()  # 反向传播计算梯度
        trainer.step()  # 更新参数
        trainer.zero_grad()  # 梯度清零，防止梯度累加
    l = loss(net(features), labels)  # 计算整个数据集的损失
    print(f'epoch {epoch + 1}, loss {float(l):f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)



