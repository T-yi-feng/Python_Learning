import torch
from torch import nn
import torchvision 
from matplotlib import pyplot as plt

def load_data_fashion_mnist(batch_size, resize=None, root='C:\\Users\\21495\\Desktop\\python\\DeepLearning\\2_Multilayer_Perceptrons\\data'):
    """
    下载Fashion-MNIST数据集，然后将其加载到内存中
    """
    transforms = []
    if resize:
        transforms.append(torchvision.transforms.Resize(resize))
    transforms.append(torchvision.transforms.ToTensor())
    trans = torchvision.transforms.Compose(transforms)

    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, transform=trans, download=True
    )

    return (
        torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False),
    )

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
num_hiddens = 256    #256个隐藏单元h

w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [w1,b1,w2,b2]

'''
实现ReLU激活函数；
'''
def ReLU(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)

'''
实现模型：
'''
def net(X):
    X = X.reshape((-1, num_inputs))
    h = ReLU(torch.matmul(X, w1) + b1)
    return torch.matmul(h, w2) + b2

loss = nn.CrossEntropyLoss(reduction='none')

'''
多层感知机训练过程：(与softmax回归的训练过程类似)
'''
num_epochs, lr = 10, 0.2

updater = torch.optim.SGD(params, lr=lr)



def accuracy(y_hat, y):
    """计算预测正确的样本数。"""
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    return float((y_hat.type(y.dtype) == y).type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算模型在数据集上的准确率。"""
    metric_correct, metric_total = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            metric_correct += accuracy(net(X), y)
            metric_total += y.numel()
    return metric_correct / metric_total


# train loss：训练集上的平均损失，越小表示模型拟合训练集越好
# train acc：训练集上的准确率，越大表示训练集预测正确的比例越高
# test acc：测试集上的准确率，越大表示模型在未见数据上的泛化能力越好
train_l, train_acc, test_acc = [], [], []

for epoch in range(num_epochs):
    metric_loss, metric_correct, metric_total = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()

        metric_loss += float(l.sum())
        metric_correct += accuracy(y_hat, y)
        metric_total += y.numel()

    train_l.append(metric_loss / metric_total)
    train_acc.append(metric_correct / metric_total)
    test_acc.append(evaluate_accuracy(net, test_iter))

    print(
        f'epoch {epoch + 1}: '
        f'train loss {train_l[-1]:.4f}, '
        f'train acc {train_acc[-1]:.4f}, '
        f'test acc {test_acc[-1]:.4f}'
    )

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_l, label='train loss')
plt.plot(range(1, num_epochs + 1), train_acc, label='train acc')
plt.plot(range(1, num_epochs + 1), test_acc, label='test acc')
plt.xlabel('epoch')
plt.ylabel('value')
plt.title('Training Process')
plt.legend()
plt.grid(True)
plt.show()




























