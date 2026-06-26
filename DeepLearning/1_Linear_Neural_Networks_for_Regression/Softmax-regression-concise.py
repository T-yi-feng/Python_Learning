import torch
from torch import nn
import torchvision
from matplotlib import pyplot as plt

def load_data_fashion_mnist(batch_size, resize=None, root='C:\\Users\\21495\\Desktop\\python\\DeepLearning\\Linear_Neural_Networks_for_Regression\\data'):
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
train_iter,test_iter = load_data_fashion_mnist(batch_size)

'''
初始化模型参数
我们在线性层前定义了展平层（flatten），来调整网络输入的形状
'''
net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_weights)

'''
交叉熵损失函数：
'''
loss = nn.CrossEntropyLoss(reduction='none')

'''
优化算法：
在这里，我们使用学习率为0.1的小批量随机梯度下降作为优化算法。
这与我们在线性回归例子中的相同，这说明了优化器的普适性。
'''

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

'''
训练：
'''
num_epochs = 20

'''
展示：
'''

def accuracy(y_hat, y):
    """计算预测正确的样本数。"""
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    return float((y_hat.type(y.dtype) == y).type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算模型在数据集上的准确率。"""
    net.eval()
    metric_correct, metric_total = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            metric_correct += accuracy(net(X), y)
            metric_total += y.numel()
    net.train()
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
        trainer.zero_grad()
        l.mean().backward()
        trainer.step()

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


def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def predict_and_show(net, data_iter, n=9):
    net.eval()
    X, y = next(iter(data_iter))
    X, y = X[:n], y[:n]
    with torch.no_grad():
        preds = net(X).argmax(dim=1)

    pred_labels = get_fashion_mnist_labels(preds)
    true_labels = get_fashion_mnist_labels(y)

    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[i].squeeze().numpy(), cmap='gray')
        plt.title(f'pred: {pred_labels[i]}\ntrue: {true_labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    net.train()


predict_and_show(net, test_iter, n=9)






