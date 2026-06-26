import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt

def load_data_fashion_mnist(batch_size, resize=None, root='C:\\Users\\21495\\Desktop\\python\\DeepLearning\\Multilayer_Perceptrons\\data'):
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


net = nn.Sequential(
    nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10)
)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01) 
net.apply(init_weights)
        
batch_size = 256
lr = 0.1
num_epochs = 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = load_data_fashion_mnist(batch_size)


















