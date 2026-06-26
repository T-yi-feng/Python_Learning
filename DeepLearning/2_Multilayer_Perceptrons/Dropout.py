import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt

# dropout_layer 函数:
#   该函数以dropout的概率丢弃张量输入X中的元素
#   如上所述重新缩放剩余部分：将剩余部分除以1.0-dropout。

def dropout_layer(X,dropout_prob):
    assert 0 <= dropout_prob <= 1
    # 在训练模式下使用丢弃法
    if dropout_prob == 1:
        return torch.zeros_like(X)
    if dropout_prob == 0:
        return X
    mask = (torch.rand(X.shape) > dropout_prob).float()
    return mask * X / (1.0 - dropout_prob)

'''
                    测试dropout_layer函数
'''
# X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
# print(X)
# print(dropout_layer(X, 0.))
# print(dropout_layer(X, 0.5))
# print(dropout_layer(X, 1.))

'''
                    定义模型参数
两个隐藏层的多层感知机，每个隐藏层包含256个单元
使用ReLU激活函数
'''
num_inputs=784
num_outputs=10
num_hiddens=256
num_hiddens2=256

'''
                        定义模型
我们可以将暂退法应用于每个隐藏层的输出（在激活函数之后）
并且可以为每一层分别设置暂退概率： 
    常见的技巧是在靠近输入层的地方设置较低的暂退概率。
    模型将第一个和第二个隐藏层的暂退概率分别设置为0.2和0.5
    并且暂退法只在训练期间有效。
'''

dropout1,dropout2=0.2,0.5

class Net(nn.Module):
    def __init__(self,num_inputs,num_hiddens,num_hiddens2,
                 is_training=True):
        super(Net,self).__init__()  
         #继承Module类的属性和方法
        self.num_inputs=num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs,num_hiddens)
        self.lin2 = nn.Linear(num_hiddens,num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2,num_outputs)
        self.relu = nn.ReLU()

    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
        #在训练模式下使用丢弃法:
        if self.training == True:
            H1 = dropout_layer(H1,dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2,dropout2)
        return self.lin3(H2)

'''
                    训练和测试
类似之前的
'''
num_epochs,lr,batch_size=10,0.5,256

loss = nn.CrossEntropyLoss(reduction='none')

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


train_iter,test_iter=load_data_fashion_mnist(batch_size)

trainer = torch.optim.SGD(Net(num_inputs,num_hiddens,num_hiddens2).parameters(),lr=lr)






