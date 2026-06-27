import hashlib
import os
import tarfile
import zipfile
import requests

#建立一个dict，存放数据集的相关二元组
# 包含数据集的url和检验文件完整性所需的sha-1密钥

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

'''
下载数据集并返回本地文件名，如果本地存在则不再下载
'''
def download(name, cache_dir=os.path.join('DeepLearning\\2_Multilayer_Perceptrons', 'Kaggle_data')):
    '''下载文件,返回本地文件名'''
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url,sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir,url.split('/')[-1])
    if os.path.exists(fname):
        hash_sha1 = hashlib.sha1()
        with open(fname,'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                hash_sha1.update(data)
        if hash_sha1.hexdigest() == sha1_hash:
            return fname
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url,stream = True)
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return fname

'''
我们还需实现两个实用函数：
一个将下载并解压缩一个zip或tar文件
另一个是将使用的所有数据集从DATA_HUB下载到缓存目录中。
'''

def download_extract(name, folder=None):
    '''下载并解压zip/tar文件'''
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    '''下载DATA_HUB中所有数据集'''
    for name in DATA_HUB:
        download(name)

'''             Kaggle房价预测数据集             '''

import pandas as pd
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d2a53e81')
DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011fbd6a2e8b9c03b0a3bfa5a5d0e')

train_data = pd.read_csv(download('kaggle_house_train'))
train_test = pd.read_csv(download('kaggle_house_test'))

print('Training data shape:', train_data.shape)
print('Test data shape:', train_test.shape)

#让我们看看前四个和最后两个特征，以及相应标签（房价）。
print('Training data samples:')
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
print('Test data samples:')
print(train_test.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

#我们可以看到，在每个样本中，第一个特征是ID， 这有助于模型识别每个训练样本。 虽然这很方便，但它不携带任何用于预测的信息。
# 因此，在将数据提供给模型之前，我们将其从数据集中删除。
all_features = pd.concat((train_data.iloc[:, 1:-1], train_test.iloc[:, 1:]))

'''         1、数据预处理            '''
'''
在开始建模之前，我们需要对数据进行预处理。
    首先，我们将所有缺失的值替换为相应特征的平均值。
    我们通过将特征重新缩放到零均值和单位方差来标准化数据
'''
#可根据训练数据计算均值和标准差
# 取出所有数值型特征的列名（非 object 类型列）
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

# 对数值型特征做标准化：
# 1. 减去该列均值，使数据中心靠近 0
# 2. 除以该列标准差，使数据尺度统一到单位方差
# 这样可以避免不同特征量纲差异过大，利于模型训练
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

'''接下来，我们处理离散值:
 这包括诸如“MSZoning”之类的特征,用独热标签去标记，并且当做特征向量。
 eg.“MSZoning”包含值“RL”和“Rm”：
    我们将创建两个新特征“MSZoning_RL”和“MSZoning_Rm”
    并将其设置为0或1，表示每个样本是否具有该值。

    pandas软件包会自动为我们实现这一点。
'''
all_features = pd.get_dummies(all_features, dummy_na=True).astype(float).fillna(0)
 #dummy_na=True表示将缺失值也作为一个特征；astype(float)确保所有列为数值类型


'''
此转换会将特征的总数量从79个增加到331个。
 我们可以从pandas格式中提取NumPy格式并将其转换为张量表示用于训练。
'''
n_train = train_data.shape[0]

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.to_numpy().reshape(-1, 1), dtype=torch.float32)

'''     2、训练        '''
loss = nn.MSELoss()  # 均方误差损失函数

in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 1),
    )
    return net

#房价就像股票价格一样，我们关心的是相对数量，而不是绝对数量。
#解决这个问题的一种方法是用价格预测的对数来衡量差异。

def log_rmse(net,features,labels):
    #将预测值限制在1以上，取对数后不会出现负无穷
    clipped_preds = torch.clamp(net(features), min=1.0)
    rmse = torch.sqrt(loss(clipped_preds.log(), 
                           labels.log()))
    return rmse.item()

#Adam优化器的主要吸引力在于它对初始学习率不那么敏感。

def train(net,train_features,train_labels,test_features,test_labels,
         num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls = [],[]
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features,train_labels),
        batch_size=batch_size,
        shuffle=True
    )
    #使用Adam优化器:
    optimizer = torch.optim.Adam(net.parameters(), 
                                lr=learning_rate,
                                weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X),y)
            l.backward()
            optimizer.step()

        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

'''
K折交叉验证， 它有助于模型选择和超参数调整
具体地说，它选择第i个切片作为验证数据，其余部分作为训练数据
'''
def get_k_fold_data(k,i,X,y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = [], []
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        else:
            X_train.append(X_part)
            y_train.append(y_part)
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    return X_train, y_train, X_valid, y_valid

#K折交叉验证的训练和验证:

def k_fold(k,X_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k,i,X_train,y_train)
        net = get_net()
        train_ls,valid_ls = train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plt.plot(range(1,num_epochs+1),train_ls,label='train')
            plt.plot(range(1,num_epochs+1),valid_ls,label='valid')
            plt.xlabel('epoch')
            plt.ylabel('rmse')
            plt.legend()
            plt.show()
        print(f'折{i+1},训练log rmse{float(train_ls[-1]):f},'
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

'''         3、模型选择和超参数调整            '''

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64

train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    plt.plot(np.arange(1, num_epochs + 1), train_ls)
    plt.xlabel('epoch')
    plt.ylabel('log rmse')
    plt.xlim(1, num_epochs)
    plt.yscale('log')
    plt.show()
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, train_test,
               num_epochs, lr, weight_decay, batch_size)
















