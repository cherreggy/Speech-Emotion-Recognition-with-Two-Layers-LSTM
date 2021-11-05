import torch
import torch.nn as nn
from Model import LSTMNet
import torch.utils.data as Data
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from draw_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

DATA_BASE = 'iemocap'  # 数据库接口
if DATA_BASE == 'emodb':
    CLASS_NUM = 7
    BATCH_SIZE = 20
else:
    CLASS_NUM = 4
    BATCH_SIZE = 256

torch.manual_seed(1)  # 随机种子
np.random.seed(0)

# 加载数据集
# (535, 60, 39) (data_length, time_step, MFCCs_size)
X_or = np.load('data/'+DATA_BASE+'_data_set.npy')
# (535, 7)
Y_or = np.load('data/'+DATA_BASE+'_label_set.npy')
# (10, 2)
record_list = np.load('data/'+DATA_BASE+'_record_list.npy')
data_len = len(X_or)  # 数据数目


def z_score(data, mean, std):
    """对输入数据进行z-score归一化，所选维度为timestep * N"""
    s = data.shape
    data = data.reshape([s[0] * s[1], s[2]])
    data = (data - mean) / std
    return data.reshape(s)


# 对每折数据提取特征
for session in range(5):
    # 将数据集分为第一个人、第二个人和其他人三部分
    person1_X = X_or[record_list[session * 2][0]:record_list[session * 2][1] + 1]
    person1_Y = Y_or[record_list[session * 2][0]:record_list[session * 2][1] + 1]
    person2_X = X_or[record_list[session * 2 + 1][0]:record_list[session * 2 + 1][1] + 1]
    person2_Y = Y_or[record_list[session * 2 + 1][0]:record_list[session * 2 + 1][1] + 1]
    other_X = np.delete(X_or, [i for i in range(record_list[session * 2][0], record_list[session * 2 + 1][1] + 1)], 0)
    other_Y = np.delete(Y_or, [i for i in range(record_list[session * 2][0], record_list[session * 2 + 1][1] + 1)], 0)

    for turn in range(2):
        print('正在进行第{}折特征提取----------------------------------'.format(session * 2 + turn + 1))
        # 正则化数据
        other_X_mean = np.mean(np.mean(other_X, 1), 0)  # 保留均值、标准差，验证用
        other_X_std = np.std(np.std(other_X, 1), 0)
        other_X = z_score(other_X, other_X_mean, other_X_std)  # 正则化训练集
        person1_X = z_score(person1_X, other_X_mean, other_X_std)  # 正则化第一个人的数据
        person2_X = z_score(person2_X, other_X_mean, other_X_std)  # 正则化第二个人的数据
        # 设置DATA LOADER，交叉二者信息
        if turn == 0:
            train_ds = Data.TensorDataset(torch.FloatTensor(other_X), torch.LongTensor(other_Y))
            valid_ds = Data.TensorDataset(torch.FloatTensor(person2_X), torch.LongTensor(person2_Y))
            test_ds = Data.TensorDataset(torch.FloatTensor(person1_X), torch.LongTensor(person1_Y))
        else:
            train_ds = Data.TensorDataset(torch.FloatTensor(other_X), torch.LongTensor(other_Y))
            valid_ds = Data.TensorDataset(torch.FloatTensor(person1_X), torch.LongTensor(person1_Y))
            test_ds = Data.TensorDataset(torch.FloatTensor(person2_X), torch.LongTensor(person2_Y))
        # 载入本折最优模型
        lstm_dict = torch.load((DATA_BASE + '_model/network{}.pt').format(session * 2 + turn + 1))  # 加载最好模型
        lstm = LSTMNet(39)
        lstm.load_state_dict(lstm_dict)
        lstm.eval()  # 关闭dropout
        # 构建Data Loader
        train_loader = Data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
        valid_loader = Data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = Data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 提取训练集特征
        tmp_x = []
        tmp_y = []
        for batch, data in enumerate(train_loader):
            init_hidden1 = lstm.init_Hidden(len(data[0]), 1, 512)
            init_hidden2 = lstm.init_Hidden(len(data[0]), 1, 256)
            x, y = Variable(data[0]), Variable(data[1])
            tmp_y.extend(y.tolist())
            _ = lstm(x, init_hidden1, init_hidden2)  # 将数据塞入但不使用
            tmp_x.extend(lstm.vector.tolist())
        t_x = np.array(tmp_x)
        t_y = np.array(tmp_y)
        np.save(DATA_BASE + '/fold{}/train_x.npy'.format(session * 2 + turn + 1), t_x)
        np.save(DATA_BASE + '/fold{}/train_y.npy'.format(session * 2 + turn + 1), t_y)

        # 提取验证集特征
        tmp_x = []
        tmp_y = []
        for batch, data in enumerate(valid_loader):
            init_hidden1 = lstm.init_Hidden(len(data[0]), 1, 512)
            init_hidden2 = lstm.init_Hidden(len(data[0]), 1, 256)
            x, y = Variable(data[0]), Variable(data[1])
            tmp_y.extend(y.tolist())
            _ = lstm(x, init_hidden1, init_hidden2)  # 将数据塞入但不使用
            tmp_x.extend(lstm.vector.tolist())
        t_x = np.array(tmp_x)
        t_y = np.array(tmp_y)
        np.save(DATA_BASE + '/fold{}/vali_x.npy'.format(session * 2 + turn + 1), t_x)
        np.save(DATA_BASE + '/fold{}/vali_y.npy'.format(session * 2 + turn + 1), t_y)

        # 提取测试集特征
        tmp_x = []
        tmp_y = []
        for batch, data in enumerate(test_loader):
            init_hidden1 = lstm.init_Hidden(len(data[0]), 1, 512)
            init_hidden2 = lstm.init_Hidden(len(data[0]), 1, 256)
            x, y = Variable(data[0]), Variable(data[1])
            tmp_y.extend(y.tolist())
            _ = lstm(x, init_hidden1, init_hidden2)  # 将数据塞入但不使用
            tmp_x.extend(lstm.vector.tolist())
        t_x = np.array(tmp_x)
        t_y = np.array(tmp_y)
        np.save(DATA_BASE + '/fold{}/test_x.npy'.format(session * 2 + turn + 1), t_x)
        np.save(DATA_BASE + '/fold{}/test_y.npy'.format(session * 2 + turn + 1), t_y)
