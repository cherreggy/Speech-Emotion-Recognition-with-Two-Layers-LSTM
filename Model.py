import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMNet(nn.Module):
    def __init__(self, input_size, num_layer=1):
        super(LSTMNet, self).__init__()

        self.dropout = nn.Dropout(0.5)  # 是否需要？
        self.lstm1 = nn.LSTM(input_size, 512, num_layer, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, num_layer, batch_first=True)
        # self.fc1 = nn.Linear(256, 64)
        # self.fc2 = nn.Linear(64, 1024)
        self.out_layer = nn.Linear(256, 7)
        self.bn1 = nn.BatchNorm1d(60, eps=1e-5, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(60, eps=1e-5, momentum=0.1)
        self.soft_max = nn.LogSoftmax(dim=1)  # 这里是否需要还要看输出

    def forward(self, input_data, hidden1=None, hidden2=None):
        # 输入尺寸：batch_size, time_step, length
        output, hidden1 = self.lstm1(input_data, hidden1)  # 隐藏状态：layer_num, batch_size, hidden_size
        output = self.dropout(output)
        output = self.bn1(output)
        output, hidden2 = self.lstm2(output, hidden2)
        output = self.dropout(output)
        output = self.bn2(output)
        self.vector = output[:, -1, ...]
        # output = self.fc1(output[:, -1, ...])
        # output = self.dropout(output)
        # output = self.fc2(output)
        # output = self.dropout(output)
        output = self.out_layer(output[:, -1, ...])
        # 输出尺寸：batch_size, hidden_size
        output = self.soft_max(output)
        return output, hidden1, hidden2

    @staticmethod
    def init_Hidden(batch_size, num_layers, hidden_size, if_use_gpu = False):
        """暂时利用最原始的办法初始化，全为0矩阵"""
        hidden1 = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        cell1 = Variable(torch.zeros(num_layers, batch_size, hidden_size))
        if if_use_gpu:
            hidden1 = hidden1.cuda()
            cell1 = cell1.cuda()
        return hidden1, cell1

    def get_features(self):
        """获得隐藏层输出特征"""
        return self.vector
