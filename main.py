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

"""
还未完成的任务：
将代码改为5*2折，分成section，先跑一个：一男一女为1个section，前4个section作为训练集，最后男为验证集，女为测试集
仅在测试集上做归一化
写一个函数，能够保存验证集的最好模型
部署服务器，使用GPU运行
LSTM初始化的问题，后面还是以0初始化的
每一折都要重置参数
每一折都有一个混淆矩阵
画出每一折验证集的
实际上测试集测试的应该是最好模型的结果
加入数据库接口
能够输出保存训练过程中的特征向量
"""


def rightness(prediction, labels):
    """计算预测准确率函数"""
    pre = torch.max(prediction.data, 1)[1]
    rights = pre.eq(labels.data).sum()
    return rights / len(labels)


def z_score(data, mean, std):
    """对输入数据进行z-score归一化，所选维度为timestep * N"""
    s = data.shape
    data = data.reshape([s[0] * s[1], s[2]])
    data = (data - mean) / std
    return data.reshape(s)


def init_lstm(weight, m):
    """初始化LSTM层"""
    for w in weight.chunk(4, 0):
        nn.init.normal_(w, mean=0, std=np.sqrt(2 / m.input_size))


def init_model(mod):
    """初始化模型参数函数"""
    for m in mod.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=np.sqrt(2 / m.in_features))
        if isinstance(m, nn.LSTM):
            init_lstm(m.weight_ih_l0, m)
            init_lstm(m.weight_hh_l0, m)


DATA_BASE = 'iemocap'  # 数据库接口
if DATA_BASE == 'emodb':
    CLASS_NUM = 7
    BATCH_SIZE = 20
else:
    CLASS_NUM = 4
    BATCH_SIZE = 256
EPOCH = 100
# BATCH_SIZE = 256  # 数据本身就不多，搞了256个是不是太大？是的
LR = 0.01
is_use_gpu = False

torch.manual_seed(1)  # 随机种子
np.random.seed(0)

# (535, 60, 39) (data_length, time_step, MFCCs_size)
X_or = np.load('data/'+DATA_BASE+'_data_set.npy')
# (535, 7)
Y_or = np.load('data/'+DATA_BASE+'_label_set.npy')
# (10, 2)
record_list = np.load('data/'+DATA_BASE+'_record_list.npy')
data_len = len(X_or)  # 数据数目

button = False  # 一折测试按钮

# 外部容器
test_acc = []  # 所有折准确率
best_epoches = []  # 每一折最佳epoch合集
wa_s = []  # 每折UA值
# 整体逻辑
for session in range(5):
    # 将数据集分为第一个人、第二个人和其他人三部分
    person1_X = X_or[record_list[session * 2][0]:record_list[session * 2][1] + 1]
    person1_Y = Y_or[record_list[session * 2][0]:record_list[session * 2][1] + 1]
    person2_X = X_or[record_list[session * 2 + 1][0]:record_list[session * 2 + 1][1] + 1]
    person2_Y = Y_or[record_list[session * 2 + 1][0]:record_list[session * 2 + 1][1] + 1]
    other_X = np.delete(X_or, [i for i in range(record_list[session * 2][0], record_list[session * 2 + 1][1] + 1)], 0)
    other_Y = np.delete(Y_or, [i for i in range(record_list[session * 2][0], record_list[session * 2 + 1][1] + 1)], 0)
    # 每个session进行两次测试、验证，交换
    for turn in range(2):
        print('正在进行第{}折----------------------------------'.format(session * 2 + turn + 1))
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
            test_Y = person1_Y  # 绘制混淆矩阵时候用
        else:
            train_ds = Data.TensorDataset(torch.FloatTensor(other_X), torch.LongTensor(other_Y))
            valid_ds = Data.TensorDataset(torch.FloatTensor(person1_X), torch.LongTensor(person1_Y))
            test_ds = Data.TensorDataset(torch.FloatTensor(person2_X), torch.LongTensor(person2_Y))
            test_Y = person2_Y

        # 构建Data Loader
        train_loader = Data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = Data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = Data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        # 模型构建和初始化
        lstm = LSTMNet(39)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lstm.parameters(), lr=LR)  # 这里还要完成一个LR的自动削减？
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 自动变化器，每10epoch下降1%
        init_model(lstm)
        # 将网络放在gpu上
        if is_use_gpu:
            lstm = lstm.cuda()
            criterion = criterion.cuda()
        # 存储容器
        losses = []  # 本折loss
        valid_acc = []  # 验证集在每一折所有epoch的准确率，不需要每个epoch都有，按照下面interval进行
        val_losses = []  # 外部容器，验证集每次验证的损失
        interval = 5  # 每5个epoch进行一次验证
        max_acc = 0  # 最大准确率，用于存储验证集上的最优模型参数，防止过拟合
        min_loss = np.Inf  # 验证集上最小损失
        min_epoch = 0  # 验证集最小损失的epoch值
        max_epoch = 0  # 最佳模型对应的epoch值
        test_results = []  # 存放多个batch的测试集结果，最后取平均值

        # 开始训练
        for epoch in range(EPOCH):
            # 训练集上训练
            loss_tmp = []  # 临时容器
            for batch, data in enumerate(train_loader):  # 加载训练数据
                lstm.train()  # 做标记，dropout启动
                init_hidden1 = lstm.init_Hidden(len(data[0]), 1, 512, if_use_gpu=is_use_gpu)
                init_hidden2 = lstm.init_Hidden(len(data[0]), 1, 256, if_use_gpu=is_use_gpu)
                optimizer.zero_grad()  # 梯度归零
                x, y = Variable(data[0]), Variable(data[1])

                if is_use_gpu:
                    x = x.cuda()
                    y = y.cuda()

                outputs = lstm(x, init_hidden1, init_hidden2)
                loss = criterion(outputs[0], y)
                loss_tmp.append(loss.data.numpy())  # 这里是每个batch有一个loss，加入临时容器
                loss.backward()
                optimizer.step()
                # scheduler.step()

            losses.append(np.mean(loss_tmp))

            # 验证集上验证
            val_loss = []  # 临时容器
            if epoch % interval == 0:  # 间隔几个epoch验证一次
                val_acc = []  # 验证集准确率临时容器
                for batch, data in enumerate(valid_loader):
                    lstm.eval()  # 关闭开关，用于测试
                    init_hidden1 = lstm.init_Hidden(len(data[0]), 1, 512)
                    init_hidden2 = lstm.init_Hidden(len(data[0]), 1, 256)
                    x, y = Variable(data[0]), Variable(data[1])
                    outputs = lstm(x, init_hidden1, init_hidden2)
                    loss = criterion(outputs[0], y)  # 计算验证集的loss
                    val_loss.append(loss.data.numpy())  # 将loss添加到临时容器
                    rate = rightness(outputs[0], y)  # 验证集上的准确率
                    val_acc.append(rate)  # 存储到临时容器中
                valid_acc.append(np.mean(val_acc))  # 将临时容器的准确率存储到全局容器中
                val_losses.append(np.mean(val_loss))  # 将临时容器的loss存储到全局容器中
                print('EPOCH{}\tLOSS:{:.6f}\tValid_Acc:{:.2f}\tValid_Loss:{:.2f}'.format(epoch, losses[-1],
                                                                                         valid_acc[-1], val_losses[-1]))
                # if valid_acc[-1] > max_acc:  # 如果准确率更大
                #     max_acc = valid_acc[-1]
                #     max_epoch = epoch
                #     torch.save(lstm.state_dict(), 'network{}.pt'.format(session * 2 + turn + 1))  # 存储并覆盖模型参数
                if val_losses[-1] < min_loss:  # 如果小于验证集最小loss
                    min_loss = val_losses[-1]
                    min_epoch = epoch
                    torch.save(lstm.state_dict(), (DATA_BASE + '_model/network{}.pt').format(session * 2 + turn + 1))  #
                    # 存储并覆盖模型参数

        # 测试集测试准确率
        print('第{}折最优模型的epoch：{}'.format(session * 2 + turn + 1, min_epoch))
        lstm_dict = torch.load((DATA_BASE + '_model/network{}.pt').format(session * 2 + turn + 1))  # 这里加载的是最好模型的结果
        lstm = LSTMNet(39)
        lstm.load_state_dict(lstm_dict)
        all_out = []  # 存放所有模型的输出
        lstm.eval()  # 关闭开关，用于测试

        for batch, data in enumerate(test_loader):
            init_hidden1 = lstm.init_Hidden(len(data[0]), 1, 512)
            init_hidden2 = lstm.init_Hidden(len(data[0]), 1, 256)
            x, y = Variable(data[0]), Variable(data[1])
            outputs = lstm(x, init_hidden1, init_hidden2)
            all_out.extend(np.argmax(outputs[0].detach().numpy(), 1))
            rate = rightness(outputs[0], y)
            test_results.append(rate)

        # 计算WA的值
        WA = 0
        for c in range(CLASS_NUM):
            ind = np.where(test_Y == c)  # 获得所有该类的下标
            num = len(ind[0])  # 获得该类元素个数
            corr = np.sum(np.equal(test_Y[ind], np.array(all_out)[ind]))  # 统计该类判断正确个数
            WA += corr / (CLASS_NUM * num)
        wa_s.append(WA)

        # 绘制混淆矩阵
        if DATA_BASE == 'emodb':
            db_name = 'EMO-DB'
        else:
            db_name = 'IEMOCAP'
        m = confusion_matrix(test_Y, all_out)
        plot_confusion_matrix(m, DATA_BASE + '_results/matrix_fold{}.png'.format(session * 2 + turn + 1), title='Confusion Matrix '
                                                                                                   'Of' + db_name,
                              data_base=DATA_BASE)

        test_mean = np.mean(test_results)  # 计算本折测试集准确率
        test_acc.append(test_mean)  # 添加结果到全局准确率容器
        print('第{}折测试集准确率： UA {:.2f}  |  WA {:.2F}'.format(session * 2 + turn + 1, test_mean, WA))

        # 绘图区
        plt.figure()  # 图1，每折的loss曲线
        plt.plot([i for i in range(len(losses))], losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss of Fold {}'.format(session * 2 + turn + 1))
        plt.savefig(DATA_BASE + '_results/loss_of_fold{}.png'.format(session * 2 + turn + 1), format='png')

        plt.figure()  # 图2，每折验证集准确率曲线
        plt.plot([i * interval for i in range(len(valid_acc))], valid_acc)
        plt.xlabel('Epochs')
        plt.ylabel('Validation Set Accuracy')
        plt.title('Validation Set Accuracy of Fold {}'.format(session * 2 + turn + 1))
        plt.savefig(DATA_BASE + '_results/validation_acc_of_fold{}.png'.format(session * 2 + turn + 1), format='png')

        plt.figure()  # 图3，每折验证集loss曲线
        plt.plot([i * interval for i in range(len(val_losses))], val_losses)
        plt.xlabel('Epochs')
        plt.ylabel('Validation Set Loss')
        plt.title('Validation Set Loss of Fold {}'.format(session * 2 + turn + 1))
        plt.savefig(DATA_BASE + '_results/validation_loss_of_fold{}.png'.format(session * 2 + turn + 1), format='png')

        plt.show()

        if button is True:
            break

    if button is True:
        break  # 一折开关开启

if button is False:
    print('十折交叉验证综合结果：')
    print('平均UA值：{:.2f}'.format(np.mean(test_acc)))
    print('平均WA值：{:.2f}'.format(np.mean(wa_s)))
