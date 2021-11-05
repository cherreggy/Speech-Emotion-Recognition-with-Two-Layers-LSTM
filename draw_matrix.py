import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, savename, title='Confusion Matrix', data_base = None):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    if data_base == 'emodb':
        classes = ['Ang', 'Bor', 'Dis', 'Anx', 'Hap', 'Sad', 'Neu']
    else:
        classes = ['Ang', 'Hap', 'Neu', 'Sad']

    # 在混淆矩阵中每格的概率值
    # ind_array = np.arange(len(classes))
    # x, y = np.meshgrid(ind_array, ind_array)
    # for x_val, y_val in zip(x.flatten(), y.flatten()):
    #     c = cm[y_val][x_val]
    #     if c > 0.001:
    #         plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    for i in range(len(classes)):
        sum_x = np.sum(cm[i])
        for j in range(len(classes)):
            c = cm[i][j] / sum_x
            plt.text(j, i, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()