"""
画热力图等
"""

from tensorflow.keras import models
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# 氨基酸字母
amino_acid = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def show_every_fold_acc(train_acc, val_acc, save_path):
    """
    画交叉验证结果

    :param train_acc:
    :param val_acc:
    :param save_path:
    :return:
    """
    plt.figure()
    plt.plot(range(1, len(train_acc) + 1), train_acc, 'bo-', label='train_acc')
    plt.plot(range(1, len(val_acc) + 1), val_acc, 'r+-', label='val_acc')
    plt.xlabel("kFold")
    plt.ylabel("acc")
    plt.legend(loc='best')
    #plt.title("train_acc VS val_acc")
    plt.savefig(save_path)
    plt.close()

def show_train_val(train_acc, val_acc, save_path):
    """
    训练集和验证集 acc 折线图

    :param train_acc:
    :param val_acc:
    :param save_path:
    :return:
    """
    plt.figure()
    plt.plot(range(1, len(train_acc) + 1), train_acc, 'bo-', label='train_acc')
    plt.plot(range(1, len(val_acc) + 1), val_acc, 'r+-', label='val_acc')
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend(loc='best')
    plt.title("train_acc VS val_acc")
    plt.savefig(save_path)
    plt.close()


def show_loss(loss, save_path):
    """
    损失值图像
    :param loss:
    :param save_path:
    :return:
    """
    plt.figure()
    plt.plot(range(1, len(loss) + 1), loss, label='loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("loss curve")
    plt.savefig(save_path)
    plt.close()

def show_acc(acc, save_path):
    """
    绘制精确度曲线

    :param acc:
    :param save_path:

    :return:
    """
    plt.figure()
    plt.plot(range(1, len(acc)+1), acc, label='acc')
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.title("acc curve")
    plt.savefig(save_path)
    plt.close()



def show_ROC(fpr, tpr, auc, save_path):
    """
    绘制 ROC 曲线

    :param fpr:
    :param tpr:
    :param auc:
    :param save_path:
    :return:
    """
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--') # 画中间那条对角线
    plt.plot(fpr, tpr, label='AUC={:.3f}'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(save_path)
    #plt.show()
    plt.close()

def specific_dis(dis_data):
    """
    特异性距离

    :param dis_data: 欧式距离矩阵
    :return: 特异性距离数组
    """
    s = np.zeros((20,))

    for i in range(20):
        for j in range(20):
            if i != j:
                s[i] += dis_data[i][j]

    return s

def show_bar(specific_data):
    """
    绘制特异性距离条形图

    :param specific_dis: 特异性距离
    """
    labels = amino_acid
    plt.bar(x=labels, height=specific_data)

    plt.show()

def calc_dis(j_1, j_2, kernal:np.ndarray):
    """
    求欧式距离之和

    :param j_1: 氨基酸编号
    :param j_2: 另一个氨基酸编号
    :param kernal: 加权求和卷积核
    :return:欧式距离之和
    """
    dis = 0
    for i in range(kernal.shape[0]):
        dis += pow(kernal[i][j_1] - kernal[i][j_2], 2)
    dis = dis ** 0.5

    return dis

def get_distance(kernal:np.ndarray):
    """
    氨基酸欧式距离

    :param kernal:加权求和卷积核
    :return:氨基酸欧式距离矩阵
    """
    dis_data = np.zeros((20, 20))
    for i in range(20):
        for j in range(i):
            dis_data[i][j] = dis_data[j][i] = calc_dis(i, j, kernal)

    return dis_data


def show_distance(dis_data:np.ndarray):
    """
    显示氨基酸欧式距离热图

    :param data: 氨基酸欧式距离矩阵
    """
    f, ax = plt.subplots(figsize=(15, 12))
    #
    # 必须先绘制热图，再设置 x 轴标签
    #
    sns.heatmap(dis_data, annot=True, fmt='.1f', ax=ax)
    ax.yaxis.set_ticks_position("left")
    ax.set_yticklabels(amino_acid)
    ax.xaxis.set_ticks_position("top")
    ax.set_xticklabels(amino_acid)
    f.show()



def show_heatmap(kernal:np.ndarray):
    """
    根据给定的数据，显示热图

    :param kernal:加权求和卷积核
    """
    f, ax = plt.subplots(figsize=(12, 6))

    #
    # 必须先绘制热图，再设置 x 轴标签
    #
    sns.heatmap(kernal, annot=True, fmt='.1f', ax=ax)

    ax.xaxis.set_ticks_position("top")
    ax.set_xticklabels(amino_acid)
    f.show()


def kernal_weight(model_path=None):
    """
    获取模型第一层卷积层卷积核的权重

    :param model_path: 模型路径
    :return: 权重
    """
    if model_path is None:
        model_path = "./model/cnn_model_acc_0.85_val_acc_0.64.hdf5"

    #
    # 加载模型
    #
    model = models.load_model(model_path)
    model.summary()

    #
    # 获取模型的每一层权重
    #
    weights = model.get_weights()

    kernal_weight = weights[0]      # (11,20,64),channel_last 格式的
    dense_weight = weights[2]       # (64,1)
    dense_weight = dense_weight.reshape(64, )

    #
    # 加权求和
    #
    k = 0
    data = np.empty((kernal_weight.shape[0], kernal_weight.shape[1]), dtype=float)
    for i in range(kernal_weight.shape[0]):
        for j in range(kernal_weight.shape[1]):
            data[i][j] = np.multiply(kernal_weight[i][j], dense_weight).sum()
            k += 1

    return data

if __name__ == '__main__':
    show_acc([0.5,0.6,0.7,0.8,0.9,0.92,0.95,0.97,0.99], "./acc.png")


