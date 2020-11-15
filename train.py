import datetime
import os
import sys

from tensorflow.python.keras import losses, optimizers, metrics

from CNN import callback_list, MLP_protein, tianjin_LSTM, CNN_16kernal, tianjin_a_use_activation
from draw_image import *
from read_protein import merged_data, get_onehot_gen, get_number_gen, merged_data_arr, shuffle_data
import numpy as np
from sklearn.metrics import roc_curve, auc
from custom_metrics import *
import time
from math import floor
from collections import Counter
from imblearn.ensemble import EasyEnsembleClassifier
import _pickle as cPickle

from util import create_dir, config_log, close_log, time_spent



def train_model(model_func, save_dir):
    """
    加载数据，训练模型

    :param model_func:
    :return:
    """

    #
    # 配置参数
    #
    k_fold = 10  # 多少折交叉验证
    one_fold_len = 1000  # 一折多少条数据
    batch_size = 16
    every_fold_epochs = 10  # 每一折跑多少个循环
    loss, loss_name = (losses.binary_crossentropy, "binary_crossentropy")
    opt, opt_name = (optimizers.Adam(), "Adam")
    val_met = ['acc', Precision, Recall, F1, G_mean]
    print("loss:", loss_name)
    print("optimizer:", opt_name)

    #
    # 获取数据
    # x 为蛋白质序列列表
    # y 为标签，ndarray
    #
    x, y = merged_data(num=k_fold*one_fold_len)   # 获取数据
    # x, y = merged_data("./data/protein_sequence/equal/positive_fixed.txt",
    #                    "./data/protein_sequence/equal/negative_fixed.txt")
    # x, y = merged_data_arr()
    if y.shape[0] < k_fold * one_fold_len:      # 有可能数据没那么多
        one_fold_len = floor(y.shape[0] / k_fold)
    # x, y = shuffle_data(x, y)
    # x = x[:1000]
    # y = y[:1000]
    # print("原始数据：", Counter(y))

    #
    # 欠采样
    #
    # ee = EasyEnsembleClassifier(random_state=0)
    # x_resampled, y_resampled = ee.fit(x, y)
    # print(sorted(Counter(y_resampled[0]).items()))
    # return

    #
    # 过采样
    #



    #
    # 交叉验证
    #
    train_history = {}  # 用来保存所有交叉验证最后的结果
    val_history = {}
    auc_history = []
    for i in range(k_fold):
        #
        # 数据分块
        #
        start = i * one_fold_len
        end = (i + 1) * one_fold_len
        x_train = x[:start] + x[end:]
        y_train = np.concatenate((y[:start], y[end:]))

        x_val = x[start:end]
        y_val = y[start:end]  # 是一个 ndarray 数组

        available_train_steps = int(y_train.shape[0] / batch_size)
        available_val_steps = int(y_val.shape[0] / batch_size)

        # 真实用的数据量，有可能除不尽 batch_size，多了那么一丢丢数据
        x_train = x_train[:available_train_steps * batch_size]
        y_train = y_train[:available_train_steps * batch_size]

        x_val = x_val[:available_val_steps * batch_size]
        y_val = y_val[:available_val_steps * batch_size]

        real_train_num = len(y_train)
        real_val_num = len(y_val)

        #
        # 数据生成器
        #
        # one-hot 格式
        train_gen = get_onehot_gen(x_train, y_train, batch_size, available_train_steps)
        val_gen = get_onehot_gen(x_val, y_val, batch_size, available_val_steps)
        roc_gen = get_onehot_gen(x_val, y_val, batch_size, available_val_steps)
        # 数字矩阵格式
        # train_gen = get_number_gen(x_train, y_train, batch_size)
        # val_gen = get_number_gen(x_val, y_val, batch_size)

        #
        # 获取一个新的模型，从头开始训练
        #
        model, model_name = model_func()
        model.compile(loss=loss, optimizer=opt, metrics=val_met)

        #
        # 开始训练
        #
        print("%d/%d 开始训练 batch_size=%d 训练数据共%d 评估数据共%d" % (i + 1, k_fold, batch_size, real_train_num, real_val_num))
        start_time = time.time()
        print("开始训练时间：%s" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        history = model.fit_generator(train_gen, available_train_steps, epochs=every_fold_epochs, verbose=1,
                                           callbacks=callback_list(save_dir, cur_k_fold=i + 1),
                                      validation_data=val_gen, validation_steps=available_val_steps)
        end_time = time.time()
        print("结束训练时间：%s" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
        print("训练用时：%s" % time_spent(int(end_time-start_time)))    # 打印出共花费多少时间

        #
        # 取出本次训练的最终训练结果数据
        #
        every_train_result = {}
        for k, v in history.history.items():
            every_train_result[k] = v[-1]
        # 把训练结果和评估结果分开保存
        for k in every_train_result.keys():
            if k.find("val_") == -1:
                if k not in train_history.keys():
                    # 首次添加到字典中
                    train_history[k] = [every_train_result[k]]
                else:
                    train_history[k] += [every_train_result[k]]
            else:
                if k not in val_history.keys():
                    val_history[k] = [every_train_result[k]]
                else:
                    val_history[k] += [every_train_result[k]]
        # 绘制每折交叉验证的 acc 折线图
        show_train_val(history.history['acc'], history.history['val_acc'], save_dir+"/%d_train_val_acc.png" % (i+1))


        #
        # 计算 ROC
        #
        y_score = model.predict_generator(roc_gen, available_val_steps, verbose=1)
        auc_history += [calc_auc(y_score, y_val, i)]

    #
    # 统计最终结果
    #
    print("训练结果：", train_history)
    print("评估结果：", val_history)
    print("auc:", auc_history)
    show_every_fold_acc(train_history['acc'], val_history['val_acc'], save_dir + "/every_fold.png")
    # 输出总结一下，方便复制到表格中
    summary(model_name, loss_name, batch_size, opt_name, real_train_num, real_val_num,
            k_fold*every_fold_epochs, k_fold, auc_history, train_history, val_history)
    # 给日志文件夹重命名为人方便看的文件名
    rename_dir(save_dir, model_name, real_train_num, np.mean(train_history['acc']))

def calc_auc(y_score, y_val, i):
    """

    :param y_score:
    :param y_val:
    :param i: kFold 编号
    :return:
    """
    # ROC 数据（必要要得到 y_score 才能计算 roc）
    y_score = np.reshape(y_score, (y_score.shape[0],))  # 返回的数据是 (x,1)，多了一个维度
    fpr, tpr, thresholds = roc_curve(y_val, y_score)
    auc_value = auc(fpr, tpr)
    print("auc:", auc_value)
    roc_data = [fpr, tpr, thresholds, auc_value]
    # 保存 ROC 数据和图片
    np.save(save_dir + "/%d_fpr_tpr_thresholds_auc.npy" % (i + 1), roc_data)  # 保存 ROC 数据
    img_path = save_dir + "/%d_roc.jpg" % (i + 1)
    show_ROC(fpr, tpr, auc_value, img_path)

    return auc_value

def summary(model_name, loss_name, batch_size, optimizer_name, train_size, val_size, epochs, kFold, auc_history, train_result, val_result):
    """
    总结一下训练过程

    """
    out_str = (model_name+"\t"+loss_name+"\t"+str(batch_size)+"\t"+optimizer_name+"\t"+
          str(train_size)+"\t"+str(val_size)+"\t"+str(epochs)+"\t"+str(kFold)+"\t")

    out_str += "%.4f\t" % np.mean(auc_history)

    for v in train_result.values():
        out_str += "%.4f\t" % np.mean(v)
    # 验证评估分数是计算所有验证评估分数的平均值
    for v in val_result.values():
        out_str += "%.4f\t" % np.mean(v)
    print("( •̀ ω •́ )y 交叉验证统计结果，所有交叉验证的平均值：")
    print(out_str)

def rename_dir(save_dir, model_name, data_size, acc):
    """
    重命名文件见，方便人能看懂

    :param save_dir:
    :param model_name:
    :param data_size:
    :param acc:
    :return:
    """
    #
    # 获取当前文件夹的上一级目录
    #
    upper_dir = os.path.abspath(os.path.join(save_dir, ".."))
    new_save_dir = os.path.join(upper_dir, model_name + "_data-%d"%data_size + "_acc-%.2f"%acc)

    #
    # 预防重名
    #
    t = new_save_dir
    i = 0
    while os.path.exists(t):
        i += 1
        t = new_save_dir + "_%d" % i
    if i != 0:
        new_save_dir = t

    #
    # 重命名
    #
    close_log() # 必须要先关闭日志文件
    os.rename(save_dir, new_save_dir)   # 对根目录重命名


if __name__ == '__main__':
    #
    # 开始训练前的一些准备工作
    #
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"    # 不显示提示信息
    save_dir = create_dir()                     # 创建一个保存目录
    config_log(save_dir)                        # 配置日志文件

    #
    # 开始训练
    #
    #model, model_name = MLP_protein()    # 获取模型
    train_model(tianjin_a_use_activation, save_dir)          # 训练模型