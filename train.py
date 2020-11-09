import os
import sys

from tensorflow.python.keras import losses, optimizers, metrics

from CNN import callback_list, MLP_protein
from draw_image import *
from read_protein import merged_data, get_gen
import numpy as np
from sklearn.metrics import roc_curve, auc
from custom_metrics import *
import time
from math import floor

from util import create_dir, config_log, close_log


def get_loss(loss):
    if loss == losses.binary_crossentropy:
        return loss, "binary_crossentropy"
    elif loss == losses.mean_squared_error:
        return loss, "mean_squared_error"

def get_optimizer(optimizer):
    if optimizer == optimizers.Adam:
        return optimizers.Adam(), "Adam()"


def train_model(model, model_name, save_dir):
    """
    真正的训练
    加载数据，训练模型

    :param model: 创建好的模型
    :return:
    """
    #
    # 获取数据
    #
    k_fold = 5              # 多少折交叉验证
    one_fold_len = 1000     # 一折多少条数据
    x, y = merged_data(k_fold * one_fold_len)   # 获取数据
    if y.shape[0] < k_fold * one_fold_len:      # 有可能数据没那么多
        one_fold_len = floor(y.shape[0] / k_fold)

    #
    # 配置参数
    #
    batch_size = 16
    every_flod_epochs = 1  # 每一折跑多少个循环
    loss, loss_name = get_loss(losses.mean_squared_error)
    opt, opt_name = get_optimizer(optimizers.Adam)
    val_met = ['acc', Precision, Recall, F1, G_mean]
    model.compile(loss=loss, optimizer=opt, metrics=val_met)
    # 打印输出
    print("model name:", model_name)
    print("loss:", loss_name)
    print("optimizer:", opt_name)

    #
    # 交叉验证
    #
    initial_epoch = 0  # 开始 epoch 的编号
    train_history = None    # 记录训练指标数据
    val_history = None
    for i in range(k_fold):
        # 数据分块
        start = i * one_fold_len
        end = (i + 1) * one_fold_len
        x_train = x[:start] + x[end:]
        y_train = np.concatenate((y[:start], y[end:]))
        x_val = x[start:end]
        y_val = y[start:end]  # 是一个 ndarray 数组
        train_steps = floor(y_train.shape[0] / batch_size)
        val_steps = floor(y_val.shape[0] / batch_size)
        # 真实用的数据量，有可能除不尽 batch_size，多了那么一丢丢数据
        x_train = x_train[:train_steps * batch_size]
        y_train = y_train[:train_steps * batch_size]
        x_val = x_val[:val_steps * batch_size]
        y_val = y_val[:val_steps * batch_size]
        real_train_num = len(y_train)
        real_val_num = len(y_val)
        # 数据生成器
        train_gen = get_gen(x_train, y_train, batch_size)
        val_gen = get_gen(x_val, y_val, batch_size)

        #
        # 开始训练
        #
        print("%d/%d开始训练 batch_size=%d 训练数据共%d" % (i + 1, k_fold, batch_size, real_train_num))
        start_time = time.time()
        print("当前时间：%s" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        history = model.fit_generator(train_gen, train_steps, epochs=(i + 1) * every_flod_epochs, verbose=1,
                                      callbacks=callback_list(save_dir, cur_k_fold=i + 1), initial_epoch=initial_epoch)
        end_time = time.time()
        print("完成训练时间：%s" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
        # 记录相关指标
        if train_history is None:
            train_history = history.history
        else:
            for k, v in history.history.items():
                train_history[k] += v
        initial_epoch += every_flod_epochs  # 下次训练从这个编号开始记录

        #
        # 评估模型
        #
        print("%d/%d开始评估 batch_size=%d 评估数据共%d" % (i+1, k_fold, batch_size, real_val_num))
        y_score = model.predict_generator(val_gen, val_steps)
        # ROC 数据（必要要得到 y_score 才能计算 roc）
        y_score = np.reshape(y_score, (y_score.shape[0],))  # 返回的数据是 (x,1)，多了一个维度
        fpr, tpr, thresholds = roc_curve(y_val, y_score)
        auc_value = auc(fpr, tpr)
        roc_data = [fpr, tpr, thresholds, auc_value]
        # 保存 ROC 数据和图片
        np.save(save_dir + "/%d_fpr_tpr_thresholds_auc.npy" % (i+1), roc_data)  # 保存 ROC 数据
        img_path = save_dir + "/%d_roc.jpg" % (i+1)
        show_ROC(fpr, tpr, auc_value, img_path)
        # 计算相关评估指标
        with tf.Session() as sess:
            y_val = tf.cast(y_val, tf.float32)
            y_score = tf.cast(y_score, tf.float32)
            val_met = {}
            val_met['loss'] = [losses.binary_crossentropy(y_val, y_score).eval()]  # 传进去的都必须是 tensor
            val_met['acc'] = [metrics.binary_accuracy(y_val, y_score).eval()]
            val_met['Precision'] = [Precision(y_val, y_score).eval()]
            val_met['Recall'] = [Recall(y_val, y_score).eval()]
            val_met['F1'] = [F1(y_val, y_score).eval()]
            val_met['G_mean'] = [G_mean(y_val, y_score).eval()]
            print("评估结果:", val_met)
            if val_history is None:
                val_history = val_met
            else:
                for k, v in val_met.items():
                    val_history[k] += v


    #
    # 绘制训练 acc,loss 图等，善后
    #
    print("train_history:", train_history)
    print("val_history:", val_history)
    #show_acc(train_history['acc'], save_dir+"/acc.png")
    #show_loss(train_history['loss'], save_dir+"/loss.png")
    show_train_val(train_history['acc'], val_history['acc'], save_dir+"/train_val_acc.png")
    np.save(save_dir+"/train_history.npy", train_history)
    np.save(save_dir+"/val_history.npy", val_history)
    # 输出总结一下，方便复制到表格中
    summary(model_name, loss_name, batch_size, opt_name, real_train_num, real_val_num,
            k_fold*every_flod_epochs, k_fold, train_history, val_history)
    # 给日志文件夹重命名为人方便看的文件名
    rename_dir(save_dir, model_name, real_train_num, train_history['acc'][k_fold*every_flod_epochs-1])

def summary(model_name, loss_name, batch_size, optimizer_name, train_size, val_size, epochs, kFold, train_data, val_data):
    """
    总结一下训练过程

    :param model_name:
    :param loss_name:
    :param batch_size:
    :param optimizer_name:
    :param train_size:
    :param val_size:
    :param epochs:
    :param kFold:
    :param train_data:
    :param val_data:
    :return:
    """
    out_str = (model_name+"\t"+loss_name+"\t"+str(batch_size)+"\t"+optimizer_name+"\t"+
          str(train_size)+"\t"+str(val_size)+"\t"+str(epochs)+"\t"+str(kFold)+"\t")
    # 训练评估分数是最后一个 epoch 的最后一个 batch 计算得出的
    for k, v in train_data.items():
        out_str += "%.4f\t" % v[epochs-1]
    # 验证评估分数应该是计算所有验证评估分数的平均值
    for k, v in val_data.items():
        mean = np.mean(v)   # 求平均值
        out_str += "%.4f\t" % mean

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
    model, model_name = MLP_protein()    # 获取模型
    train_model(model, model_name, save_dir)          # 训练模型