import datetime
import os
import sys
from collections import Counter

from tensorflow.python.keras import losses, optimizers, metrics

from CNN import callback_list, MLP_protein, tianjin_LSTM, CNN_16kernal, tianjin_a_use_activation
from draw_image import *
from read_protein import merged_data, get_onehot_gen, get_number_gen, merged_data_arr, shuffle_data, get_pkl_data, \
    merged_pkl_data, merged_pkl_data_2
import numpy as np
from sklearn.metrics import roc_curve, auc
from tf_custom_metrics import *
import time
from math import floor
from sklearn.model_selection import KFold
from sklearn import metrics

# 抽样方法
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 集成分类器
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier, BalancedRandomForestClassifier, \
    RUSBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# 简单分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from util import create_dir, config_log, close_log, time_spent


def extract_train_result(train_result, train_history, val_history):
    """
    提取最后一个epoch训练结果，单独保存在训练和验证字典中

    :param train_result:
    :param train_history:
    :param val_history:
    :return:
    """
    every_train_result = {}
    for k, v in train_result.items():
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
    return train_history, val_history


def get_kFold_data(i, one_fold_len, x, y, batch_size):
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

    # real_train_num = len(y_train)
    # real_val_num = len(y_val)

    return x_train, y_train, x_val, y_val, available_train_steps, available_val_steps

def train_SMOTE_KNN(log_dir, x, y):
    print("model_name：SMOTE_KNN")
    val_history = {}
    val_history["val_acc"] = []
    val_history["val_precision"] = []
    val_history["val_recall"] = []
    val_history["val_f1"] = []
    val_history["auc_value"] = []

    kf = KFold(n_splits=10, shuffle=True)
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        cur_k += 1
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        print("抽样前：", Counter(y_train))
        sm = SMOTE(random_state=42, n_jobs=-1)
        x_train, y_train = sm.fit_resample(x_train, y_train)
        print("抽样后：", Counter(y_train))

        classifier = KNeighborsClassifier(n_jobs=-1)

        # 训练模型
        classifier.fit(x_train, y_train)

        # 预测
        y_pred = classifier.predict(x_val)
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)

        y_pred = classifier.predict_proba(x_val)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_pred)

        val_history["val_acc"].append(val_acc)
        val_history["val_precision"].append(val_precision)
        val_history["val_recall"].append(val_recall)
        val_history["val_f1"].append(val_f1)
        val_history["auc_value"].append(auc_value)

        show_ROC(fpr, tpr, auc_value, log_dir + "/%d_roc.png" % cur_k)

        print(cur_k)
        print("val_acc: %.4f" % val_acc)
        print("val_precision: %.4f" % val_precision)
        print("val_recall: %.4f" % val_recall)
        print("val_f1: %.4f" % val_f1)
        print("auc_value: %.4f" % auc_value)

    show_acc(val_history["val_acc"], log_dir + "/val_acc.png")
    print(val_history)

    # 统计，求平均值和标准差
    for k in val_history.keys():
        print("%s:%.4f ±%.4f" % (k, np.mean(val_history[k]), np.std(val_history[k])))

    rename_log_dir(log_dir, classifier_name, len(y_train), np.mean(val_history['val_acc']))

def train_Classifier_2(classifier, classifier_name, log_dir, x, y, sampling_method):
    print("classifier_name:", classifier_name)

    val_history = {}
    val_history["val_acc"] = []
    val_history["val_precision"] = []
    val_history["val_recall"] = []
    val_history["val_f1"] = []
    val_history["auc_value"] = []

    kf = KFold(n_splits=10, shuffle=True)
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        cur_k += 1
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        # sm = SMOTE(random_state=42)
        # x_train, y_train = sm.fit_resample(x_train, y_train)

        # rus = RandomUnderSampler(random_state=42)
        # x_train, y_train = rus.fit_resample(x_train, y_train)

        print("抽样前：", Counter(y_train))
        sampler = sampling_method(random_state=42)
        x_train, y_train = sampler.fit_resample(x_train, y_train)
        print("抽样后：", Counter(y_train))

        # 训练模型
        classifier.fit(x_train, y_train)

        # 预测
        y_pred = classifier.predict(x_val)
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)

        y_pred = classifier.predict_proba(x_val)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_pred)

        val_history["val_acc"].append(val_acc)
        val_history["val_precision"].append(val_precision)
        val_history["val_recall"].append(val_recall)
        val_history["val_f1"].append(val_f1)
        val_history["auc_value"].append(auc_value)

        show_ROC(fpr, tpr, auc_value, log_dir + "/%d_roc.png" % cur_k)

        print(cur_k)
        print("val_acc: %.4f" % val_acc)
        print("val_precision: %.4f" % val_precision)
        print("val_recall: %.4f" % val_recall)
        print("val_f1: %.4f" % val_f1)
        print("auc_value: %.4f" % auc_value)

    show_acc(val_history["val_acc"], log_dir + "/val_acc.png")
    print(val_history)

    # 统计，求平均值和标准差
    for k in val_history.keys():
        print("%s:%.4f ±%.4f" % (k, np.mean(val_history[k]), np.std(val_history[k])))

    rename_log_dir(log_dir, classifier_name, len(y_train), np.mean(val_history['val_acc']))


def train_Classifier(classifier, classifier_name, log_dir, x, y):
    print("classifier_name:", classifier_name)

    val_history = {}
    val_history["val_acc"] = []
    val_history["val_precision"] = []
    val_history["val_recall"] = []
    val_history["val_f1"] = []
    val_history["auc_value"] = []

    kf = KFold(n_splits=10, shuffle=True)
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        cur_k += 1
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        print(Counter(y_train))

        # 训练模型
        classifier.fit(x_train, y_train)

        # 预测
        y_pred = classifier.predict(x_val)
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)

        y_pred = classifier.predict_proba(x_val)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_pred)

        val_history["val_acc"].append(val_acc)
        val_history["val_precision"].append(val_precision)
        val_history["val_recall"].append(val_recall)
        val_history["val_f1"].append(val_f1)
        val_history["auc_value"].append(auc_value)

        show_ROC(fpr, tpr, auc_value, log_dir + "/%d_roc.png" % cur_k)

        print(cur_k)
        print("val_acc: %.4f" % val_acc)
        print("val_precision: %.4f" % val_precision)
        print("val_recall: %.4f" % val_recall)
        print("val_f1: %.4f" % val_f1)
        print("auc_value: %.4f" % auc_value)

    show_acc(val_history["val_acc"], log_dir + "/val_acc.png")
    print(val_history)

    # 统计，求平均值和标准差
    for k in val_history.keys():
        print("%s:%.4f ±%.4f" % (k, np.mean(val_history[k]), np.std(val_history[k])))

    rename_log_dir(log_dir, classifier_name, len(y_train), np.mean(val_history['val_acc']))


def train_RandomForestClassifier(log_dir, x, y):
    model_name = "RandomForestClassifier"
    val_history = {}
    val_history["val_acc"] = []
    val_history["val_precision"] = []
    val_history["val_recall"] = []
    val_history["val_f1"] = []
    val_history["auc_value"] = []

    kf = KFold(n_splits=5, shuffle=True)
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        cur_k += 1
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        # 构建模型
        rfc = RandomForestClassifier(random_state=42, verbose=1)
        # 训练模型
        rfc.fit(x_train, y_train)
        # 预测
        y_pred = rfc.predict(x_val)
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)

        y_pred = rfc.predict_proba(x_val)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_pred)

        val_history["val_acc"].append(val_acc)
        val_history["val_precision"].append(val_precision)
        val_history["val_recall"].append(val_recall)
        val_history["val_f1"].append(val_f1)
        val_history["auc_value"].append(auc_value)

        show_ROC(fpr, tpr, auc_value, log_dir + "/%d_roc.png" % cur_k)

        print(cur_k)
        print("val_acc: %.4f" % val_acc)
        print("val_precision: %.4f" % val_precision)
        print("val_recall: %.4f" % val_recall)
        print("val_f1: %.4f" % val_f1)
        print("auc_value: %.4f" % auc_value)

    show_acc(val_history["val_acc"], log_dir + "/val_acc.png")
    print(val_history)

    # 统计，求平均值和标准差
    for k in val_history.keys():
        print("%s:%.4f ±%.4f" % (k, np.mean(val_history[k]), np.std(val_history[k])))

    rename_log_dir(log_dir, model_name, len(y_train), np.mean(val_history['val_acc']))


def train_EasyEnsembleClassifier(log_dir, x, y):
    """

    :param log_dir: 日志目录
    :param x: 数据
    :param y: 标签
    :return:
    """
    model_name = "EasyEnsembleClassifier"
    val_history = {}
    val_history["val_acc"] = []
    val_history["val_precision"] = []
    val_history["val_recall"] = []
    val_history["val_f1"] = []
    val_history["auc_value"] = []

    kf = KFold(n_splits=5, shuffle=True)
    cur_k = 0
    for train_index, val_index in kf.split(x, y):
        cur_k += 1
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        # 构建模型
        eec = EasyEnsembleClassifier(random_state=42, verbose=1)
        # 训练模型
        eec.fit(x_train, y_train)
        # 预测
        y_pred = eec.predict(x_val)
        val_acc = metrics.accuracy_score(y_val, y_pred)
        val_precision = metrics.precision_score(y_val, y_pred)
        val_recall = metrics.recall_score(y_val, y_pred)
        val_f1 = metrics.f1_score(y_val, y_pred)

        y_pred = eec.predict_proba(x_val)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred)
        auc_value = metrics.roc_auc_score(y_val, y_pred)

        val_history["val_acc"].append(val_acc)
        val_history["val_precision"].append(val_precision)
        val_history["val_recall"].append(val_recall)
        val_history["val_f1"].append(val_f1)
        val_history["auc_value"].append(auc_value)

        show_ROC(fpr, tpr, auc_value, log_dir + "/%d_roc.png" % cur_k)

        print(cur_k)
        print("val_acc: %.4f" % val_acc)
        print("val_precision: %.4f" % val_precision)
        print("val_recall: %.4f" % val_recall)
        print("val_f1: %.4f" % val_f1)
        print("auc_value: %.4f" % auc_value)

    show_acc(val_history["val_acc"], log_dir + "/val_acc.png")
    print(val_history)

    # 统计，求平均值和标准差
    for k in val_history.keys():
        print("%s:%.4f ±%.4f" % (k, np.mean(val_history[k]), np.std(val_history[k])))

    rename_log_dir(log_dir, model_name, len(y_train), np.mean(val_history['val_acc']))


def train_model(model_func, save_dir):
    """
    加载数据，训练模型

    :param model_func:
    :return:
    """

    #
    # 配置参数
    #
    k_fold = 5  # 多少折交叉验证
    one_fold_len = 1000  # 一折多少条数据
    batch_size = 16
    every_fold_epochs = 1  # 每一折跑多少个循环
    loss, loss_name = (losses.binary_crossentropy, "binary_crossentropy")
    opt, opt_name = (optimizers.Adam(), "Adam")
    val_met = ['acc', Precision, Recall, F1, G_mean]
    print("loss:", loss_name)
    print("optimizer:", opt_name)

    #
    # 获取数据
    #
    data_num = k_fold * one_fold_len
    x, y = merged_pkl_data()

    # if y.shape[0] < data_num:  # 有可能数据没那么多
    #     one_fold_len = int(y.shape[0] / k_fold)

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
    model_name = "model_name"
    real_train_num = 0
    real_val_num = 0
    kf = KFold(k_fold, True, 1116)
    k = 0
    for train_index, val_index in kf.split(x):
        k += 1
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        real_train_num = len(y_train)
        real_val_num = len(y_val)
        available_train_steps = int(real_train_num / batch_size)
        available_val_steps = int(real_val_num / batch_size)

        # for i in range(k_fold):
        #     # 获取训练数据
        #     x_train, y_train, x_val, y_val, available_train_steps, available_val_steps = get_kFold_data(i, one_fold_len, x, y, batch_size)
        #     real_train_num = len(y_train)
        #     real_val_num = len(y_val)
        #     #
        #     # 数据生成器
        #     #
        #     # one-hot 格式
        #     train_gen = get_onehot_gen(x_train, y_train, batch_size, available_train_steps)
        #     val_gen = get_onehot_gen(x_val, y_val, batch_size, available_val_steps)
        #     roc_gen = get_onehot_gen(x_val, y_val, batch_size, available_val_steps)
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
        print("%d/%d 开始训练 batch_size=%d 训练数据共%d 评估数据共%d" % (k, k_fold, batch_size, real_train_num, real_val_num))
        start_time = time.time()
        print("开始训练时间：%s" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
        history = model.fit(x_train, y_train, batch_size, every_fold_epochs,
                            callbacks=callback_list(save_dir, cur_k_fold=k),
                            validation_data=(x_val, y_val))
        # history = model.fit_generator(train_gen, available_train_steps, epochs=every_fold_epochs, verbose=1,
        #                                    callbacks=callback_list(save_dir, cur_k_fold=i + 1),
        #                               validation_data=val_gen, validation_steps=available_val_steps)
        end_time = time.time()
        print("结束训练时间：%s" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
        print("训练用时：%s" % time_spent(int(end_time - start_time)))  # 打印出共花费多少时间

        # 绘制每折交叉验证的 acc 折线图
        show_train_val(history.history['acc'], history.history['val_acc'], save_dir + "/%d_train_val_acc.png" % k)

        #
        # 取出本次训练的最终训练结果数据
        #
        train_history, val_history = extract_train_result(history.history, train_history, val_history)

        #
        # 计算 ROC
        #
        # y_score = model.predict_generator(roc_gen, available_val_steps, verbose=1)
        y_score = model.predict(x_val, batch_size, verbose=1)
        fpr, tpr, thresholds, auc_value = calc_auc(y_score, y_val)
        show_ROC(fpr, tpr, auc_value, save_dir + "/%d_roc.png" % k)
        save_auc_data(save_dir, k, [fpr, tpr, thresholds, auc_value])
        auc_history.append(auc_value)

    #
    # 统计最终结果
    #
    print("训练结果：", train_history)
    print("评估结果：", val_history)
    print("auc:", auc_history)
    show_every_fold_acc(train_history['acc'], val_history['val_acc'], save_dir + "/every_fold.png")
    # 输出总结一下，方便复制到表格中
    # summary(model_name, loss_name, batch_size, opt_name, real_train_num, real_val_num,
    #         k_fold*every_fold_epochs, k_fold, auc_history, train_history, val_history)
    rename_log_dir(save_dir, model_name, real_train_num, np.mean(train_history['acc']))


def calc_auc(y_score, y_val):
    """
    计算 AUC

    :param y_score:
    :param y_val:
    :param i:
    :return:
    """
    # ROC 数据（必要要得到 y_score 才能计算 roc）
    y_score = np.reshape(y_score, (y_score.shape[0],))  # 返回的数据是 (x,1)，多了一个维度
    fpr, tpr, thresholds = roc_curve(y_val, y_score)
    auc_value = auc(fpr, tpr)

    return fpr, tpr, thresholds, auc_value


def save_auc_data(save_dir, kFold_number, roc_data):
    """
    保存 ROC 数据
    :param save_dir:
    :param kFold_number:
    :param roc_data:
    :return:
    """
    np.save(save_dir + "/%d_fpr_tpr_thresholds_auc.npy" % kFold_number, roc_data)  # 保存 ROC 数据


def summary(model_name, loss_name, batch_size, optimizer_name, train_size, val_size, epochs, kFold, auc_history,
            train_result, val_result):
    """
    总结一下训练过程

    """
    out_str = (model_name + "\t" + loss_name + "\t" + str(batch_size) + "\t" + optimizer_name + "\t" +
               str(train_size) + "\t" + str(val_size) + "\t" + str(epochs) + "\t" + str(kFold) + "\t")

    out_str += "%.4f\t" % np.mean(auc_history)

    for v in train_result.values():
        out_str += "%.4f\t" % np.mean(v)
    # 验证评估分数是计算所有验证评估分数的平均值
    for v in val_result.values():
        out_str += "%.4f\t" % np.mean(v)
    print("( •̀ ω •́ )y 交叉验证统计结果，所有交叉验证的平均值：")
    print(out_str)


def rename_log_dir(log_dir, model_name, data_size, acc):
    """
    给日志文件夹重命名，方便人能看懂

    :param log_dir:
    :param model_name:
    :param data_size:
    :param acc:
    :return:
    """
    #
    # 获取当前文件夹的上一级目录
    #
    upper_dir = os.path.abspath(os.path.join(log_dir, ".."))
    new_save_dir = os.path.join(upper_dir, model_name + "_data-%d" % data_size + "_acc-%.2f" % acc)

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
    close_log()  # 必须要先关闭日志文件
    os.rename(log_dir, new_save_dir)  # 对根目录重命名


if __name__ == '__main__':
    #
    # 开始训练前的一些准备工作
    #
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"    # 不显示提示信息
    # save_dir = create_dir()                     # 创建一个保存目录
    # config_log(save_dir)                        # 配置日志文件

    #
    # 开始训练
    #
    # model, model_name = MLP_protein()    # 获取模型
    # train_model(tianjin_a_use_activation, save_dir)          # 训练模型

    # 数据
    # x, y = merged_pkl_data("./data/protein_sequence/equal/positive_fixed_number-arr.pkl",
    #                        "./data/protein_sequence/equal/negative_fixed_number-arr.pkl")
    x, y = merged_pkl_data_2()

    # save_dir = create_dir()  # 创建一个保存目录
    # config_log(save_dir)  # 配置日志文件
    # classifier = EasyEnsembleClassifier(random_state=42, n_jobs=-1)
    # classifier_name = "EasyEnsembleClassifier"
    # train_Classifier(classifier, classifier_name, save_dir, x, y)
    #
    # save_dir = create_dir()  # 创建一个保存目录
    # config_log(save_dir)  # 配置日志文件
    # classifier = BalancedBaggingClassifier(random_state=42, n_jobs=-1)
    # classifier_name = "BalancedBaggingClassifier"
    # train_Classifier(classifier, classifier_name, save_dir, x, y)
    #
    # save_dir = create_dir()  # 创建一个保存目录
    # config_log(save_dir)  # 配置日志文件
    # classifier = BalancedRandomForestClassifier(random_state=42, n_jobs=-1)
    # classifier_name = "BalancedRandomForestClassifier"
    # train_Classifier(classifier, classifier_name, save_dir, x, y)
    #
    # save_dir = create_dir()  # 创建一个保存目录
    # config_log(save_dir)  # 配置日志文件
    # classifier = RUSBoostClassifier(random_state=42)
    # classifier_name = "RUSBoostClassifier"
    # train_Classifier(classifier, classifier_name, save_dir, x, y)

    # save_dir = create_dir()  # 创建一个保存目录
    # config_log(save_dir)  # 配置日志文件
    # classifier = KNeighborsClassifier(n_jobs=-1)
    # classifier_name = "SMOTE-KNeighborsClassifier"
    # train_Classifier_2(classifier, classifier_name, save_dir, x, y, SMOTE)
    #
    # save_dir = create_dir()  # 创建一个保存目录
    # config_log(save_dir)  # 配置日志文件
    # classifier = DecisionTreeClassifier(random_state=42)
    # classifier_name = "SMOTE-DecisionTreeClassifier"
    # train_Classifier_2(classifier, classifier_name, save_dir, x, y, SMOTE)
    #
    # save_dir = create_dir()  # 创建一个保存目录
    # config_log(save_dir)  # 配置日志文件
    # classifier = KNeighborsClassifier(n_jobs=-1)
    # classifier_name = "RUS-KNeighborsClassifier"
    # train_Classifier_2(classifier, classifier_name, save_dir, x, y, RandomUnderSampler)
    #
    # save_dir = create_dir()  # 创建一个保存目录
    # config_log(save_dir)  # 配置日志文件
    # classifier = DecisionTreeClassifier(random_state=42)
    # classifier_name = "RUS-DecisionTreeClassifier"
    # train_Classifier_2(classifier, classifier_name, save_dir, x, y, RandomUnderSampler)

    save_dir = create_dir()  # 创建一个保存目录
    config_log(save_dir)  # 配置日志文件
    train_SMOTE_KNN(save_dir, x, y)