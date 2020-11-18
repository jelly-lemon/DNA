import tensorflow as tf
from tensorflow.python.keras import backend as K


def G_mean(y_true, y_pred):
    """
    G-mean
    :param y_true:
    :param y_pred:
    :return:
    """
    total = tf.cast(tf.size(y_true), tf.float32)
    tp_fn = tf.reduce_sum(y_true)
    tn_fp = total - tp_fn

    tp_fp = K.sum(K.round(K.clip(y_pred, 0, 1)))
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = total - (tp_fp + tp_fn - tp)


    # TPR = TP/(TP+FN) = Recall
    tpr = tp/(tp_fn + K.epsilon())

    # TNR = TN/(TN+FP)
    tnr = tn/(tn_fp + K.epsilon())

    g_mean = (tpr*tnr)**0.5

    return g_mean


def Precision(y_true, y_pred):
    """
    精确率
    :param y_true: 都是张量
    :param y_pred:
    :return:
    """
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    # pp = TP + FP
    tp_fp = K.sum(K.round(K.clip(y_pred, 0, 1))) # predicted positives
    precision = tp/(tp_fp + K.epsilon())
    return precision

def Recall(y_true, y_pred):
    """
    召回率
    :param y_true:
    :param y_pred:
    :return:
    """
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    # pp = TP + FN
    tp_fn = K.sum(K.round(K.clip(y_true, 0, 1)))           # possible positives
    recall = tp / (tp_fn + K.epsilon())
    return recall

def F1(y_true, y_pred):
    """
    F1-score
    :param y_true:
    :param y_pred:
    :return:
    """
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1