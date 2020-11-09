"""
 CNN 网络，用来对一维数据进行二分类

 DNA 序列二分类，判断是不是结合位点

 蛋白质序列二分类，判断是不是结合蛋白

"""

from tensorflow.python.keras import Input, Model, callbacks, Sequential, regularizers, metrics, losses
from tensorflow.python.keras.layers import Conv1D, Flatten, Dense, \
    GlobalMaxPooling1D, Permute, concatenate, LSTM, Embedding, Dropout, Convolution1D, MaxPooling1D, Activation
from tensorflow.python.keras import layers


from util import *


def CNN_16kernal() -> Model:
    """
    一个基本的 CNN 网络，用来对一维数据进行二分类
    :return: 创建好的模型
    """
    print("model: cnn_16kernal")

    # input = Input((101, 4))
    input = Input((1000, 20))
    x = Conv1D(16, 24, activation='relu')(input)
    x = GlobalMaxPooling1D()(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.summary()

    return model


def CNN_32kernal() -> Model:
    """
    一个基本的 CNN 网络，用来对一维数据进行二分类
    :return:创建好的模型
    """
    input = Input((101, 4))
    x = Conv1D(32, 24, activation='relu')(input)
    x = GlobalMaxPooling1D()(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.summary()

    return model


def CNN_16_32kernal() -> Model:
    """
    一个基本的 CNN 网络，用来对一维数据进行二分类

    :return:创建好的模型
    """
    input = Input((101, 4))
    x = Conv1D(16, 24, activation='relu')(input)
    x = Conv1D(32, 24, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.summary()

    return model

def tianjin_a_original() -> Model:
    """
    天津大学 cnn 原始模型
    a 分类方式
    :return:创建好的模型
    """
    print("model: tianjin_a_original")


    #
    # 序列处理模块
    #
    input = Input((1000, 20))
    x = Conv1D(64, 11, padding='same', activation=None, use_bias=False)(input)
    x = Permute([2, 1])(x)  # 对矩阵进行转置
    x = Conv1D(1, 1, padding='same', activation=None, use_bias=False)(x)

    #
    # 分类模块 a
    #
    x = Flatten()(x)
    output = Dense(1, activation=None, use_bias=False)(x)  # 标签没有 one-hot 化，故输出节点仅 1 个

    model = Model(inputs=input, outputs=output)
    print("model: tianjin_a")
    model.summary()

    return model

def tianjin_a_use_activation() -> Model:
    """
    天津大学 cnn，添加了激活函数
    a 分类方式
    :return:创建好的模型
    """
    print("model: tianjin_a_use_activation")

    #
    # 序列处理模块
    #
    input = Input((1000, 20))
    x = Conv1D(64, 11, padding='same', activation='relu', use_bias=True)(input)
    x = Permute([2, 1])(x)  # 对矩阵进行转置
    x = Conv1D(1, 1, padding='same', activation='relu', use_bias=True)(x)

    #
    # 分类模块 a
    #
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid', use_bias=True)(x)  # 标签没有 one-hot 化，故输出节点仅 1 个

    model = Model(inputs=input, outputs=output)
    model.summary()

    return model


def tianjin_b_original():
    """
    天津大学 cnn
    b 分类方式

    :return:创建好的模型
    """
    #
    # 序列处理模块
    #
    input = Input((1000, 20))
    x = Conv1D(64, 11, padding='same', activation=None, use_bias=False)(input)
    x = Permute([2, 1])(x)  # 对矩阵进行转置
    x = Conv1D(1, 1, padding='same', activation=None, use_bias=False)(x)

    #
    # 分类模块 b
    #
    x = Flatten()(x)
    x = subnet_layer_original()(x)
    output = Dense(1, activation=None, use_bias=False, kernel_regularizer=regularizers.l1(0.01))(x)

    model = Model(inputs=input, outputs=output)
    model.summary()

    model_name = "tianjin_b_original"

    return model, model_name

def tianjin_b_use_bias():
    """
    天津大学 cnn
    b 分类方式

    :return:创建好的模型
    """
    model_name = "tianjin_b_use_bias"

    #
    # 序列处理模块
    #
    input = Input((1000, 20))
    x = Conv1D(64, 11, padding='same', activation='relu', use_bias=True)(input)
    x = Permute([2, 1])(x)  # 对矩阵进行转置
    x = Conv1D(1, 1, padding='same', activation='relu', use_bias=True)(x)

    #
    # 分类模块 b
    #
    x = Flatten()(x)
    x = subnet_layer_original()(x)
    output = Dense(1, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.l1(0.01))(x)

    model = Model(inputs=input, outputs=output)
    model.summary()

    return model, model_name

def tianjin_LSTM():
    """
    天津大学英文论文 LSTM 模型
    """
    input = Input((1000,))
    x = Embedding(21, 128, input_length=1000)(input)
    x = Dropout(0.5)(x)
    x = Conv1D(64, 10, padding='valid', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 5, padding='valid', activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = LSTM(70)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.summary()
    model_name = "tianjin_LSTM"

    return model, model_name

class subnet_layer_original(layers.Layer):
    """
    天津大学学位论文子网络层
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_subnet(self):
        """
        返回子网络

        :return: 子网络
        """
        subnet = Sequential([
            Dense(12, activation='sigmoid', use_bias=False),
            Dense(6, activation='tanh', use_bias=False),
            Dense(1, activation='sigmoid')
        ])

        return subnet

    def call(self, inputs, **kwargs):
        """
        拼接子网络为层

        :param inputs:模型输入
        :param kwargs:子网络层
        :return:
        """
        all_subnet = []
        ops_num = inputs.shape[1]
        for i in range(ops_num):
            all_subnet.append(self.get_subnet()(inputs[:, i:i + 1]))
        outputs = concatenate(all_subnet)

        return outputs

class subnet_layer_use_bias(layers.Layer):
    """
    天津大学学位论文子网络层
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_subnet(self):
        """
        返回子网络

        :return: 子网络
        """
        subnet = Sequential([
            Dense(12, activation='sigmoid', use_bias=True),
            Dense(6, activation='tanh', use_bias=True),
            Dense(1, activation='sigmoid')
        ])

        return subnet

    def call(self, inputs, **kwargs):
        """
        拼接子网络为层

        :param inputs:模型输入
        :param kwargs:子网络层
        :return:
        """
        all_subnet = []
        ops_num = inputs.shape[1]
        for i in range(ops_num):
            all_subnet.append(self.get_subnet()(inputs[:, i:i + 1]))
        outputs = concatenate(all_subnet)

        return outputs


def MLP_DNA_binding():
    """
    多层感知机
    :return:
    """
    print("model name: MLP_DNA_binding")

    input = Input((101, 4))
    x = Dense(128, activation='relu')(input)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Flatten()(x)
    output = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.summary()

    return model

def MLP_protein():
    """
    多层感知机

    :return:
    """
    input = Input((1000, 20))
    x = Dense(128, activation='relu')(input)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=output)
    model.summary()
    model_name = "MLP_protein"

    return model, model_name


def callback_list(save_dir, cur_k_fold=0):
    """
    训练模型回调函数

    :param save_dir:
    :param cur_k_fold: 当前第几次 k 折交叉验证
    :return:
    """
    # 创建用来保存的文件夹
    os.makedirs(save_dir, exist_ok=True)
    # 模型保存文件名
    model_save_path = save_dir + "/%d_epoch{epoch:02d}_acc{acc:.2f}.hdf5" % cur_k_fold
    checkpoint = callbacks.ModelCheckpoint(model_save_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
    # earlystopper = callbacks.EarlyStopping(monitor='acc', patience=10, verbose=1)
    tensor_board = callbacks.TensorBoard(save_dir, histogram_freq=0, write_graph=True, write_grads=True)  # 训练日志

    return [checkpoint, tensor_board]






