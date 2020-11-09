"""
读取蛋白质序列
"""
import numpy as np
import random

from math import floor

convert_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
                'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
                'V': 18, 'W': 19, 'Y': 20}

illegal_char = {'B', 'J', 'O', 'U', 'X', 'Z'}

def to_onehot_matrix(original_str: str) -> np.ndarray:
    """
    将原始蛋白质序列字符串转化为 one-hot 编码矩阵 (1000,20)

    序列长度不足 1000 则在后面补全 0 向量

    例如：
    "AC" ->
    [[1 0 0 ... 0 0 0]
            ...
     [0 0 0 ... 0 0 0]]

    :param original_str: 原始蛋白质序列字符串
    :return: one-hot矩阵
    """
    #
    # 检查
    #
    if original_str in ("", "\n"):
        raise ValueError("传入的蛋白质序列为空串")

    #
    # 去掉末尾的 \n
    #
    original_str = original_str.replace("\n", "")
    length = len(original_str)

    #
    # 不足 1000 则补齐
    #
    if length < 1000:
        dif = 1000 - length  # 差额，需要补多少个字母
        new_str = original_str + "X" * dif
    else:
        new_str = original_str[:1000]  # 大于等于 1000 时则只截取前 1000 个字符

    #
    # 转为 One-hot 矩阵
    #
    onehot_mat = None
    for c in new_str:
        try:
            vct = to_vector(c)  # 将字母转化为 one-hot 向量
        except ValueError:
            vct = to_vector('X')  # 遇到非氨基酸字母，就处理为 'X'

        if onehot_mat is None:
            onehot_mat = vct
        else:
            onehot_mat = np.concatenate((onehot_mat, vct))

    return onehot_mat


def to_vector(c: str) -> np.ndarray:
    """
    将一个字母转化为 one-hot 向量

    并且将向量 reshape 为 (1,20)，方便后面进行拼接
    如果是 'X'，直接返回全零向量

    例如：
    'A' -> [[1 0 0 0 0 ... 0 0 0]]
    'X' -> [[0 0 0 0 0 ... 0 0 0]]


    :param c: 一个字母
    :return: 转换后的向量（np 数组）
    """

    #
    # 全零向量
    #
    vec = np.zeros((20,), np.uint8)

    #
    # 字母 c 是氨基酸字母，才会将对应位置为 1
    #
    if c in convert_dict:
        vec[convert_dict[c] - 1] = 1
    elif c == 'X':
        pass
    else:
        raise ValueError("字母'%c'不是氨基酸" % c)

    #
    # reshape 为 (1,20)，方便后面 concatenate
    #
    vec = np.reshape(vec, (1, 20))

    return vec


def replace(original_str: str) -> str:
    """
    将文件中的一行序列里面出现的不合法字符替换为 'X'

    :param original_str: 从文件中读取的一行
    :return:
    """

    new_str = ""
    for c in original_str:
        if c in convert_dict or c == '\n':
            new_str += c
        else:
            new_str += 'X'

    return str(new_str)


def fix_data():
    """
    去掉原始文件中的标识行和不合法字符

    :return:
    """
    # positive.fasta
    # file_path = "./data/protein_sequence/equal/positive.fasta"
    # new_file_path = "./data/protein_sequence/equal/positive_fixed.txt"
    # negative.fasta
    file_path = "./data/protein_sequence/equal/negative.fasta"
    new_file_path = "./data/protein_sequence/equal/negative_fixed.txt"

    with open(file_path) as file, open(new_file_path, "w") as new_file:
        # 读取标识行
        line = file.readline()
        while line:
            # 读取蛋白质序列行
            line = file.readline()
            # 替换不合法字符（不是氨基酸）
            line = replace(line)
            # 写入新文件
            new_file.write(line)
            # 读取标识行
            line = file.readline()
        file.close()
        new_file.close()


def test():
    file_path = "./data/protein_sequence/equal/positive_fixed.txt"
    with open(file_path) as file:
        lines = file.readlines()
        print(lines[:5])
        random.shuffle(lines)
        print(lines[:5])
        file.close()


def convert_lines(lines: list) -> np.ndarray:
    """
    将数行氨基酸序列转换为 1000*20 的 one-hot 矩阵

    :param lines:
    :return:
    """
    one_hot_batch = None
    for line in lines:
        t = to_onehot_matrix(line)  # 将一行氨基酸序列转换为 one_hot 矩阵
        t = t.reshape(1, t.shape[0], t.shape[1])

        if one_hot_batch is None:
            one_hot_batch = t
        else:
            one_hot_batch = np.concatenate((one_hot_batch, t))

    return one_hot_batch


def merged_data(num = None):
    """
    合并正负数据集，并且打乱，返回所有数据

    @:param num 指定返回数据的数量
    :return:
    """
    positive_path = "./data/protein_sequence/equal/positive_fixed.txt"  # 正数据集
    negative_path = "./data/protein_sequence/equal/negative_fixed.txt"  # 负数据集

    with open(positive_path) as pos_file, open(negative_path) as neg_file:
        # 读取文件中的数据
        pos_lines = pos_file.readlines()
        pos_label = np.ones((len(pos_lines),), dtype=np.uint8)

        neg_lines = neg_file.readlines()
        neg_label = np.zeros((len(neg_lines),), dtype=np.uint8)

        x = pos_lines + neg_lines
        y = np.concatenate((pos_label, neg_label))

        # 打乱数据
        x, y = shuffle_data(x, y)
        y = np.array(y)

        if num is not None and num < y.shape[0]:
            return x[:num], y[:num]

        return x, y

def merged_batch(batch_size):
    """
    数据生成器，将正负数据合并了，一个生成一个 batch
    :param batch_size:
    :return:
    """
    x, y = merged_data()  # 已经打乱了

    available_steps = np.floor(y.shape[0] / batch_size)

    i = 0
    while True:
        if i + 1 == available_steps:
            x, y = shuffle_data(x, y)  # 打乱数据
            i = 0
        start = i * batch_size
        end = (i + 1) * batch_size
        x_batch = x[start:end]
        x_batch = convert_lines(x_batch)
        y_batch = y[start:end]
        y_batch = np.array(y_batch)
        yield x_batch, y_batch
        i += 1


def shuffle_data(x, y):
    """
    打乱数据
    :param x:
    :param y:
    :return:
    """
    data = list(zip(x, y))
    random.shuffle(data)
    x, y = zip(*data)

    return x, y


def get_gen(x, y, batch_size):
    """
    根据给定数据和批大小，转换为one-hot矩阵，按批产出
    :param x: 氨基酸序列，列表
    :param batch_size: 批大小
    :return:
    """
    # 计算可以生成多少个 batch
    steps = floor(len(x) / batch_size)

    # 无限循环生成 batch
    i = 0
    while True:
        start = i * batch_size
        end = (i + 1) * batch_size
        x_train = convert_lines(x[start:end])

        if y is None:
            yield x_train
        else:
            yield x_train, y[start:end]

        i += 1

        # 当所有数据都已经生成一遍了，再从头开始
        if i == steps:
            i = 0


def data_gen(batch_size, data_num: int = 80000):
    """
    数据迭代器，按 batch_size 读取数据

    :param batch_size: 批大小
    :param data_num: 规定取数据集多少条，剩下的用来作验证集
    """
    positive_path = "./data/protein_sequence/positive.fasta"  # 正数据集
    negative_path = "./data/protein_sequence/negative.fasta"  # 负数据集

    error_str = ["", "\n"]  # 如果读取到的某一行是不合法的字符串
    half_size = batch_size / 2  # 正负数据各取一半
    half_num = data_num / 2  # 正负数据各取一半

    #
    # 开始读取
    #
    with open(positive_path) as pos_file, open(negative_path) as neg_file:
        while True:
            #
            # 从头开始
            #
            pos_file.seek(0, 0)
            neg_file.seek(0, 0)
            i = 0  # 第几条蛋白质序列
            x_train = None  # 训练数据集
            y_train = None

            #
            # 读取标识行
            #
            pos_line = pos_file.readline()
            neg_line = neg_file.readline()

            while pos_line and neg_line:
                # 读取蛋白质序列行
                pos_line = pos_file.readline()
                neg_line = neg_file.readline()

                if (pos_line not in error_str) and (neg_line not in error_str):
                    i += 1
                else:
                    #
                    # 读取的蛋白质序列为空串
                    # 证明到读到文件末尾了
                    # 那就跳出 while 循环
                    #
                    break

                #
                # 超过指定读取的数量了
                # 跳出循环，从头开始读
                #
                if i > half_num:
                    break

                #
                # 蛋白质序列转化为 one-hot 矩阵
                #
                pos_onehot = to_onehot_matrix(pos_line)
                pos_onehot = pos_onehot.reshape((1,) + pos_onehot.shape)

                neg_onehot = to_onehot_matrix(neg_line)
                neg_onehot = neg_onehot.reshape((1,) + neg_onehot.shape)

                #
                # 拼接在一起
                #
                if x_train is None:
                    x_train = np.concatenate((pos_onehot, neg_onehot))
                    y_train = np.array([1, 0], dtype=np.uint8)
                else:
                    x_train = np.concatenate((x_train, pos_onehot, neg_onehot))
                    y_train = np.concatenate((y_train, np.array([1, 0], dtype=np.uint8)))

                #
                # 生成一个 batch
                #
                if i % half_size == 0:
                    # print("wow，生成一个 batch")
                    yield x_train, y_train
                    x_train = y_train = None

                #
                # 读取标识行，判断是否还有序列可以读
                #
                pos_line = pos_file.readline()
                neg_line = neg_file.readline()


def val_data():
    """
    验证集

    :return:验证集
    """

    positive_path = "./data/protein_sequence/positive.fasta"  # 正数据集
    negative_path = "./data/protein_sequence/negative.fasta"  # 负数据集

    error_str = ["", "\n"]  # 如果读取到的某一行是不合法的字符串
    i = 0
    with open(positive_path) as pos_file, open(negative_path) as neg_file:
        x_val = None  # 训练数据集
        y_val = None

        # 读取标识行
        pos_line = pos_file.readline()
        neg_line = neg_file.readline()

        while pos_line and neg_line:
            # 总共有 4w2k+ 条
            if i < 40000:
                # 读取蛋白质序列
                pos_file.readline()
                pos_file.readline()

                i += 1

                # 读取标识行
                neg_file.readline()
                neg_file.readline()

                continue

            # 读取蛋白质序列行
            pos_line = pos_file.readline()
            neg_line = neg_file.readline()

            if (pos_line not in error_str) and (neg_line not in error_str):
                i += 1
            else:
                # 读取的蛋白质序列为空串
                # 证明到读到文件末尾了
                # 那就跳出 while 循环
                break

            # 蛋白质序列转化为 one-hot 矩阵
            pos_onehot = to_onehot_matrix(pos_line)
            pos_onehot = pos_onehot.reshape((1,) + pos_onehot.shape)

            neg_onehot = to_onehot_matrix(neg_line)
            neg_onehot = neg_onehot.reshape((1,) + neg_onehot.shape)

            # 拼接在一起
            if x_val is None:
                x_val = np.concatenate((pos_onehot, neg_onehot))
                y_val = np.array([1, 0], dtype=np.uint8)
            else:
                x_val = np.concatenate((x_val, pos_onehot, neg_onehot))
                y_val = np.concatenate((y_val, np.array([1, 0], dtype=np.uint8)))

            # 读取标识行
            pos_line = pos_file.readline()
            neg_line = neg_file.readline()

    return x_val, y_val


if __name__ == '__main__':
    gen = merged_batch(32)
    x, y = next(gen)
    print(x)
    print(y)
    x, y = next(gen)
    print(x)
    print(y)

    # pos_label = np.ones((3,), dtype=np.uint8)
    # print(pos_label)
    # neg_label = np.zeros((1, ), dtype=np.uint8)
    # print(neg_label)
    # y = pos_label + neg_label
