"""
读取蛋白质序列
"""
import os

import numpy as np
import random

from math import floor

convert_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
                'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
                'V': 18, 'W': 19, 'Y': 20, 'X':0}

illegal_char = {'B', 'J', 'O', 'U', 'X', 'Z'}

def to_number_array(original_str: str):
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
    # 转为数字矩阵
    #
    number_arr = []
    for c in new_str:
        number_arr += [convert_dict[c]]

    number_arr = np.array(number_arr, dtype=np.uint8)
    return number_arr



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


def fix_data(file_path: str):
    """
    去掉原始文件中的标识行和不合法字符

    可以对所有 fasta 文件调用该函数
    """
    file_name = os.path.basename(file_path).split('.')[0]
    new_file_path = os.path.abspath(os.path.dirname(file_path)) + "/" + file_name + "_fixed.txt"

    with open(file_path) as file, open(new_file_path, "w") as new_file:
        # 读取标识行
        line = file.readline()
        while line:
            # 读取蛋白质序列行（\r 直接忽略，不会认为有换行）
            line = file.readline()
            # 替换不合法字符（不是氨基酸）
            line = replace(line)
            # 写入新文件
            new_file.write(line)
            # 读取标识行
            line = file.readline()
        file.close()
        new_file.close()

def get_species():
    """
    获取指定物种的蛋白质序列

    包含 DNA-binding 和 非 DNA-binding
    标签行也保留
    去掉小于40或大于1000的序列
    替换不合法氨基酸字符为 X
    物种名字有 Human, Mouse, Rice
    """
    species_name = "Rice"
    species_name = species_name.upper()
    original_file_path = "./data/protein_sequence/unbalanced/unbalance.fasta"
    # 获取文件名
    file_name = os.path.basename(original_file_path).split('.')[0]
    species_file_path = os.path.dirname(original_file_path) + "/" + species_name + ".txt"

    # 给人看的
    line_num = 0
    i = 0
    with open(original_file_path) as file, open(species_file_path, "w") as new_file:
        # 读取标识行
        id_line = file.readline()
        line_num += 1

        while id_line:
            print("id_line_num:%d" % line_num)

            # 读取蛋白质序列行
            # 序列被 LF 分割成了好几段
            # 读进来会被当做 \n
            # 这个时候需要去掉中间的 \n，然后拼接成完整的序列
            seq_line = file.readline()
            line_num += 1
            merged_seq_line = None

            # 标识行 >sp 开头，返回 -1 表示不是标识行
            # 还要考虑读到最后一串的情况
            while seq_line.find(">sp") == -1 and seq_line != "":
                if merged_seq_line is None:
                    merged_seq_line = seq_line.replace("\n", "")
                else:
                    merged_seq_line += seq_line.replace("\n", "")
                seq_line = file.readline()
                line_num += 1

            # 难道存在连续两行都是 >sp 开头？
            if merged_seq_line is not None:
                # 判断该标识行是否包含指定物种单词
                if id_line.upper().find(species_name) != -1:
                    # 只保留长度为 40~1000 的序列
                    if 40 <= len(merged_seq_line) and len(merged_seq_line) <= 1000:
                        merged_seq_line += "\n"  # 最后加一个换行
                        # 替换不合法字符（不是氨基酸）
                        merged_seq_line = replace(merged_seq_line)
                        # 写入新文件
                        new_file.write(id_line) # 标识行也一起写入
                        new_file.write(merged_seq_line)
            # 跳出 while 循环的 seq_line 内容肯定是标识行
            id_line = seq_line
        print("保存到文件中...")
        file.close()
        new_file.close()
        print("保存成功")

def remove_DNA_seq():
    """
    去掉 unbalanced.fasta 文件中的 DNA-binding 序列

    因为原始 unbalanced.fasta 文件中有 DNA-binding 序列
    文件中的换行是 LF，读进来都会被当做 \n
    1. 去掉 DNA-binding 序列
    2. 去掉小于40或大于1000的序列
    3. 替换不合法氨基酸字符为 X
    最后只剩下：每一行一条序列
    """
    original_file_path = "./data/protein_sequence/unbalanced/unbalance.fasta"
    # 获取文件名
    file_name = os.path.basename(original_file_path).split('.')[0]
    no_DNA_binding_file_path = os.path.dirname(original_file_path) + "/" + file_name + "_no_DNA-binding_fixed.txt"

    # 给人看的
    line_num = 0
    i = 0
    with open(original_file_path) as file, open(no_DNA_binding_file_path, "w") as new_file:
        # 读取标识行
        id_line = file.readline()
        line_num += 1

        while id_line:
            print("id_line_num:%d" % line_num)
            # 发现有 DNA-binding
            if id_line.find("DNA-binding") != -1:
                i += 1
                print("发现第%d条DNA-binding序列，%s" % (i, id_line))
                # 跳过后面所有的序列
                seq_line = file.readline()
                line_num += 1
                while seq_line.find(">sp") == -1:
                    seq_line = file.readline()
                    line_num += 1
            else:
                # 读取蛋白质序列行
                # 序列被 LF 分割成了好几段
                # 读进来会被当做 \n
                # 这个时候需要去掉中间的 \n，然后拼接成完整的序列
                seq_line = file.readline()
                line_num += 1
                merged_seq_line = None

                # 标识行 >sp 开头，返回 -1 表示不是标识行
                # 还有考虑读到最后一串的情况
                while seq_line.find(">sp") == -1 and seq_line != "":
                    if merged_seq_line is None:
                        merged_seq_line = seq_line.replace("\n", "")
                    else:
                        merged_seq_line += seq_line.replace("\n", "")
                    seq_line = file.readline()
                    line_num += 1

                # 难道存在连续两行都是 >sp 开头？
                if merged_seq_line is not None:
                    # 只保留长度为 40~1000 的序列
                    if 40 <= len(merged_seq_line) and len(merged_seq_line) <= 1000:
                        merged_seq_line += "\n"  # 最后加一个换行
                        # 替换不合法字符（不是氨基酸）
                        merged_seq_line = replace(merged_seq_line)
                        # 写入新文件
                        new_file.write(merged_seq_line)
            # 跳出 while 循环的 seq_line 内容肯定是标识行
            id_line = seq_line
        print("保存到文件中...")
        file.close()
        new_file.close()
        print("保存成功")

def convert_lines_to_number(lines: list):
    """
    将蛋白质序列列表转化为数字矩阵

    :param lines:
    :return:
    """
    number_matrix = None
    for line in lines:
        t = to_number_array(line)
        t = t.reshape(1, t.shape[0])

        if number_matrix is None:
            number_matrix = t
        else:
            number_matrix = np.concatenate((number_matrix, t))

    return number_matrix

def convert_lines_to_onthot(lines: list) -> np.ndarray:
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
        x = pos_lines + neg_lines                   # 数据类型是 ['abc', 'abc']
        y = np.concatenate((pos_label, neg_label))  # 数据类型是 ndarray

        # 打乱数据
        x, y = shuffle_data(x, y)
        y = np.array(y) # 数据类型是 ndarray

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
        x_batch = convert_lines_to_onthot(x_batch)
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

def get_number_gen(x, y, batch_size):
    """
    根据给定数据和批大小，转换为数字矩阵，按批产出

    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    # 计算可以生成多少个 batch
    available_steps = floor(len(x) / batch_size)

    # 无限循环生成 batch
    i = 0
    while True:
        start = i * batch_size
        end = (i + 1) * batch_size
        x_train = convert_lines_to_number(x[start:end])

        if y is None:
            yield x_train
        else:
            yield x_train, y[start:end]

        i += 1

        # 当所有数据都已经生成一遍了，再从头开始
        if i == available_steps:
            i = 0

def get_onehot_gen(x, y, batch_size):
    """
    根据给定数据和批大小，转换为 one-hot 矩阵，按批产出

    :param x: 氨基酸序列，列表
    :param batch_size: 批大小
    :return:
    """
    # 计算可以生成多少个 batch
    available_steps = floor(len(x) / batch_size)

    # 无限循环生成 batch
    i = 0
    while True:
        start = i * batch_size
        end = (i + 1) * batch_size
        x_train = convert_lines_to_onthot(x[start:end])

        if y is None:
            yield x_train
        else:
            yield x_train, y[start:end]

        i += 1

        # 当所有数据都已经生成一遍了，再从头开始
        if i == available_steps:
            i = 0








if __name__ == '__main__':
    pass