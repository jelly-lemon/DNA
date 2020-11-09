"""
读取 DNA 序列文件，也就是 *.txt，然后转存为 *.npy
DNA 序列只包含 A, T, C, G
"""

import numpy as np


def to_number(str) -> np.ndarray:
    """
    把一个 DNA 序列转成数字数组
    如 'ATGC...' -> [0 1 1 3]
    :param str:DNA序列
    :return:数字数组
    """
    DNA_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    num_list = []
    for c in str:
        num_list.append(DNA_dict[c])
    return np.array(num_list)


def save_npy(file_path, save_suffix):
    """
    读取 DNA 序列文件，将 DNA 序列和标签保存为 *.npy 文件
    :param file_path:文件路径
    :param save_suffix:保存后缀
    """
    # read *.txt file
    with open(file_path) as file:
        # get total numbers of lines
        lines = file.readlines()

        # for circle to resolve every line.
        x_train = None
        y_train = None
        for index, line in enumerate(lines):
            print("line", index)
            str_list = line.split()
            x = to_number(str_list[1]).reshape(1, 101)
            y = np.array([int(str_list[2])])

            if x_train is None:
                x_train = x
                y_train = y
            else:
                x_train = np.concatenate((x_train, x))
                y_train = np.concatenate((y_train, y))

        # x_train = np_utils.to_categorical(x_train)
        print(x_train)
        print(y_train)
        # save ndarray as *.npy file
        np.save("./data/x_ori_%s.npy" % save_suffix, x_train)
        np.save("./data/y_ori_%s.npy" % save_suffix, y_train)




if __name__ == '__main__':
    # save as *.npy
    file_path = "./data/train.data.txt"
    save_npy(file_path, "train")

    # y_val = np.load("./DNA/y_val.npy")
    # print(y_val.dtype)
    # print(y_val)
