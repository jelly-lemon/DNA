import os
import sys

from Logger import Logger


def create_dir(root_dir="./logs") -> str:
    """
    自动创建一个目录，用来保存训练相关的文件

    :param root_dir:指定一个根目录
    :return: 创建好的目录路径
    """
    # 根目录不存在就创建
    if os.path.exists(root_dir) is False:
        os.mkdir(root_dir)

    # 创建一个新日志目录
    file_list = os.listdir(root_dir)
    number_file = False
    for file_name in file_list:
        path = os.path.join(root_dir, file_name)
        if os.path.isfile(path):
            # 编号文件数字加 1，作为这次的文件夹编号
            dir_number = str(int(file_name) + 1)
            os.rename(path, os.path.join(root_dir, dir_number))
            number_file = True
            break

    # 如果编号文件不存在，就创建一个
    if number_file is False:
        dir_number = 1
        with open(os.path.join(root_dir, "1"), "w") as file:
            file.close()

    dir = os.path.join(root_dir, "test_%s" % dir_number)
    os.mkdir(dir)

    return dir

def config_log(save_dir: str, name="log.txt"):
    """
    配置日志相关

    :param save_dir:保存路径
    :param name: 日志文件名字
    """
    #
    # 日志文件
    #
    log_path = os.path.join(save_dir, name)
    sys.stdout = Logger(log_path, sys.stdout)

    #
    # 报错文件
    #
    #err_path = os.path.join(save_dir, "err.txt")
    #sys.stderr = Logger(err_path, sys.stderr)

def close_log():
    """
    关闭日志文件，恢复标准输入输出流
    """
    sys.stdout.recover()

def time_spent(all_seconds):
    """
    把花费的秒数换算成几天几小时几秒的格式

    :param all_seconds: 总秒数
    :return:
    """
    time_str = ""   # 输出的花费时间字符串
    days = hours = minutes = seconds = 0

    # 判断一定要加一个 int，不然会有小数，永远不可能为 0
    if int(all_seconds/(24*60*60)) > 0:
        days = int(all_seconds/(24*60*60))
        all_seconds -= days*(24*60*60)
        time_str += "%2dd " % days
    if int(all_seconds/(60*60)) > 0:
        hours = int(all_seconds/(60*60))
        all_seconds -= hours*(60*60)
        time_str += "%2dh " % hours
    if int(all_seconds/60) > 0:
        minutes = int(all_seconds/60)
        all_seconds -= minutes*60
        time_str += "%2dm " % minutes
    seconds = all_seconds
    time_str += "%2ds" % seconds

    return time_str

