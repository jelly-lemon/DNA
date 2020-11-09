import sys


class Logger(object):
    """
    日志类，保存控制台输出到文件，同时控制台可以正常输出
    """

    def __init__(self, save_path, stream=sys.stdout):
        self.terminal = stream
        self.log_file = open(save_path, 'a', encoding="utf-8")

    def write(self, message):
        """
        向控制台和文件写入信息

        :param message:
        """
        # 我也不知道为啥可以读出 \r
        message = message.replace("\b", "")     # 去掉退格
        message = message.replace("\r", "\n")   # 把回车替换掉成回车换行

        self.terminal.write(message)
        self.log_file.write(message)        # 日志文件写详细信息


    def flush(self):
        pass


    def recover(self):
        """
        关闭日志文件，恢复标准输入输出流
        """
        self.log_file.close()
        sys.stdout = self.terminal



