import random
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from read_data import yeast1


class DBUOperator:
    def __init__(self, x, y):

        # 参考多少个邻居
        self.K_Neighbor = 5
        self.H_Neighbor = 5

        # 样本分类
        self.T_p = [list(x[i]) for i in range(len(y)) if y[i] == 1] # 正样本集
        self.T_n = [list(x[i]) for i in range(len(y)) if y[i] == 0] # 负样本集
        self.sample_number = len(x)  # 样本总数量
        self.R = len(self.T_p) / len(self.T_n)  # 样本比例（多比少，大于1）
        self.N = len(self.T_n)  # 负样本数量
        self.delt_star = None
        self.Ri = []    # 采样间隔范围
        self.delt_p_i_vec = np.zeros((len(self.T_p, )), dtype=np.float32)
        self.delt_n_i_vec = np.zeros((len(self.T_p, )), dtype=np.float32)
        self.d_p_i_vec = np.zeros((len(self.T_p, )), dtype=np.float32)
        self.d_n_i_vec = np.zeros((len(self.T_p, )), dtype=np.float32)
        self.delt_i_vec = np.zeros((len(self.T_p, )), dtype=np.float32)


        # self.d_p_ij_mat = self.d_ij_mat(self.T_p)
        # self.d_n_ij_mat = self.d_ij_mat(self.T_n)

    # def d_ij_mat(self, data):
    #     """
    #     欧式距离矩阵，下标从1开始
    #
    #     :param data:样本，(n_samples, sample_length)
    #     :return:
    #     """
    #     n = len(data)
    #     mat = np.zeros((n + 1, n + 1))  # 下标从1开始
    #     for i, i_data in enumerate(data):
    #         for j, j_data in enumerate(data):
    #             mat[i][j] = np.sqrt(np.sum(np.square(np.array(i_data) - np.array(j_data))))
    #
    #     return mat

    def get_new_set(self):
        """
        下采样生产数据集
        :return:
        """
        count = 0
        T_p_new = []

        while count <= len(self.T_n) - 10:
            # 随机生成 [0,1] 的一个数
            r = random.uniform(0, 1)
            for j in range(1, len(self.T_n) + 1):
                start, end = self.Ri_2(j)
                if start < r <= end:
                    if self.T_p[j - 1] not in T_p_new:
                        count += 1
                        print("加入一个样本（编号%d），还差%d个"%(j, len(self.T_n) - count))
                        T_p_new.append(self.T_p[j - 1])

                        break

        T_new = T_p_new + self.T_n
        T_new = np.array(T_new)

        y_p = np.ones((len(T_p_new),), dtype=np.uint8)
        y_n = np.zeros((len(self.T_n),), dtype=np.uint8)
        y = np.concatenate((y_p, y_n))

        return T_new, y

    # def d_p_ij(self, i, j):
    #     """
    #     i 与 j 的欧式距离
    #     :param i:
    #     :param j:
    #     :return:
    #     """
    #     return self.d_p_ij_mat[i][j]

    # def d_n_ij(self, i, j):
    #     return self.d_n_ij_mat[i][j]

    # def di(self, i):
    #     """
    #     di 是 Xi 到 k 近邻的平均距离
    #
    #     :param i:第几个样本
    #     :param dij_matrix:
    #     :return:
    #     """
    #     di_sum = 0
    #     # TODO 要改成 k 近邻
    #     for j in range(1, self.K_Neighbor + 1):
    #         di_sum += self.dij(i, j)
    #     di_sum /= self.K_Neighbor
    #
    #     return di_sum

    def d_p_i(self, i):
        """

        :param i: 正样本中的某个数据下标，从0开始
        :return:
        """
        if self.d_p_i_vec[i-1] == 0:
            # 先找最近的 k 个邻居
            dis = self.get_dis(self.T_p[i - 1], self.T_p)
            # 挑出 k 近邻
            res = np.sum(dis[:self.K_Neighbor]) / self.K_Neighbor
            self.d_p_i_vec[i - 1] = res

        return self.d_p_i_vec[i - 1]

    def get_dis(self, cur_point, all_point):
        """
        获取当前节点到其余节点的欧式距离（升序）

        :param cur_point:当前节点
        :param all_point:所有节点
        :return:
        """
        # 对拷贝进行操作
        all_point = all_point.copy()

        if cur_point in all_point:
            all_point.remove(cur_point)
        dis = []
        for point in all_point:
            t = np.sqrt(np.sum(np.square(np.array(cur_point) - np.array(point))))
            dis.append(t)
        sorted(dis)

        return dis

    def d_n_i(self, i):
        """

        :param i: 正样本中的某个数据下标，从0开始
        :return:
        """
        if self.d_n_i_vec[i - 1] == 0:
            # 先找最近的 h 个邻居
            dis = self.get_dis(self.T_p[i - 1], self.T_n)
            # 取前 h 个最近的，进行求和，求 d_n_i
            res = np.sum(dis[:self.H_Neighbor]) / self.H_Neighbor
            self.d_n_i_vec[i - 1] = res

        return self.d_n_i_vec[i - 1]

    # def delt_star(self):
    #     """
    #     probabilistic factor 概率因子
    #
    #     :return:
    #     """
    #     pf = 0
    #     for i in range(1, self.sample_number + 1):
    #         pf += (1 / self.di(i))
    #
    #     return pf

    # def delt_i(self, i):
    #     """
    #
    #     :param i:
    #     :return:
    #     """
    #     if i == 0:
    #         return 0
    #
    #     return self.delt_i(i - 1) + 1 / self.di(i)

    # def Ri(self, i):
    #     """
    #     interval range
    #
    #     :param i: 从 1 开始
    #     :return:
    #     """
    #     start = self.delt_i(i - 1) / self.delt_star()
    #     end = self.delt_i(i) / self.delt_star()
    #
    #     return start, end

    def Ri_2(self, i):
        """
        interval range

        :param i: 从 1 开始
        :return:
        """
        if len(self.Ri) == 0:
            for i in range(1, len(self.T_p)+1):
                start = self.delt_i_2(i - 1) / self.delt_star_2()
                end = self.delt_i_2(i) / self.delt_star_2()
                self.Ri.append((start, end))

        return self.Ri[i-1]

    def delt_p_i(self, i):

        """

        :param i: 来自正样本集合的下标
        :return:
        """

        if self.delt_p_i_vec[i-1] == 0:
            res = 1 / self.d_p_i(i)
            self.delt_p_i_vec[i - 1] = res

        return self.delt_p_i_vec[i - 1]


    def delt_n_i(self, i):
        """

        :param i: 来自负样本集合的下标
        :return:
        """
        if self.delt_n_i_vec[i-1] == 0:
            res = 1 / self.d_n_i(i)
            self.delt_n_i_vec[i-1] = res

        return self.delt_n_i_vec[i-1]

    def delt_i_2(self, i):
        if i == 0:
            return 0

        if self.delt_i_vec[i-1] == 0:

            res = self.delt_i_2(i - 1) + self.delt_p_i(i) + self.delt_n_i(i)
            self.delt_i_vec[i-1] = res

        return self.delt_i_vec[i-1]

    def delt_star_2(self):
        if self.delt_star is not None:
            return self.delt_star
        else:
            res = 0
            for i in range(1, len(self.T_p) + 1):
                res += self.delt_p_i(i)
            for i in range(1, len(self.T_n) + 1):
                res += self.delt_n_i(i)
            self.delt_star = res

            return res

def train(x, y):
    kf = KFold(n_splits=5, shuffle=True)
    cur_k = 0
    val_history = {}
    val_history["val_acc"] = []
    val_history["val_precision"] = []
    val_history["val_recall"] = []
    val_history["val_f1"] = []
    val_history["auc_value"] = []
    for train_index, val_index in kf.split(x, y):
        cur_k += 1
        x_train, y_train = x[train_index], y[train_index]
        x_val, y_val = x[val_index], y[val_index]

        classifier = KNeighborsClassifier(n_jobs=-1)
        classifier.fit(x_train, y_train)

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

        print(cur_k)
        print("val_acc: %.4f" % val_acc)
        print("val_precision: %.4f" % val_precision)
        print("val_recall: %.4f" % val_recall)
        print("val_f1: %.4f" % val_f1)
        print("auc_value: %.4f" % auc_value)

    # 统计，求平均值和标准差
    for k in val_history.keys():
        print("%s:%.4f ±%.4f" % (k, np.mean(val_history[k]), np.std(val_history[k])))

if __name__ == '__main__':
    print("原数据：")
    x, y = yeast1()
    train(x, y)


    print("经过采样后的数据：")
    x, y = yeast1()
    dbu = DBUOperator(x, y)
    x, y = dbu.get_new_set()
    train(x, y)

    print("随机采样：")
    x, y = yeast1()
    x, y = RandomUnderSampler().fit_resample(x, y)
    train(x, y)
    # print("x.shape", x.shape)
    # print(x)
    # print("y.shape", y.shape)
    # print(y)
