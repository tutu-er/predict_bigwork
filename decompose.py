import numpy as np
import matplotlib.pyplot as plt
import csv

import pandas as pd
from vmdpy import VMD

from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import orthogonal_mp
from sklearn.datasets import make_sparse_coded_signal
from sklearn.metrics import mean_squared_error as MSE
from scipy.optimize import nnls

from data import Data

class KSVD:
    def __init__(self, data, n_components = 20, max_iteration = 20, max_iteration_svd = 3,
                 focuss_lambda_max = 1e3, focuss_p = 1, focuss_max_iter = 200, focuss_eps = 1e-6):

        self.X = data
        self.n_components = n_components
        self.max_iteration = max_iteration
        self.max_iteration_svd = max_iteration_svd
        self.D = self.random_init_D()
        # self.A = self.data_focuss(self.D, self.X)
        self.focuss_lambda_max = focuss_lambda_max
        self.focuss_p = focuss_p
        self.focuss_max_iter = focuss_max_iter
        self.focuss_eps = focuss_eps

    def random_init_D(self):
        # D = np.random.rand(self.n_components-1, self.X.shape[1])
        # D = np.concatenate((D, np.ones((1, self.X.shape[1])) * 0.5), axis=0)
        # return D
        return np.random.rand(self.n_components, self.X.shape[1])

    def fit(self, iter_focuss=False, every_focuss = 50):
        A = self.data_focuss(self.D, self.X)
        for j in range(self.max_iteration):
            if iter_focuss and j > 0 :
                if j % every_focuss == 0:
                    A = self.data_focuss(self.D, self.X)
            print(f"iter:{j} MSE:{MSE(self.X, A @ self.D)}")
            for k in range(self.n_components):
                non_zeros_set = A[:, k] > 1e-8
                if any(non_zeros_set):
                    E_mat = self.X - A @ self.D + A[:, k].reshape(-1, 1) @ self.D[k, :].reshape(1, -1)
                    E_mat_restricted = E_mat[non_zeros_set, :]
                    U, s, VT = np.linalg.svd(E_mat_restricted)

                    a = U[:, 0]
                    d = VT[0, :]

                    if np.sum(a > 0) < np.sum(a < 0):
                        a = -a
                        d = -d

                    a[a < 0] = 0
                    d[d < 0] = 0

                    a = a.reshape(1, -1)
                    d = d.reshape(-1, 1)

                    for _ in range(self.max_iteration_svd):
                        d = (a @ E_mat_restricted) / (a @ a.T)
                        if np.sum(d > 0) < np.sum(d < 0):
                            a = -a
                            d = -d
                        d[d < 0] = 0
                        d = d.reshape(-1, 1)
                        a = (E_mat_restricted @ d) / (d.T @ d)
                        if np.sum(a > 0) < np.sum(a < 0):
                            a = -a
                            d = -d
                        a[a < 0] = 0
                        a = a.reshape(1, -1)

                    d = d/np.abs(s[0])
                    a = a*np.abs(s[0])

                    self.D[k, :] = d.T
                    A[non_zeros_set, k] = a.T.reshape(-1)

        return self.D, A

    def load_and_fit(self, A, iter_times=100, iter_focuss=False, every_focuss = 50):
        if iter_focuss:
            A = self.data_focuss(self.D, self.X)
        for j in range(iter_times):
            if iter_focuss and j > 0 :
                if j % every_focuss == 0:
                    A = self.data_focuss(self.D, self.X)
            print(f"iter:{j} MSE:{MSE(self.X, A @ self.D)}")
            for k in range(self.n_components):
                non_zeros_set = A[:, k] > 1e-8
                if any(non_zeros_set):
                    E_mat = self.X - A @ self.D + A[:, k].reshape(-1, 1) @ self.D[k, :].reshape(1, -1)
                    E_mat_restricted = E_mat[non_zeros_set, :]
                    U, s, VT = np.linalg.svd(E_mat_restricted)

                    a = U[:, 0]
                    d = VT[0, :]

                    if np.sum(a > 0) < np.sum(a < 0):
                        a = -a
                        d = -d

                    a[a < 0] = 0
                    d[d < 0] = 0

                    a = a.reshape(1, -1)
                    d = d.reshape(-1, 1)

                    for _ in range(self.max_iteration_svd):
                        d = (a @ E_mat_restricted) / (a @ a.T)
                        if np.sum(d > 0) < np.sum(d < 0):
                            a = -a
                            d = -d
                        d[d < 0] = 0
                        d = d.reshape(-1, 1)
                        a = (E_mat_restricted @ d) / (d.T @ d)
                        if np.sum(a > 0) < np.sum(a < 0):
                            a = -a
                            d = -d
                        a[a < 0] = 0
                        a = a.reshape(1, -1)

                    d = d/np.abs(s[0])
                    a = a*np.abs(s[0])

                    self.D[k, :] = d.T
                    A[non_zeros_set, k] = a.T.reshape(-1)

        return self.D, A

    def data_focuss(self, D, X):
        A = np.zeros((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            A[i, :] = self.focuss_plus(D.T, X[i, :].T, lambda_max=self.focuss_lambda_max,
                                       p=self.focuss_p, max_iter=self.focuss_max_iter, eps=self.focuss_eps)
        
        return A

    def focuss_plus(self, A, y, lambda_max=1e5, p=1, max_iter=100, eps=1e-6):
        x = np.linalg.pinv(A) @ y  # 初始化解2
        prev_x = x.copy()

        for _ in range(max_iter):
            abs_x = np.abs(x)
            abs_x[abs_x < 1e-10] = 1e-10
            W = np.diag(abs_x ** (2 - p))
            W[W < 1e-8] = 0
            lambda_k = np.abs(lambda_max * (1 - np.linalg.norm(y - A @ x) / np.linalg.norm(y) ) )
            x = W @ A.T @ np.linalg.inv(lambda_k * np.eye(A.shape[0]) + A @ W @ A.T) @ y
            x[x < 1] = 0
            if np.linalg.norm(x - prev_x) < eps:
                break
            prev_x = x.copy()

        return x

if __name__ == "__main__":

    data = Data('STLF_DATA_IN_1.xls')

    """ VMD """
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t)

    # VMD 参数设置
    alpha = 2000  # 模态带宽约束（影响模态中心频率的紧密性）
    tau = 0.0  # 噪声容忍度（通常设为 0）
    K = 2  # 分解的模态数
    DC = 0  # 是否包含直流分量（0 表示不包含）
    init = 1  # 初始化方法（1 为均匀分布）
    tol = 1e-7  # 收敛容差

    # 执行 VMD
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

    # plt.plot(u[0, :])
    # plt.plot(u[1, :])
    # plt.plot(u[0, :] + u[1,:])
    # plt.plot(signal)
    #
    # plt.show()

    """ DictionaryLearning """
    # n_components = 20  # 字典原子数
    #
    # X = data.np_day_pd_96[data.row_all_right, :]
    #
    # # 初始化字典学习模型（使用OMP稀疏编码）
    # dict_learner = DictionaryLearning(
    #     n_components=n_components,
    #     alpha=0.1,  # 稀疏性约束（L1正则化系数）
    #     max_iter=100,  # 最大迭代次数
    #     fit_algorithm='cd',  # 坐标下降法求解稀疏编码
    #     transform_algorithm='omp',  # 正交匹配追踪
    #     random_state=42
    # )
    #
    # # 训练字典
    # D = dict_learner.fit(X).components_
    #
    # gamma = orthogonal_mp(D.T, X.T, n_nonzero_coefs=5).T

    # X_reconstructed = gamma @ D

    # for i in range(n_components):
    #     plt.plot(D[i,:])

    # plt.plot(D[1, :])
    # plt.show()
    # 输出字典和稀疏编码
    # print("学习到的字典形状:", D.shape)

    """ K-SVD """
    row_normal = [i for i in range(data.days) if i in data.row_all_right and not i in data.weekend_and_holiday]

    # X = data.np_day_pd_96[row_normal, :]
    X = data.np_day_pd_96[data.row_all_right, :]

    n_components = 10
    max_iteration = 1200
    max_iteration_svd = 2

    ksvd_model = KSVD(
        data = X,
        n_components = n_components,
        max_iteration = max_iteration,
        max_iteration_svd = max_iteration_svd,
        focuss_lambda_max = 1e5,
        focuss_p = 1,
        focuss_max_iter = 200,
        focuss_eps = 1e-6
    )
    D, A = ksvd_model.fit(iter_focuss=True, every_focuss=300)

    with open(f'D_ksvd_{n_components}_normal.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in D:
            writer.writerow(row)

    with open(f'A_ksvd_{n_components}_normal.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in A:
            writer.writerow(row)

    D = pd.read_csv(f'D_ksvd_{n_components}_normal.csv', header=None).to_numpy()
    A = pd.read_csv(f'A_ksvd_{n_components}_normal.csv', header=None).to_numpy()

    # ksvd_model.D = D
    # D, A = ksvd_model.load_and_fit(A, iter_times = 2, iter_focuss= True, every_focuss = 1000)
    #
    # with open(f'D_ksvd_{n_components}_test.csv', mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     for row in D:
    #         writer.writerow(row)
    #
    # with open(f'A_ksvd_{n_components}_test.csv', mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     for row in A:
    #         writer.writerow(row)

    for i in range(n_components):
        plt.plot(D[i,:])

    plt.plot(D[1, :])
    plt.show()
    print("学习到的字典形状:", D.shape)

    pass



