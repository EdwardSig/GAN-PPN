import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data_to_dict(train_path, test_path, sep):
    """
    首先将训练集和测试集中的数据加载成字典的形式
    :param train_path: 训练集的路径
    :param test_path: 测试集的路径
    :param sep: 数据文件的分割形式
    :return:
    """
    train_set_dict, test_set_dict = {}, {}
    df_train = pd.read_csv(train_path, sep=sep)-1
    df_test = pd.read_csv(test_path, sep=sep)-1
    for item in df_train.itertuples():
        urow, icol, rating = item[1], item[2], item[3]+1
        train_set_dict.setdefault(urow, {}).setdefault(icol, rating)
    for item in df_test.itertuples():
        urow, icol, rating = item[1], item[2], item[3]+1
        test_set_dict.setdefault(urow, {}).setdefault(icol, rating)
    return train_set_dict, test_set_dict


def get_matrix(train_set_dict, test_set_dict, nb_user, nb_item):
    train_matrix, test_matrix = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))
    for u in train_set_dict.keys():
        for i in train_set_dict[u].keys():
            train_matrix[u][i] = train_set_dict[u][i]
    for u in test_set_dict.keys():
        for i in test_set_dict[u]:
            test_matrix[u][i] = test_set_dict[u][i]
    return train_matrix, test_matrix

