import random
import sys

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore
import pickle


def create_item_sim_sparse_matrix(data):
    """
    创建物品相似度稀疏矩阵
    如果太占内存，后续再优化
    :param data:
    :return:
    """
    # 创建用户-物品评分矩阵
    max_user_id, max_item_id = np.max(data['user_id']), np.max(data['item_id'])
    user_item_matrix = np.zeros(shape=(max_user_id, max_item_id), dtype=np.float32)
    for idx, row in data.iterrows():
        rating = float(row['rating'])
        uid = int(row['user_id'])
        iid = int(row['item_id'])
        user_item_matrix[uid - 1, iid - 1] = rating
    # 计算物品间余弦相似度
    item_similarity_matrix = cosine_similarity(user_item_matrix.T)

    item_similarity_matrix = item_similarity_matrix.astype(np.float32)
    item_similarity_sparse_matrix = sparse.csr_matrix(item_similarity_matrix)

    # # 转换成字典
    # item_sim_dict = dict()
    # for row in range(0, len(item_similarity_matrix)):
    #     for col in range(0, len(item_similarity_matrix[row])):
    #         if row not in item_sim_dict:
    #             item_sim_dict[row] = {}
    #         if row <= col:
    #             item_sim_dict[row][col] = item_similarity_matrix[row][col]

    return item_similarity_sparse_matrix


def select_negative_items(batch_history_data, nb_zr, nb_pm):
    '''
    :param history_data:用户和项目交互的信息
    :param nb_zr:zr采样的个数
    :param nb_pm:pm采样的个数
    :return:
    '''
    data = np.array(batch_history_data)
    idx_zr, idx_pm = np.zeros_like(data), np.zeros_like(data)
    for i in range(data.shape[0]):
        # 得到所有为0的项目下标
        items = np.where(data[i] == 0)[0].tolist()
        # 随机抽取一定数量的下标
        tmp_zr = random.sample(items, nb_zr)
        tmp_pm = random.sample(items, nb_pm)
        # 这些位置的值为1
        idx_zr[i][tmp_zr] = 1
        idx_pm[i][tmp_pm] = 1
    return idx_zr, idx_pm


def mapping_func(data):
    # ID值映射
    uid2internal, internal2uid = dict(), dict()
    iid2internal, internal2iid = dict(), dict()
    uid_list = data['user_id'].unique()
    iid_list = data['item_id'].unique()
    for i, uid in enumerate(uid_list):
        uid2internal[int(uid)] = i + 1
        internal2uid[i + 1] = int(uid)
    for i, iid in enumerate(iid_list):
        iid2internal[int(iid)] = i + 1
        internal2iid[i + 1] = int(iid)
    data['user_id'] = data['user_id'].map(uid2internal)
    data['item_id'] = data['item_id'].map(iid2internal)
    return data


def create_item_similarity_matrix(data_file_path, save_file_path, mapping_file_path, dataset):
    """
    该方法做了以下几件事情：
    1. 把没有大于等于4的评分的用户删除
    2. 将原文件做了用户ID和物品ID做了映射处理
    3. 构建了物品相似度矩阵
    4. 构建了一个仅保存大于等于4分值的用户物品评分矩阵
    5. 保存3中物品的相似度矩阵
    :param data_file_path: 数据集文件路径 (str)
    :param save_file_path:  物品相似度文件保存路径(str)
    :param mapping_file_path:  用户ID和物品ID映射过的文件(str)
    :param dataset:  数据集类型(str): [ml-100k, ml-latest, ciao]
    :return: 物品相似度矩阵(numpy), 4中用户物品评分矩阵(numpy),映射处理后的data(pandas)
    """

    #####################################
    # 由于在数据集中有些用户只会给物品打低分，这个时候没有所谓大于4的正样本，因此要把这样的用户删除
    ####################################
    # 读取数据文件
    if dataset == 'ml-100k':
        data = pd.read_csv(data_file_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    elif dataset == 'ml-latest':
        data = pd.read_csv(data_file_path)
        data = data.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
    elif dataset == 'ciao':
        data = pd.read_csv(data_file_path,
                           names=['user_id', 'item_id', 'movie_category_id', 'review_id', 'rating', 'datetime'],
                           parse_dates=['datetime'])
        data['timestamp'] = data.datetime.values.astype(np.int64) // 10 ** 9
        data = data[['user_id', 'item_id', 'rating', 'timestamp']]
    elif dataset == 'douban-music':
        data = pd.read_csv(data_file_path)
    else:
        print(Fore.RED + '程序错误，请输入正确的数据集!!!')
        sys.exit(1)
    # 找出评分大于等于4的用户ID并去重
    high_rating_users = data[data['rating'] >= 4]['user_id'].unique()
    # 找出评分小于4的用户ID并去重
    low_rating_users = data[data['rating'] < 4]['user_id'].unique()
    # 如果low_rating_users中的user_id没有出现在high_rating_users中，则将该user_id在数据集中的所有数据都删除
    for user_id in low_rating_users:
        if user_id not in high_rating_users:
            data = data[data['user_id'] != user_id]

    ###################################
    # 为保证数据集中从1-到最大ID值均有用户或物品的评分操作，将用户和物品ID做了映射
    ###################################
    data = mapping_func(data)
    # 保存筛选后的数据到新文件中
    data.to_csv(mapping_file_path, index=False)

    #################################
    # 该物品相似度矩阵基于全数据的用户物品评分矩阵构建（包含用户的低值评分）
    ################################
    # 创建用户-物品评分矩阵
    max_user_id, max_item_id = np.max(data['user_id']), np.max(data['item_id'])
    user_item_matrix = np.zeros(shape=(max_user_id, max_item_id))
    for idx, row in data.iterrows():
        rating = float(row['rating'])
        uid = int(row['user_id'])
        iid = int(row['item_id'])
        user_item_matrix[uid - 1, iid - 1] = rating
    # 计算物品间余弦相似度
    item_similarity_matrix = cosine_similarity(user_item_matrix.T)

    #################################
    # 该用户物品评分矩阵中仅把大于等于4的作为正样本
    ################################
    # 将用户-物品评分矩阵中的值进行处理，将评分小于4的项置为0，而评分大于等于4的项保持原值。
    user_item_matrix_new = user_item_matrix
    for uid in range(0, user_item_matrix.shape[0]):
        for iid in range(0, user_item_matrix.shape[1]):
            if user_item_matrix[uid, iid] < 4:
                user_item_matrix_new[uid, iid] = 0

    #################################
    # 保存物品相似度矩阵
    ################################
    # 保存物品间相似度矩阵为 .pkl 文件
    with open(save_file_path, 'wb') as f:
        pickle.dump(item_similarity_matrix, f)

    print("Item similarity matrix saved successfully.")
    return item_similarity_matrix, user_item_matrix_new, data


# 取出该用户对所有item评分小于4的为missing_items样本  >=4的为positive_samples样本
def find_missing_items(user_row, user_item_matrix_new):
    """
    1. 找到某个用户的所有未评分的样本
    2. 计算正样本和未评分样本的长度
    :param user_id: 用户ID
    :param user_item_matrix_new: 评分大于4的正样本的矩阵
    :return: 一个用户的所有大于等于4的正样本(numpy), 一个用户的所有未评分的样本(numpy),正样本数量(int),未评分样本数量(int)
    """
    ####################################
    # 寻找用户ID为user_id的用户的未评分过的样本
    ####################################
    positive_col_samples = np.where(user_item_matrix_new[int(user_row)] > 0)[0]
    all_col_items = np.arange(user_item_matrix_new.shape[1])
    missing_col_items = np.setdiff1d(all_col_items, positive_col_samples)

    #########################################
    # 计算正负样本数量
    #########################################
    poslens = len(positive_col_samples)
    mislens = len(missing_col_items)

    return positive_col_samples, missing_col_items, poslens, mislens


def find_top_n_negative_items(positive_col_samples, missing_col_items, item_similarity_matrix, top_n):
    """
    根据相似度找到前TopN个负样本
    :param positive_col_samples: 大于等于4的正样本列表
    :param missing_col_items: 未评分过的样本列表
    :param item_similarity_matrix: 物品相似度矩阵
    :param top_n:
    :return: 前TopN个相似度最高的负样本
    """
    one_user_nega_list = []
    for positive_col in positive_col_samples:
        negative_dict = {}
        for missing_col in missing_col_items:
            similarity = item_similarity_matrix[missing_col, positive_col]
            negative_dict[missing_col + 1] = similarity

        one_item_nega_list = sorted(negative_dict.items(), key=lambda x: x[1], reverse=True)
        one_item_nega_list = np.array(one_item_nega_list)[:top_n].tolist()
        one_user_nega_list.extend(one_item_nega_list)

    one_user_nega_list = sorted(one_user_nega_list, key=lambda x: x[1], reverse=True)

    one_user_nega_topn_list = []
    i = 0
    for nega in one_user_nega_list:
        if nega[0] not in one_user_nega_topn_list:
            one_user_nega_topn_list.append(nega[0])
            i += 1
        if i == top_n:
            break
    return one_user_nega_topn_list


def find_ratio_negative_items(positive_col_samples, missing_col_items, item_similarity_matrix, poslens, ratio):
    """
    按比例找出用户的负样本
    :param positive_samples: 某个用户大于等于4的正样本
    :param missing_items: 某个用户所有未交互过的物品
    :param item_similarity_matrix: 物品相似度矩阵
    :param poslens: 大于等于4的正样本的长度
    :param ratio: 负样本的比例，最后的负样本数=正样本数(poslens) * ratio
    :return: 根据相似度降序的未交互的物品列表(numpy)
    """

    one_user_nega_ratio_list = []
    for positive_col in positive_col_samples:
        negative_dict = {}
        for missing_col in missing_col_items:
            similarity = item_similarity_matrix[missing_col, positive_col]
            negative_dict[missing_col + 1] = similarity

        one_item_nega_ratio_list = sorted(negative_dict.items(), key=lambda x: x[1], reverse=True)
        one_item_nega_ratio_list = np.array(one_item_nega_ratio_list)[:ratio * poslens].tolist()
        one_user_nega_ratio_list.extend(one_item_nega_ratio_list)

    one_user_nega_ratio_list = sorted(one_user_nega_ratio_list, key=lambda x: x[1], reverse=True)

    one_user_nega_ratio_fina_list = []
    i = 0
    for nega in one_user_nega_ratio_list:
        if nega[0] not in one_user_nega_ratio_fina_list:
            one_user_nega_ratio_fina_list.append(nega[0])
            i += 1
        if i == (ratio * poslens):
            break
    return one_user_nega_ratio_fina_list
