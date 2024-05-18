'''
GAN-PPN-P runs on Douban-Music
'''
import math
import random

import numpy as np
import pandas as pd
import torch

from load_data import load_data_to_dict, get_matrix
from train import train_vgan2_non_pop_non_negative_pref


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    train_file = '../data/douban-music/preprocessing/douban-music-train.csv'
    test_file = '../data/douban-music/preprocessing/douban-music-test_filter.csv'
    sep = ','
    train_df = pd.read_csv(train_file, sep=sep)
    test_df = pd.read_csv(test_file, sep=sep)
    df = pd.concat([train_df, test_df])

    nb_user = np.max(df['user_id'].unique())
    nb_item = np.max(df['item_id'].unique())
    top_k = 20
    delta = 0.2
    S_POS, S_NEG, S_TOPN_NEG = 200, 500, 800
    pop_groups = 10
    neg_dis_size = nb_item
    pos_dis_size = nb_item
    discriminator_vector_dim = 64
    generator_vector_dim = 64
    batch_size = 512
    epoch = 200
    alpha_pos, alpha_neg = 0.1, 0.1
    alpha_pos_zr, alpha_neg_zr, alpha_topn_neg_zr = 0.05, 0.05, 0.2
    hard_ratio = 0.4

    item_col_pop_list = [0] * nb_item
    for idx, row in train_df.iterrows():
        iid = int(row['item_id'])
        item_col_pop_list[iid - 1] += 1
    item_col_pop_list = np.array(item_col_pop_list)
    item_col_pop_list = (item_col_pop_list - np.min(item_col_pop_list)) / (np.max(item_col_pop_list) - np.min(item_col_pop_list))

    item_col_category_by_pop = dict()
    for i in range(1, pop_groups+1):
        item_col_category_by_pop[i] = []
    for col, pop in enumerate(item_col_pop_list):
        if pop == 1:
            item_col_category_by_pop[pop_groups].append(col)
        else:
            g = math.floor(pop * pop_groups) + 1
            item_col_category_by_pop[g].append(col)

    user_rows2item_cols_pos = dict()
    user_rows2item_cols_neg = dict()
    for uid in train_df['user_id'].unique():
        user_rows2item_cols_pos[int(uid) - 1] = []
        user_rows2item_cols_neg[int(uid) - 1] = []
    for idx, row in train_df.iterrows():
        uid = int(row['user_id'])
        iid = int(row['item_id'])
        rating = float(row['rating'])
        if rating >= 4:
            user_rows2item_cols_pos[uid - 1].append(iid - 1)
        if rating <= 2:
            user_rows2item_cols_neg[uid - 1].append(iid - 1)

    positive_samples_mean_std_list = []
    for uid in df['user_id'].unique():
        positive_samples_mean_std_list.append([0.0, 1.0])

    negative_samples_mean_std_list = []
    for uid in df['user_id'].unique():
        negative_samples_mean_std_list.append([0.0, 1.0])

    print('='*50, '数据预处理结束', '='*50)

    setup_seed(2024)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_set_dict, test_set_dict = load_data_to_dict(train_file, test_file, sep=sep)

    train_matrix, test_matrix = get_matrix(train_set_dict, test_set_dict, nb_user=nb_user, nb_item=nb_item)
    positive_distribution_item_cols, pur = [], 0
    for mean, std in positive_samples_mean_std_list:
        sample_pop = np.random.normal(loc=mean, scale=std, size=S_POS)
        tmp = []
        for sp in sample_pop:
            g = 1
            if 0 <= sp < 1:
                g = math.floor(sp * pop_groups) + 1
            if sp == 1:
                g = pop_groups
            idx = np.random.randint(0, len(item_col_category_by_pop[g]))

            if item_col_category_by_pop[g][idx] not in train_set_dict[pur]:
                tmp.append(item_col_category_by_pop[g][idx])
        positive_distribution_item_cols.append(tmp)
        pur += 1

    negative_distribution_item_cols, nur = [], 0
    for mean, std in negative_samples_mean_std_list:
        sample_pop = np.random.normal(loc=mean, scale=std, size=S_NEG)
        tmp = []
        for sp in sample_pop:
            g = 1
            if 0 <= sp < 1:
                g = math.floor(sp * pop_groups) + 1
            if sp == 1:
                g = pop_groups
            idx = np.random.randint(0, len(item_col_category_by_pop[g]))
            if item_col_category_by_pop[g][idx] not in train_set_dict[nur]:
                tmp.append(item_col_category_by_pop[g][idx])
        negative_distribution_item_cols.append(tmp)
        nur += 1

    pos_length = [len(l) for l in positive_distribution_item_cols]
    neg_length = [len(l) for l in negative_distribution_item_cols]
    min_pos_length = np.min(pos_length)
    min_neg_length = np.min(neg_length)
    positive_distribution_item_cols_copy = positive_distribution_item_cols.copy()
    negative_distribution_item_cols_copy = negative_distribution_item_cols.copy()
    positive_distribution_item_cols = [cols[: min_pos_length] for cols in positive_distribution_item_cols_copy]
    negative_distribution_item_cols = [cols[: min_neg_length] for cols in negative_distribution_item_cols_copy]
    train_matrix = torch.FloatTensor(train_matrix).to(DEVICE)
    positive_distribution_item_cols = torch.IntTensor(positive_distribution_item_cols)
    positive_distribution_item_cols = positive_distribution_item_cols.to(DEVICE)
    negative_distribution_item_cols = torch.IntTensor(negative_distribution_item_cols)
    negative_distribution_item_cols = negative_distribution_item_cols.to(DEVICE)
    item_col_pop_list = torch.FloatTensor(item_col_pop_list)
    item_col_pop_list = item_col_pop_list.to(DEVICE)

    vgan_train_parameter = {
        'train_matrix': train_matrix,
        'train_set_dict': train_set_dict,
        'positive_distribution_item_cols': positive_distribution_item_cols,
        'negative_distribution_item_cols': negative_distribution_item_cols,
        'positive_samples_mean_std_list': positive_samples_mean_std_list,
        'negative_samples_mean_std_list': negative_samples_mean_std_list,
        'item_col_pop_list': item_col_pop_list,
        'nb_user': nb_user,
        'nb_item': nb_item,
        'S_POS': S_POS,
        'S_NEG': S_NEG,
        'S_TOPN_NEG': S_TOPN_NEG,
        'pos_dis_size': pos_dis_size,
        'neg_dis_size': neg_dis_size,
        'discriminator_vector_dim': discriminator_vector_dim,
        'generator_vector_dim': generator_vector_dim,
        'alpha_pos': alpha_pos,
        'alpha_neg': alpha_neg,
        'alpha_pos_zr': alpha_pos_zr,
        'alpha_neg_zr': alpha_neg_zr,
        'alpha_topn_neg_zr': alpha_topn_neg_zr,
        'hard_ratio': hard_ratio,
        'device': DEVICE
    }
    print('=' * 50, '加载数据结束', '=' * 50)

    print('=' * 50, '开始训练模型', '=' * 50)
    train_vgan2_non_pop_non_negative_pref(vgan_train_parameter, epoches=epoch, batch_size=batch_size, nb_zr=128, nb_pm=128,
               alpha_pos=alpha_pos, alpha_neg=alpha_neg, test_set_dict=test_set_dict, top_k=top_k)
