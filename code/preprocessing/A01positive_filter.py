# 过滤正样本交互数小于10的用户
import sys

import pandas as pd

from utils.neg_sampling import mapping_func

if __name__ == '__main__':
    # ====================================
    # ML-100K
    # ====================================
    # ml_100k_file = "../../data/ml-100k/original/u.data"
    # to_ml_100k_file = '../../data/ml-100k/preprocessing/ml-100k.csv'
    # ml_100k_data = pd.read_csv(ml_100k_file, names=['user_id', 'item_id', 'rating', 'timestamp'], sep='\t')
    # # data = pd.read_csv(file)
    # ml_100k_uid_list = ml_100k_data['user_id'].unique()
    # save_uid_list = []      # 保存正样本交互大于等于10的用户
    # for uid in ml_100k_uid_list:
    #     user_ratings = ml_100k_data[ml_100k_data['user_id'] == uid]
    #     positive_rating_count = user_ratings[user_ratings['rating'].isin([4, 5])].shape[0]
    #     if positive_rating_count >= 10:
    #         save_uid_list.append(uid)
    # ml_100k_data2 = ml_100k_data[ml_100k_data['user_id'].isin(save_uid_list)]
    # # 映射ID
    # ml_100k_data2 = mapping_func(ml_100k_data2)
    # ml_100k_data2.to_csv(to_ml_100k_file, index=False)

    # ====================================
    # Douban-Music
    # ====================================
    # douban_music_file = "../../data/douban-music/original/douban-music.csv"
    # to_douban_music_file = '../../data/douban-music/preprocessing/douban-music.csv'
    # douban_music_data = pd.read_csv(douban_music_file)
    # douban_music_uid_list = douban_music_data['user_id'].unique()
    # save_douban_music_uid_list = []  # 保存正样本交互大于等于10的用户
    # for uid in douban_music_uid_list:
    #     user_ratings = douban_music_data[douban_music_data['user_id'] == uid]
    #     positive_rating_count = user_ratings[user_ratings['rating'].isin([4, 5])].shape[0]
    #     if positive_rating_count >= 10:
    #         save_douban_music_uid_list.append(uid)
    # douban_music_data2 = douban_music_data[douban_music_data['user_id'].isin(save_douban_music_uid_list)]
    # # 映射ID
    # douban_music_data2 = mapping_func(douban_music_data2)
    # douban_music_data2.to_csv(to_douban_music_file, index=False)

    # ====================================
    # Douban-book
    # ====================================
    douban_music_file = "../../data/douban-book/original/douban-book.csv"
    to_douban_music_file = '../../data/douban-book/preprocessing/douban-book.csv'
    douban_music_data = pd.read_csv(douban_music_file)
    douban_music_uid_list = douban_music_data['user_id'].unique()
    save_douban_music_uid_list = []  # 保存正样本交互大于等于10的用户
    for uid in douban_music_uid_list:
        user_ratings = douban_music_data[douban_music_data['user_id'] == uid]
        positive_rating_count = user_ratings[user_ratings['rating'].isin([4, 5])].shape[0]
        if positive_rating_count >= 10:
            save_douban_music_uid_list.append(uid)
    douban_music_data2 = douban_music_data[douban_music_data['user_id'].isin(save_douban_music_uid_list)]
    # 映射ID
    douban_music_data2 = mapping_func(douban_music_data2)
    douban_music_data2.to_csv(to_douban_music_file, index=False)



