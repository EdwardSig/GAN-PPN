import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # ==================================
    # ml-100k划分测试集和训练集
    # =================================
    # ml_100k_file = '../../data/ml-100k/preprocessing/ml-100k.csv'
    # to_train_file = '../../data/ml-100k/preprocessing/ml-100k-train.csv'
    # to_test_file = '../../data/ml-100k/preprocessing/ml-100k-test.csv'
    # ml_100 = pd.read_csv(ml_100k_file)
    # ml_100k_train, ml_100k_test = train_test_split(ml_100, test_size=0.2, random_state=206)
    # ml_100k_train.to_csv(to_train_file, index=False)
    # ml_100k_test.to_csv(to_test_file, index=False)
    #
    # ml_100k_test = ml_100k_test[ml_100k_test['rating'].isin([4, 5])]
    # ml_100k_test.to_csv('../../data/ml-100k/preprocessing/ml-100k-test_filter.csv', index=False)

    # ===================================
    # douban-music划分测试集和训练集
    # ===================================
    # douban_music_file = '../../data/douban-music/preprocessing/douban-music.csv'
    # to_train_file = '../../data/douban-music/preprocessing/douban-music-train.csv'
    # to_test_file = '../../data/douban-music/preprocessing/douban-music-test.csv'
    # douban_music = pd.read_csv(douban_music_file)
    # douban_music_train, douban_music_test = train_test_split(douban_music, test_size=0.2, random_state=206)
    # douban_music_train.to_csv(to_train_file, index=False)
    # douban_music_test.to_csv(to_test_file, index=False)
    #
    # douban_music_test = douban_music_test[douban_music_test['rating'].isin([4, 5])]
    # douban_music_test.to_csv('../../data/douban-music/preprocessing/douban-music-test_filter.csv', index=False)

    # ===================================
    # douban-book划分测试集和训练集
    # ===================================
    ml_1m_file = '../../data/douban-book/preprocessing/douban-book.csv'
    to_train_file = '../../data/douban-book/preprocessing/douban-book-train.csv'
    to_test_file = '../../data/douban-book/preprocessing/douban-book-test.csv'
    ml_1m = pd.read_csv(ml_1m_file)
    ml_1m_train, ml_1m_test = train_test_split(ml_1m, test_size=0.2, random_state=206)
    ml_1m_train.to_csv(to_train_file, index=False)
    ml_1m_test.to_csv(to_test_file, index=False)

    ml_1m_test = ml_1m_test[ml_1m_test['rating'].isin([4, 5])]
    ml_1m_test.to_csv('../../data/douban-book/preprocessing/douban-book-test_filter.csv', index=False)

