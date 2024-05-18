import random

import numpy as np
import torch

from models import VGANNonPop, VGANNonNegativePref, VGANNonPositivePref
from preprocessing.utils.evaluate import test, non_positive_pref_test


def train_vgan2_non_pop(train_parameter, epoches, batch_size, nb_zr, nb_pm, alpha_pos, alpha_neg, test_set_dict, top_k):
    epoche_list, precision_list, recall_list, ndcg_list = [], [], [], []
    epoche_list2, precision_list2, recall_list2, ndcg_list2 = [], [], [], []
    vgan = VGANNonPop(
        train_parameter['nb_user'],
        train_parameter['nb_item'],
        train_parameter['pos_dis_size'],
        train_parameter['neg_dis_size'],
        train_parameter['discriminator_vector_dim'],
        train_parameter['generator_vector_dim'],
        train_parameter['device']
    )

    vgan.G.to(train_parameter['device'])
    vgan.D.to(train_parameter['device'])
    gen_opt = torch.optim.Adam(vgan.G.parameters(), lr=0.0001)
    dis_opt = torch.optim.Adam(vgan.D.parameters(), lr=0.0001)
    step_gen, step_dis = 5, 2
    for e in range(epoches):
        vgan.D.train()
        vgan.G.eval()
        for step in range(step_dis):
            idxs = random.sample(range(len(train_parameter['train_matrix'])), batch_size)
            data = train_parameter['train_matrix'][idxs]
            pos_distribution_idx_pm = torch.zeros_like(data)
            pos_distribution_data = torch.zeros_like(data)
            for r, i_cols in enumerate(train_parameter['positive_distribution_item_cols'][idxs]):
                pos_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
                pos_distribution_data[[r] * len(i_cols), i_cols] = 5

            neg_distribution_idx_pm = torch.zeros_like(data)
            for r, i_cols in enumerate(train_parameter['negative_distribution_item_cols'][idxs]):
                neg_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1

            eu = torch.zeros_like(data)
            rows, cols = torch.nonzero(data, as_tuple=True)
            eu[rows, cols] = 1

            G_predict_purchase, g_z = vgan.G(data, pos_distribution_data, vgan.pos_distribution_vector[idxs])
            G_predict_purchase = G_predict_purchase * (eu + pos_distribution_idx_pm)
            D_data_out, d_z1 = vgan.D(data, neg_distribution_idx_pm, vgan.neg_distribution_vector[idxs])
            D_purchase_out, d_z2 = vgan.D(G_predict_purchase, neg_distribution_idx_pm,
                                          vgan.neg_distribution_vector[idxs])

            loss1 = -torch.mean(torch.log(D_data_out))
            loss2 = -torch.mean(torch.log(1. - D_purchase_out))
            bce = -torch.mean(torch.sum(torch.nn.functional.log_softmax(d_z1, 1) * neg_distribution_idx_pm, -1))
            kld = - 0.5 * torch.mean(torch.sum(1 + vgan.D.std_vector(neg_distribution_idx_pm) - torch.square(
                vgan.D.mean_vector(neg_distribution_idx_pm)) - torch.exp(vgan.D.std_vector(neg_distribution_idx_pm)),
                                               dim=1))
            loss_dis = alpha_neg * (bce + kld)

            loss = loss1 + loss2 + loss_dis
            dis_opt.zero_grad()
            loss.backward()
            dis_opt.step()

        vgan.D.eval()
        whole_neg_distribution_idx_pm = torch.zeros_like(train_parameter['train_matrix'])
        for r, i_cols in enumerate(train_parameter['negative_distribution_item_cols']):
            whole_neg_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
        new_d_z = vgan.D.get_z_distribution(whole_neg_distribution_idx_pm, vgan.neg_distribution_vector)
        whole_mean, whole_std = torch.mean(new_d_z, dim=1), torch.std(new_d_z, dim=1)
        r = 0
        for mean, std in zip(whole_mean, whole_std):
            train_parameter['negative_samples_mean_std_list'][r][0] = mean
            train_parameter['negative_samples_mean_std_list'][r][1] = std
            r += 1

        vgan.G.train()
        vgan.D.eval()
        for step in range(step_gen):
            idxs = random.sample(range(len(train_parameter['train_matrix'])), batch_size)
            data = train_parameter['train_matrix'][idxs]

            pos_distribution_idx_pm = torch.zeros_like(data)
            for r, i_cols in enumerate(train_parameter['positive_distribution_item_cols'][idxs]):
                pos_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1

            neg_distribution_idx_pm = torch.zeros_like(data)
            for r, i_cols in enumerate(train_parameter['negative_distribution_item_cols'][idxs]):
                neg_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1

            eu = torch.zeros_like(data)
            rows, cols = torch.nonzero(data, as_tuple=True)
            eu[rows, cols] = 1

            G_predict_purchase, g_z = vgan.G(data, pos_distribution_idx_pm * 5, vgan.pos_distribution_vector[idxs])
            G_predict_purchase_total = G_predict_purchase * (eu + pos_distribution_idx_pm + neg_distribution_idx_pm)
            D_purchase_out, d_z = vgan.D(G_predict_purchase_total, neg_distribution_idx_pm,
                                         vgan.neg_distribution_vector[idxs])

            loss1 = torch.mean(torch.log(1. - D_purchase_out))
            loss_pos_zr = train_parameter['alpha_pos_zr'] * torch.sum(
                (G_predict_purchase * pos_distribution_idx_pm - data).pow(2).sum(dim=-1) / pos_distribution_idx_pm.sum(
                    -1))
            loss_neg_zr = train_parameter['alpha_neg_zr'] * torch.sum(
                (G_predict_purchase * neg_distribution_idx_pm - data).pow(2).sum(dim=-1) / neg_distribution_idx_pm.sum(
                    -1))
            bce = -torch.mean(torch.sum(torch.nn.functional.log_softmax(g_z, 1) * pos_distribution_idx_pm, -1))
            kld = - 0.5 * torch.mean(torch.sum(1 + vgan.G.std_vector(pos_distribution_idx_pm * 5) - torch.square(
                vgan.G.mean_vector(pos_distribution_idx_pm * 5)) - torch.exp(
                vgan.G.std_vector(pos_distribution_idx_pm * 5)), dim=1))
            loss_dis = alpha_pos * (bce + kld)

            loss = loss1 + loss_pos_zr + loss_neg_zr + loss_dis
            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()

        vgan.G.eval()
        whole_pos_distribution_idx_pm = torch.zeros_like(train_parameter['train_matrix'])
        for r, i_cols in enumerate(train_parameter['positive_distribution_item_cols']):
            whole_pos_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
        new_g_z = vgan.G.get_z_distribution(whole_pos_distribution_idx_pm * 5, vgan.pos_distribution_vector)
        whole_mean, whole_std = torch.mean(new_g_z, dim=1), torch.std(new_g_z, dim=1)
        r = 0
        for mean, std in zip(whole_mean, whole_std):
            train_parameter['positive_samples_mean_std_list'][r][0] = mean
            train_parameter['positive_samples_mean_std_list'][r][1] = std
            r += 1

        G_predict_purchase, g_z = vgan.G(train_parameter['train_matrix'], whole_pos_distribution_idx_pm * 5,
                                         vgan.pos_distribution_vector)
        mean_score_each_users = torch.mean(G_predict_purchase, dim=1)

        tmp_negative_distribution_item_cols = []
        tmp_positive_distribution_item_cols = []

        for i, one_purchase in enumerate(G_predict_purchase):
            ascending_sort_arg = torch.argsort(one_purchase)

            low_pop = train_parameter['negative_samples_mean_std_list'][i][0] - 2 * \
                      train_parameter['negative_samples_mean_std_list'][i][1]
            high_pop = train_parameter['negative_samples_mean_std_list'][i][0] + 2 * \
                       train_parameter['negative_samples_mean_std_list'][i][1]
            low_top_idxs = ascending_sort_arg[
                           0: int(len(ascending_sort_arg) * train_parameter['hard_ratio'])].detach().tolist()
            low_top_idxs_set = set(low_top_idxs)
            mask1 = (train_parameter['item_col_pop_list'] >= low_pop) & (
                        train_parameter['item_col_pop_list'] <= high_pop)
            neg_idx_list = torch.where(mask1)[0].tolist()
            neg_idx_set = set(neg_idx_list)
            co_occurrence_list = list(neg_idx_set & low_top_idxs_set)
            tmp_negative_distribution_item_cols.append(co_occurrence_list)

            high_top_idxs = ascending_sort_arg[
                            int(len(ascending_sort_arg) * (1 - train_parameter['hard_ratio'])):].detach().tolist()
            high_top_idxs_set = set(high_top_idxs)
            low_pop = train_parameter['positive_samples_mean_std_list'][i][0] - 2 * \
                      train_parameter['positive_samples_mean_std_list'][i][1]
            high_pop = train_parameter['positive_samples_mean_std_list'][i][0] + 2 * \
                       train_parameter['positive_samples_mean_std_list'][i][1]
            mask2 = (train_parameter['item_col_pop_list'] >= low_pop) & (
                        train_parameter['item_col_pop_list'] <= high_pop)
            pos_idx_list = torch.where(mask2)[0].tolist()
            pos_idx_set = set(pos_idx_list)
            co_occurrence_list = list(high_top_idxs_set & pos_idx_set)
            tmp_positive_distribution_item_cols.append(co_occurrence_list)

        tmp_pos_length = [len(l) for l in tmp_positive_distribution_item_cols]
        tmp_neg_length = [len(l) for l in tmp_negative_distribution_item_cols]
        min_pos_length = np.min(tmp_pos_length)
        min_neg_length = np.min(tmp_neg_length)
        positive_distribution_item_cols_copy = tmp_positive_distribution_item_cols.copy()
        negative_distribution_item_cols_copy = tmp_negative_distribution_item_cols.copy()
        positive_distribution_item_cols = [cols[: min_pos_length] for cols in positive_distribution_item_cols_copy]
        negative_distribution_item_cols = [cols[: min_neg_length] for cols in negative_distribution_item_cols_copy]

        train_parameter['positive_distribution_item_cols'] = torch.IntTensor(positive_distribution_item_cols).to(
            train_parameter['device'])
        train_parameter['negative_distribution_item_cols'] = torch.IntTensor(negative_distribution_item_cols).to(
            train_parameter['device'])

        if (e + 1) % 1 == 0:
            print(e + 1, '\t', '==' * 24)
            (precision, precision2), (recall, recall2), (ndcg, ndcg2) = test(vgan, test_set_dict,
                                                                             train_parameter['train_matrix'],
                                                                             train_parameter[
                                                                                 'positive_distribution_item_cols'],
                                                                             top_k=top_k)
            epoche_list.append(e + 1)
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            precision_list2.append(precision2)
            recall_list2.append(recall2)
            ndcg_list2.append(ndcg2)
    max_idx = np.array(recall_list).argmax()
    max_precision, max_recall, max_ndcg = precision_list[max_idx], recall_list[max_idx], ndcg_list[max_idx]
    max_idx = np.array(recall_list2).argmax()
    max_precision2, max_recall2, max_ndcg2 = precision_list2[max_idx], recall_list2[max_idx], ndcg_list2[max_idx]
    print('MAX_P@{0}: {1}, MAX_R@{0}: {2}, MAX_NDCG@{0}: {3}'.format(top_k, max_precision, max_recall, max_ndcg))
    print('MAX_P@{0}: {1}, MAX_R@{0}: {2}, MAX_NDCG@{0}: {3}'.format(50, max_precision2, max_recall2, max_ndcg2))


def train_vgan2_non_pop_non_negative_pref(train_parameter, epoches, batch_size, nb_zr, nb_pm, alpha_pos, alpha_neg,
                                          test_set_dict, top_k):
    epoche_list, precision_list, recall_list, ndcg_list = [], [], [], []
    epoche_list2, precision_list2, recall_list2, ndcg_list2 = [], [], [], []
    vgan = VGANNonNegativePref(
        train_parameter['nb_user'],
        train_parameter['nb_item'],
        train_parameter['pos_dis_size'],
        train_parameter['neg_dis_size'],
        train_parameter['discriminator_vector_dim'],
        train_parameter['generator_vector_dim'],
        train_parameter['device']
    )

    vgan.G.to(train_parameter['device'])
    vgan.D.to(train_parameter['device'])
    gen_opt = torch.optim.Adam(vgan.G.parameters(), lr=0.0001)
    dis_opt = torch.optim.Adam(vgan.D.parameters(), lr=0.0001)
    step_gen, step_dis = 5, 2
    for e in range(epoches):

        vgan.D.train()
        vgan.G.eval()
        for step in range(step_dis):
            idxs = random.sample(range(len(train_parameter['train_matrix'])), batch_size)
            data = train_parameter['train_matrix'][idxs]
            pos_distribution_idx_pm = torch.zeros_like(data)
            pos_distribution_data = torch.zeros_like(data)
            for r, i_cols in enumerate(train_parameter['positive_distribution_item_cols'][idxs]):
                pos_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
                pos_distribution_data[[r] * len(i_cols), i_cols] = 5

            eu = torch.zeros_like(data)
            rows, cols = torch.nonzero(data, as_tuple=True)
            eu[rows, cols] = 1

            G_predict_purchase, g_z = vgan.G(data, pos_distribution_data, vgan.pos_distribution_vector[idxs])
            G_predict_purchase = G_predict_purchase * (eu + pos_distribution_idx_pm)
            D_data_out, d_z1 = vgan.D(data, vgan.neg_distribution_vector[idxs])
            D_purchase_out, d_z2 = vgan.D(G_predict_purchase, vgan.neg_distribution_vector[idxs])

            loss1 = -torch.mean(torch.log(D_data_out))
            loss2 = -torch.mean(torch.log(1. - D_purchase_out))
            bce = -torch.mean(torch.sum(torch.nn.functional.log_softmax(d_z1, 1) * eu, -1))
            kld = - 0.5 * torch.mean(torch.sum(
                1 + vgan.D.std_vector(eu) - torch.square(vgan.D.mean_vector(eu)) - torch.exp(vgan.D.std_vector(eu)),
                dim=1))
            loss_dis = alpha_neg * (bce + kld)

            loss = loss1 + loss2 + loss_dis
            dis_opt.zero_grad()
            loss.backward()
            dis_opt.step()

        vgan.G.train()
        vgan.D.eval()
        for step in range(step_gen):
            idxs = random.sample(range(len(train_parameter['train_matrix'])), batch_size)
            data = train_parameter['train_matrix'][idxs]
            pos_distribution_idx_pm = torch.zeros_like(data)
            for r, i_cols in enumerate(train_parameter['positive_distribution_item_cols'][idxs]):
                pos_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
            neg_distribution_idx_pm = torch.zeros_like(data)
            for r, i_cols in enumerate(train_parameter['negative_distribution_item_cols'][idxs]):
                neg_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
            eu = torch.zeros_like(data)
            rows, cols = torch.nonzero(data, as_tuple=True)
            eu[rows, cols] = 1

            G_predict_purchase, g_z = vgan.G(data, pos_distribution_idx_pm * 5, vgan.pos_distribution_vector[idxs])
            G_predict_purchase_total = G_predict_purchase * (eu + pos_distribution_idx_pm)
            D_purchase_out, d_z = vgan.D(G_predict_purchase_total, vgan.neg_distribution_vector[idxs])

            loss1 = torch.mean(torch.log(1. - D_purchase_out))
            loss_pos_zr = train_parameter['alpha_pos_zr'] * torch.sum(
                (G_predict_purchase * pos_distribution_idx_pm - data).pow(2).sum(dim=-1) / pos_distribution_idx_pm.sum(
                    -1))
            bce = -torch.mean(torch.sum(torch.nn.functional.log_softmax(g_z, 1) * pos_distribution_idx_pm, -1))
            kld = - 0.5 * torch.mean(torch.sum(1 + vgan.G.std_vector(pos_distribution_idx_pm * 5) - torch.square(
                vgan.G.mean_vector(pos_distribution_idx_pm * 5)) - torch.exp(
                vgan.G.std_vector(pos_distribution_idx_pm * 5)), dim=1))
            loss_dis = alpha_pos * (bce + kld)

            loss = loss1 + loss_pos_zr + loss_dis
            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()

        vgan.G.eval()
        whole_pos_distribution_idx_pm = torch.zeros_like(train_parameter['train_matrix'])
        for r, i_cols in enumerate(train_parameter['positive_distribution_item_cols']):
            whole_pos_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
        new_g_z = vgan.G.get_z_distribution(whole_pos_distribution_idx_pm * 5, vgan.pos_distribution_vector)
        whole_mean, whole_std = torch.mean(new_g_z, dim=1), torch.std(new_g_z, dim=1)
        r = 0
        for mean, std in zip(whole_mean, whole_std):
            train_parameter['positive_samples_mean_std_list'][r][0] = mean
            train_parameter['positive_samples_mean_std_list'][r][1] = std
            r += 1

        G_predict_purchase, g_z = vgan.G(train_parameter['train_matrix'], whole_pos_distribution_idx_pm * 5,
                                         vgan.pos_distribution_vector)
        mean_score_each_users = torch.mean(G_predict_purchase, dim=1)

        tmp_negative_distribution_item_cols = []
        tmp_positive_distribution_item_cols = []

        for i, one_purchase in enumerate(G_predict_purchase):
            ascending_sort_arg = torch.argsort(one_purchase)

            low_pop = train_parameter['negative_samples_mean_std_list'][i][0] - 2 * \
                      train_parameter['negative_samples_mean_std_list'][i][1]
            high_pop = train_parameter['negative_samples_mean_std_list'][i][0] + 2 * \
                       train_parameter['negative_samples_mean_std_list'][i][1]
            low_top_idxs = ascending_sort_arg[
                           0: int(len(ascending_sort_arg) * train_parameter['hard_ratio'])].detach().tolist()
            low_top_idxs_set = set(low_top_idxs)
            mask1 = (train_parameter['item_col_pop_list'] >= low_pop) & (
                        train_parameter['item_col_pop_list'] <= high_pop)
            neg_idx_list = torch.where(mask1)[0].tolist()
            neg_idx_set = set(neg_idx_list)
            co_occurrence_list = list(neg_idx_set & low_top_idxs_set)
            tmp_negative_distribution_item_cols.append(co_occurrence_list)

            high_top_idxs = ascending_sort_arg[
                            int(len(ascending_sort_arg) * (1 - train_parameter['hard_ratio'])):].detach().tolist()
            high_top_idxs_set = set(high_top_idxs)
            low_pop = train_parameter['positive_samples_mean_std_list'][i][0] - 2 * \
                      train_parameter['positive_samples_mean_std_list'][i][1]
            high_pop = train_parameter['positive_samples_mean_std_list'][i][0] + 2 * \
                       train_parameter['positive_samples_mean_std_list'][i][1]
            mask2 = (train_parameter['item_col_pop_list'] >= low_pop) & (
                        train_parameter['item_col_pop_list'] <= high_pop)
            pos_idx_list = torch.where(mask2)[0].tolist()
            pos_idx_set = set(pos_idx_list)
            co_occurrence_list = list(high_top_idxs_set & pos_idx_set)
            tmp_positive_distribution_item_cols.append(co_occurrence_list)

        tmp_pos_length = [len(l) for l in tmp_positive_distribution_item_cols]
        tmp_neg_length = [len(l) for l in tmp_negative_distribution_item_cols]
        min_pos_length = np.min(tmp_pos_length)
        min_neg_length = np.min(tmp_neg_length)
        positive_distribution_item_cols_copy = tmp_positive_distribution_item_cols.copy()
        negative_distribution_item_cols_copy = tmp_negative_distribution_item_cols.copy()
        positive_distribution_item_cols = [cols[: min_pos_length] for cols in positive_distribution_item_cols_copy]
        negative_distribution_item_cols = [cols[: min_neg_length] for cols in negative_distribution_item_cols_copy]

        train_parameter['positive_distribution_item_cols'] = torch.IntTensor(positive_distribution_item_cols).to(
            train_parameter['device'])
        train_parameter['negative_distribution_item_cols'] = torch.IntTensor(negative_distribution_item_cols).to(
            train_parameter['device'])

        if (e + 1) % 1 == 0:
            print(e + 1, '\t', '==' * 24)
            (precision, precision2), (recall, recall2), (ndcg, ndcg2) = test(vgan, test_set_dict,
                                                                             train_parameter['train_matrix'],
                                                                             train_parameter[
                                                                                 'positive_distribution_item_cols'],
                                                                             top_k=top_k)
            epoche_list.append(e + 1)
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            precision_list2.append(precision2)
            recall_list2.append(recall2)
            ndcg_list2.append(ndcg2)
    max_idx = np.array(recall_list).argmax()
    max_precision, max_recall, max_ndcg = precision_list[max_idx], recall_list[max_idx], ndcg_list[max_idx]
    max_idx = np.array(recall_list2).argmax()
    max_precision2, max_recall2, max_ndcg2 = precision_list2[max_idx], recall_list2[max_idx], ndcg_list2[max_idx]
    print('MAX_P@{0}: {1}, MAX_R@{0}: {2}, MAX_NDCG@{0}: {3}'.format(top_k, max_precision, max_recall, max_ndcg))
    print('MAX_P@{0}: {1}, MAX_R@{0}: {2}, MAX_NDCG@{0}: {3}'.format(50, max_precision2, max_recall2, max_ndcg2))


def train_vgan2_non_pop_non_positive_pref(train_parameter, epoches, batch_size, nb_zr, nb_pm, alpha_pos, alpha_neg,
                                          test_set_dict, top_k):
    epoche_list, precision_list, recall_list, ndcg_list = [], [], [], []
    epoche_list2, precision_list2, recall_list2, ndcg_list2 = [], [], [], []
    vgan = VGANNonPositivePref(
        train_parameter['nb_user'],
        train_parameter['nb_item'],
        train_parameter['pos_dis_size'],
        train_parameter['neg_dis_size'],
        train_parameter['discriminator_vector_dim'],
        train_parameter['generator_vector_dim'],
        train_parameter['device']
    )

    vgan.G.to(train_parameter['device'])
    vgan.D.to(train_parameter['device'])
    gen_opt = torch.optim.Adam(vgan.G.parameters(), lr=0.0001)
    dis_opt = torch.optim.Adam(vgan.D.parameters(), lr=0.0001)
    step_gen, step_dis = 5, 2
    for e in range(epoches):
        vgan.D.train()
        vgan.G.eval()
        for step in range(step_dis):
            idxs = random.sample(range(len(train_parameter['train_matrix'])), batch_size)
            data = train_parameter['train_matrix'][idxs]

            neg_distribution_idx_pm = torch.zeros_like(data)
            for r, i_cols in enumerate(train_parameter['negative_distribution_item_cols'][idxs]):
                neg_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1

            eu = torch.zeros_like(data)
            rows, cols = torch.nonzero(data, as_tuple=True)
            eu[rows, cols] = 1

            G_predict_purchase, g_z = vgan.G(data, vgan.pos_distribution_vector[idxs])
            G_predict_purchase = G_predict_purchase * eu
            D_data_out, d_z1 = vgan.D(data, neg_distribution_idx_pm, vgan.neg_distribution_vector[idxs])
            D_purchase_out, d_z2 = vgan.D(G_predict_purchase, neg_distribution_idx_pm,
                                          vgan.neg_distribution_vector[idxs])

            loss1 = -torch.mean(torch.log(D_data_out))
            loss2 = -torch.mean(torch.log(1. - D_purchase_out))
            bce = -torch.mean(torch.sum(torch.nn.functional.log_softmax(d_z1, 1) * neg_distribution_idx_pm, -1))
            kld = - 0.5 * torch.mean(torch.sum(1 + vgan.D.std_vector(neg_distribution_idx_pm) - torch.square(
                vgan.D.mean_vector(neg_distribution_idx_pm)) - torch.exp(vgan.D.std_vector(neg_distribution_idx_pm)),
                                               dim=1))
            loss_dis = alpha_neg * (bce + kld)

            loss = loss1 + loss2 + loss_dis
            dis_opt.zero_grad()
            loss.backward()
            dis_opt.step()

        vgan.D.eval()
        whole_neg_distribution_idx_pm = torch.zeros_like(train_parameter['train_matrix'])
        for r, i_cols in enumerate(train_parameter['negative_distribution_item_cols']):
            whole_neg_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
        new_d_z = vgan.D.get_z_distribution(whole_neg_distribution_idx_pm, vgan.neg_distribution_vector)
        whole_mean, whole_std = torch.mean(new_d_z, dim=1), torch.std(new_d_z, dim=1)
        r = 0
        for mean, std in zip(whole_mean, whole_std):
            train_parameter['negative_samples_mean_std_list'][r][0] = mean
            train_parameter['negative_samples_mean_std_list'][r][1] = std
            r += 1

        vgan.G.train()
        vgan.D.eval()
        for step in range(step_gen):
            idxs = random.sample(range(len(train_parameter['train_matrix'])), batch_size)
            data = train_parameter['train_matrix'][idxs]

            neg_distribution_idx_pm = torch.zeros_like(data)
            for r, i_cols in enumerate(train_parameter['negative_distribution_item_cols'][idxs]):
                neg_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
            eu = torch.zeros_like(data)
            rows, cols = torch.nonzero(data, as_tuple=True)
            eu[rows, cols] = 1

            G_predict_purchase, g_z = vgan.G(data, vgan.pos_distribution_vector[idxs])
            G_predict_purchase_total = G_predict_purchase * (eu + neg_distribution_idx_pm)
            D_purchase_out, d_z = vgan.D(G_predict_purchase_total, neg_distribution_idx_pm,
                                         vgan.neg_distribution_vector[idxs])

            loss1 = torch.mean(torch.log(1. - D_purchase_out))
            loss_neg_zr = train_parameter['alpha_neg_zr'] * torch.sum(
                (G_predict_purchase * neg_distribution_idx_pm - data).pow(2).sum(dim=-1) / neg_distribution_idx_pm.sum(
                    -1))
            bce = -torch.mean(torch.sum(torch.nn.functional.log_softmax(g_z, 1) * eu, -1))
            kld = - 0.5 * torch.mean(torch.sum(
                1 + vgan.G.std_vector(eu) - torch.square(vgan.G.mean_vector(eu)) - torch.exp(vgan.G.std_vector(eu)),
                dim=1))
            loss_dis = alpha_pos * (bce + kld)

            loss = loss1 + loss_neg_zr + loss_dis
            gen_opt.zero_grad()
            loss.backward()
            gen_opt.step()

        G_predict_purchase, g_z = vgan.G(train_parameter['train_matrix'], vgan.pos_distribution_vector)
        mean_score_each_users = torch.mean(G_predict_purchase, dim=1)

        tmp_negative_distribution_item_cols = []
        tmp_positive_distribution_item_cols = []

        for i, one_purchase in enumerate(G_predict_purchase):
            ascending_sort_arg = torch.argsort(one_purchase)

            low_pop = train_parameter['negative_samples_mean_std_list'][i][0] - 2 * \
                      train_parameter['negative_samples_mean_std_list'][i][1]
            high_pop = train_parameter['negative_samples_mean_std_list'][i][0] + 2 * \
                       train_parameter['negative_samples_mean_std_list'][i][1]
            low_top_idxs = ascending_sort_arg[
                           0: int(len(ascending_sort_arg) * train_parameter['hard_ratio'])].detach().tolist()
            low_top_idxs_set = set(low_top_idxs)
            mask1 = (train_parameter['item_col_pop_list'] >= low_pop) & (
                        train_parameter['item_col_pop_list'] <= high_pop)
            neg_idx_list = torch.where(mask1)[0].tolist()
            neg_idx_set = set(neg_idx_list)
            co_occurrence_list = list(neg_idx_set & low_top_idxs_set)
            tmp_negative_distribution_item_cols.append(co_occurrence_list)

            high_top_idxs = ascending_sort_arg[
                            int(len(ascending_sort_arg) * (1 - train_parameter['hard_ratio'])):].detach().tolist()
            high_top_idxs_set = set(high_top_idxs)
            low_pop = train_parameter['positive_samples_mean_std_list'][i][0] - 2 * \
                      train_parameter['positive_samples_mean_std_list'][i][1]
            high_pop = train_parameter['positive_samples_mean_std_list'][i][0] + 2 * \
                       train_parameter['positive_samples_mean_std_list'][i][1]
            mask2 = (train_parameter['item_col_pop_list'] >= low_pop) & (
                        train_parameter['item_col_pop_list'] <= high_pop)
            pos_idx_list = torch.where(mask2)[0].tolist()
            pos_idx_set = set(pos_idx_list)
            co_occurrence_list = list(high_top_idxs_set & pos_idx_set)
            tmp_positive_distribution_item_cols.append(co_occurrence_list)

        tmp_pos_length = [len(l) for l in tmp_positive_distribution_item_cols]
        tmp_neg_length = [len(l) for l in tmp_negative_distribution_item_cols]
        min_pos_length = np.min(tmp_pos_length)
        min_neg_length = np.min(tmp_neg_length)
        positive_distribution_item_cols_copy = tmp_positive_distribution_item_cols.copy()
        negative_distribution_item_cols_copy = tmp_negative_distribution_item_cols.copy()
        positive_distribution_item_cols = [cols[: min_pos_length] for cols in positive_distribution_item_cols_copy]
        negative_distribution_item_cols = [cols[: min_neg_length] for cols in negative_distribution_item_cols_copy]

        train_parameter['positive_distribution_item_cols'] = torch.IntTensor(positive_distribution_item_cols).to(
            train_parameter['device'])
        train_parameter['negative_distribution_item_cols'] = torch.IntTensor(negative_distribution_item_cols).to(
            train_parameter['device'])

        if (e + 1) % 1 == 0:
            print(e + 1, '\t', '==' * 24)
            (precision, precision2), (recall, recall2), (ndcg, ndcg2) = non_positive_pref_test(vgan, test_set_dict,
                                                                                               train_parameter[
                                                                                                   'train_matrix'],
                                                                                               train_parameter[
                                                                                                   'positive_distribution_item_cols'],
                                                                                               top_k=top_k)
            epoche_list.append(e + 1)
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            precision_list2.append(precision2)
            recall_list2.append(recall2)
            ndcg_list2.append(ndcg2)
    max_idx = np.array(recall_list).argmax()
    max_precision, max_recall, max_ndcg = precision_list[max_idx], recall_list[max_idx], ndcg_list[max_idx]
    max_idx = np.array(recall_list2).argmax()
    max_precision2, max_recall2, max_ndcg2 = precision_list2[max_idx], recall_list2[max_idx], ndcg_list2[max_idx]
    print('MAX_P@{0}: {1}, MAX_R@{0}: {2}, MAX_NDCG@{0}: {3}'.format(top_k, max_precision, max_recall, max_ndcg))
    print('MAX_P@{0}: {1}, MAX_R@{0}: {2}, MAX_NDCG@{0}: {3}'.format(50, max_precision2, max_recall2, max_ndcg2))
