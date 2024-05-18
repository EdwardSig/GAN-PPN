import math

import numpy as np
import torch


def P_R_N_AP_RR(true, pred, topN):
    pred = pred.tolist()
    pred = np.array(pred).argsort()[::-1][:topN]
    pred = list(pred)
    p, r, ndcg, ap, rr = 0.0, 0.0, 0.0, 0.0, 0.0
    TP = list(set(true) & set(pred))

    p = float(len(TP)) / len(pred)
    r = float(len(TP)) / len(true)
    if len(TP) != 0:
        TP_index = np.array([1 + pred.index(i) for i in TP])
        DCG = 1 / np.log2(TP_index + 1)
        IDCG = 1 / np.log2(np.arange(len(TP)) + 1 + 1)
        ndcg = sum(DCG) / sum(IDCG)
        ap = sum([(list(TP_index).index(i) + 1) / i for i in TP_index])
        rr = np.mean(1 / TP_index)
    return p, r, ndcg, ap, rr


def test(vgan, test_set_dict, train_set, positive_distribution_item_cols, top_k=5):
    vgan.G.eval()
    vgan.D.eval()
    gen = vgan.G
    users = list(test_set_dict.keys())
    input_data = train_set[users]
    pos_distribution_idx_pm = torch.zeros_like(input_data)
    for r, i_cols in enumerate(positive_distribution_item_cols[users]):
        pos_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
    out, z = gen(input_data, pos_distribution_idx_pm * 5, vgan.pos_distribution_vector[users])
    out = (out - 999 * input_data).cpu().detach().numpy()
    precisions = 0
    recalls = 0
    ndcgs = 0.0
    precisions2, recalls2, ndcgs2 = 0, 0, 0
    hits = 0
    total_purchase_nb = 0
    for i, u in enumerate(users):
        p, r, ndcg, _, _ = P_R_N_AP_RR(test_set_dict[u], out[i], top_k)
        p2, r2, ndcg2, _, _ = P_R_N_AP_RR(test_set_dict[u], out[i], 50)
        precisions += p
        recalls += r
        ndcgs += ndcg
        precisions2 += p2
        recalls2 += r2
        ndcgs2 += ndcg2
    recall = recalls / len(users)
    precision = precisions / len(users)
    ndcg = ndcgs / len(users)
    recall2 = recalls2 / len(users)
    precision2 = precisions2 / len(users)
    ndcg2 = ndcgs2 / len(users)
    print('P@{0}: {1}, R@{0}: {2}, NDCG@{0}: {3}'.format(top_k, precision, recall, ndcg))
    print('P@{0}: {1}, R@{0}: {2}, NDCG@{0}: {3}'.format(50, precision2, recall2, ndcg2))
    return (precision, precision2), (recall, recall2), (ndcg, ndcg2)


def non_positive_pref_test(vgan, test_set_dict, train_set, positive_distribution_item_cols, top_k=5):
    vgan.G.eval()
    vgan.D.eval()
    gen = vgan.G
    users = list(test_set_dict.keys())
    input_data = train_set[users]
    pos_distribution_idx_pm = torch.zeros_like(input_data)
    for r, i_cols in enumerate(positive_distribution_item_cols[users]):
        pos_distribution_idx_pm[[r] * len(i_cols), i_cols] = 1
    out, z = gen(input_data, vgan.pos_distribution_vector[users])
    out = (out - 999 * input_data).cpu().detach().numpy()
    precisions = 0
    recalls = 0
    ndcgs = 0.0
    precisions2, recalls2, ndcgs2 = 0, 0, 0
    hits = 0
    total_purchase_nb = 0
    for i, u in enumerate(users):
        p, r, ndcg, _, _ = P_R_N_AP_RR(test_set_dict[u], out[i], top_k)
        p2, r2, ndcg2, _, _ = P_R_N_AP_RR(test_set_dict[u], out[i], 50)
        precisions += p
        recalls += r
        ndcgs += ndcg
        precisions2 += p2
        recalls2 += r2
        ndcgs2 += ndcg2
    recall = recalls / len(users)
    precision = precisions / len(users)
    ndcg = ndcgs / len(users)
    recall2 = recalls2 / len(users)
    precision2 = precisions2 / len(users)
    ndcg2 = ndcgs2 / len(users)
    print('P@{0}: {1}, R@{0}: {2}, NDCG@{0}: {3}'.format(top_k, precision, recall, ndcg))
    print('P@{0}: {1}, R@{0}: {2}, NDCG@{0}: {3}'.format(50, precision2, recall2, ndcg2))
    return (precision, precision2), (recall, recall2), (ndcg, ndcg2)