from numba import njit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import numpy as np
import math
import pandas as pd

watch = pd.read_csv('F:/data/digix/zsf-model/watch/watch-offline.csv')
watch_lgb = watch[[f'watch_label_{i}' for i in range(10)]].values / 5

watch_nn = np.load('F:/data/digix/watch_off.npy')
watch_mean = watch_lgb + watch_nn

score_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_classes = 10
v_y = np.load('F:/data/digix/watch_label_10.npy')
y_binary_val = label_binarize(v_y, classes=range(n_classes))

@njit
def _auc(actual, pred_ranks):
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def fast_auc(actual, predicted):
    # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    pred_ranks = rankdata(predicted)
    return _auc(actual, pred_ranks)

def cal_score(pred_prob):
    oof_train = label_binarize(np.argmax(pred_prob, axis=1), classes=range(n_classes))
    score = np.zeros(n_classes)
    weight_auc = 0
    for i in range(n_classes):
        score[i] = fast_auc(y_binary_val[:, i], oof_train[:, i])
        weight_auc += 0.7 * (score[i] * score_weights[i])
    return weight_auc

def search_weight(raw_prob, init_weight=[1.0]*n_classes, step=0.001):
    weight = init_weight.copy()
    f_best = cal_score(raw_prob)
    flag_score = 0
    round_num = 1
    while round_num < 6:
        print('round: ', round_num)
        round_num += 1
        flag_score = f_best
        for c in tqdm(range(n_classes)):
            for n_w in range(400, 2400, 10):
                num = n_w * step
                new_weight = weight.copy()
                new_weight[c] = num

                prob_df = raw_prob.copy()
                prob_df = prob_df * np.array(new_weight)

                f = cal_score(prob_df)
                if f > f_best:
                    weight = new_weight.copy()
                    f_best = f
        print(f_best)
        print(weight)
    return weight

weight = search_weight(watch_mean)
