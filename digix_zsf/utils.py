import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from gensim.models import Word2Vec
import pickle

# 数据压缩
def reduce(df):
    int_list = ['int', 'int32', 'int16']
    float_list = ['float', 'float32']
    for col in df.columns.tolist():
        col_type = df[col].dtypes
        if col_type in int_list:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif col_type in float_list:
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df

def group_fea(df, key, target):
    # 计算每一个key和多少个不同的target交互
    tmp = df.groupby(key, as_index=False)[target].agg({
        key + '_' + target + '_nunique': 'nunique',
    }).reset_index().drop('index', axis=1)

    return tmp

# 离散变量编号
def encode_sparse_feature(df,cols):
    for c in cols:
        feat_set = set(df[c].unique())
        feat2idx = {feat:i for i,feat in enumerate(feat_set)}
        df[c] = df[c].map(feat2idx)

# 数值变量归一化
def dense_feature_norm(df, feat):
    mms = MinMaxScaler(feature_range=(0, 1))
    df[feat] = mms.fit_transform(df[feat].values.reshape(-1, 1))

# 可变长变量映射为embedding处理
def emb_varlen_feature(df, feat, emb_size=4, window=20, min_count=5, epochs=5):
    sentences = df[feat].values.tolist()

    for i in range(len(sentences)):
        tags = sentences[i].split(',') if sentences[i] is not np.nan else []
        sentences[i] = [str(x) for x in tags]

    model = Word2Vec(sentences, vector_size=emb_size, window=window, min_count=min_count, sg=0, hs=0, seed=1, epochs=epochs, workers=1)
    emb_matrix = []

    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)

    emb_matrix = np.array(emb_matrix)

    for i in range(emb_size):
        df['{}_emb_{}'.format(feat, i)] = emb_matrix[:, i]

def emb_pair_feature(df, f1, f2, emb_size=4, window=20, min_count=5, epochs=5):
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})    # df[df['is_watch']==1]
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]

    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]

    model = Word2Vec(sentences, vector_size=emb_size, window=window, min_count=min_count, sg=0, hs=0, seed=1, epochs=epochs, workers=1)
    emb_matrix = []

    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)

    emb_matrix = np.array(emb_matrix)

    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]

    return tmp


# 多分类AUC
def MultiAUC(y_pred, y_label):
    '''
    :param y_pred: np.array shape [n_sample,]
    :param y_label: np.array shape [n_sample,]
    :return: aucs: np.array [n_class,],   weighted_auc: float
    '''
    aucs = np.zeros(10)
    m = len(y_pred)
    for i in range(0, 10):
        n = np.sum(y_label==i)                   # 正确标签y_label中，label=i的个数
        h = np.sum(y_pred[y_label == i] == i)    # y_pred中，y_pred=i正确的个数
        k = np.sum(y_pred==i) - h                # y_pred中，y_pred=i错误的个数

        aucs[i] = (h*(m-n-k) + 0.5*h*k + 0.5*(n-h)*(m-n-k))/((m-n)*n)

    weights = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    weighted_auc = np.sum(weights*aucs)
    return aucs, weighted_auc

# 自定义metric
def accuracy(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'accuracy', np.mean(labels == (preds > 0.5)), True

def MultiAuc_Metric(preds, train_data):
    y_label = train_data.get_label()
    # preds = preds.reshape(-1,10)
    preds = preds.reshape(10,-1).T
    y_pred = np.argmax(preds, axis=-1)
    aucs, weighted_auc = MultiAUC(y_pred, y_label)

    return 'weighted_auc', weighted_auc, True

def get_best_reweight(y_train_pred, y_label, n_iter=100, seed=1):
    '''
    :param y_train_pred: np.array [n_sample, n_class] 分类概率
    :param y_label:      np.array [n_sample,]
    :param n_iter:       迭代次数
    :return:  best_auc, best_weight
    '''
    np.random.seed(seed)
    best_auc = 0
    best_weight = None  # type: np.array
    for i in range(n_iter):
        fine_weight = np.random.normal(loc=1.0, scale=0.5, size=10)
        fine_weight[fine_weight < 0.5] = 0.5
        weight_pred = y_train_pred * fine_weight
        y_pred = np.argmax(weight_pred, axis=-1)
        auc, weighted_auc = MultiAUC(y_pred, y_label)
        print("index=%d auc=%.6f" % (i, weighted_auc))
        if (best_auc < weighted_auc):
            best_auc = weighted_auc
            best_weight = fine_weight

    return best_auc, best_weight

# save lightgbm model
def save_gbm_model(gbm, evals_result, path='model.pkl'):
    with open(path, 'wb') as fout:
        pickle.dump({'gbm': gbm, 'evals_result': evals_result}, fout)
    print('save complete')

# save kfold lightgbm model
def save_kfold_gbm_model(gbms, evals_results, path='kfold_model.pkl'):
    with open(path, 'wb') as fout:
        pickle.dump({'gbms': gbms, 'evals_results': evals_results}, fout)
    print('save complete')

# load lightgbm model
def load_gbm_model(path):
    with open(path, 'rb') as fin:
        load_dict = pickle.load(fin)
        gbm = load_dict['gbm']
        evals_result = load_dict['evals_result']
    return gbm, evals_result

# load kfold lightgbm model
def load_kflod_gbm_model(path):
    with open(path, 'rb') as fin:
        load_dict = pickle.load(fin)
        gbms = load_dict['gbms']
        evals_results = load_dict['evals_results']
    return gbms, evals_results


