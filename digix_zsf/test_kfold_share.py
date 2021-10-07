#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter,defaultdict
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys
import pickle
import tqdm
import sys
import os
import lightgbm as lgb
from utils import reduce,save_gbm_model,load_gbm_model,save_kfold_gbm_model,load_kflod_gbm_model


# In[2]:


df = pd.read_pickle('/opt/feature/total-emb16.pkl')  # 生成的特征在 /opt/feature文件夹下


# In[3]:
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'boost_from_average' : True,
    'train_metric': True, 
    'feature_fraction_seed' : 1,
    'learning_rate': 0.05,
    'is_unbalance': False,  #当训练数据是不平衡的，正负样本相差悬殊的时候，可以将这个属性设为true,此时会自动给少的样本赋予更高的权重
    'num_leaves': 256,  # 一般设为少于2^(max_depth)
    'max_depth': -1,  #最大的树深，设为-1时表示不限制树的深度
    'min_child_samples': 15,  # 每个叶子结点最少包含的样本数量，用于正则化，避免过拟合
    'max_bin': 200,  # 设置连续特征或大量类型的离散特征的bins的数量
    
    'subsample': 1,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'bagging_seed':3,
    
    'colsample_bytree': 0.5,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 2.99,  # L1 regularization term on weights
    'reg_lambda': 1.9,  # L2 regularization term on weights
    'nthread': 20,
    'verbose': 0,
    }


# In[4]:


sparse_feat = ['user_id', 'video_id', 'age', 'gender', 'country', 'province', 'city', 'device_name',"period"]   # 'city_level', 'video_duration_type',
dense_feat = [ 'video_duration', 'video_score','video_year',]

ctr_feat = [ c  for c in df.columns.to_list() if 'mean_rate' in c ]

count_feat = [ c  for c in df.columns.to_list() if '_count'in c ]
nunique_feat = [ c  for c in df.columns.to_list() if '_nunique'in c ]
emb_feat = [c for c in df.columns.to_list() if 'emb' in c]

feature = sparse_feat + dense_feat + ctr_feat + emb_feat + count_feat + nunique_feat

feature = sorted(list(set(feature)))


# In[5]:


train_df = df[df["period"]<14].reset_index(drop=True)
test_df = df[df["period"]==14].reset_index(drop=True)


del df
X_train = train_df
y_train = X_train["is_share"].astype('int32')

X_train0 = X_train[X_train['is_share']==0]
X_train1 = X_train[X_train['is_share']==1]


# In[6]:


fold = 4
gbms = []
evals_results = []
skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=1)

for i, (trn_idx, val_idx) in enumerate(skf.split(X_train0,X_train0["period"])):
    print('fold:{}'.format(i + 1))
    X_train_KF = pd.concat([X_train0.iloc[val_idx].reset_index(drop=True),X_train1], axis=0).reset_index(drop=True)
    y_train_KF = X_train_KF["is_share"].astype('int32')
                                       
    lgb_train = lgb.Dataset(X_train_KF[feature], y_train_KF)
    
    evals_result= {}
    gbm = lgb.train(params, lgb_train, num_boost_round=300,valid_sets=(lgb_train), evals_result=evals_result, verbose_eval=20,)
    gbms.append(gbm)
    evals_results.append(evals_result)
    print("")


# In[7]:


fold = len(gbms)
print('开始{}折结果融合'.format(fold))
res = np.zeros(len(test_df))
for i in range(fold):
    preds = gbms[i].predict(test_df[feature], num_iteration=180)
    res += preds


# In[8]:


sub_df = pd.read_csv('/opt/data/testdata/test.csv')
sub_df['watch_label'] = 0
sub_df['is_share'] = res/fold

sub_df = reduce(sub_df)

dir_path = '/opt/sub/zsf-model/share'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
sub_df.to_csv('/opt/sub/zsf-model/share/share-online-emb16.csv',index=False)


# In[9]:
# save_kfold_gbm_model(gbms, evals_results, path='kfold_share_model.pkl')
# In[13]:
# gbms, evals_results = load_kflod_gbm_model(path='kfold_share_model.pkl')

