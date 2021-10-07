#!/usr/bin/env python
# coding: utf-8

# In[1]:
# 生成单模成绩，保险

import pandas as pd
import numpy as np
from collections import Counter,defaultdict
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pickle
import tqdm
import sys
import lightgbm as lgb
import matplotlib.pyplot as plt
from utils import reduce,MultiAUC,MultiAuc_Metric,save_gbm_model,load_gbm_model,save_kfold_gbm_model,load_kflod_gbm_model
from utils import get_best_reweight


# In[2]:


test_share_df = pd.read_csv('/opt/sub/zsf-model/share/share-online-emb16.csv')


# In[3]:


def transform_reweight_df(df, weight):
    best_weight = np.array(weight)
    watch_cols = ['watch_label_%d'%(i) for i in range(10) ]
    res = df[watch_cols].values
    weight_res = res*best_weight
    
    test_watch_df = df.copy()
    for i in range(10):
        test_watch_df['watch_label_%d'%(i)] = weight_res[:,i]
    test_watch_df = reduce(test_watch_df)
    # test_watch_df.to_csv(path,index=False)
    
    return test_watch_df


# In[4]:


test_watch_df = pd.read_csv('/opt/sub/zsf-model/watch/watch-online-emb16.csv')
best_weight = np.array([0.4, 1.21619785, 0.83820996, 1.21191235, 1.39959, 1.63130683, 1.37598242, 1.50311951, 1.55457164, 0.5])
trans_df=transform_reweight_df(test_watch_df, weight=best_weight,)


# In[5]:


test_share_df = pd.read_csv('/opt/sub/zsf-model/share/share-online-emb16.csv')
test_watch_df = trans_df
watch_cols = ['watch_label_%d'%(i) for i in range(10) ]
res = test_watch_df[watch_cols].values

preds_args = np.argmax(res,axis=-1)
sub_df = pd.read_csv('/opt/data/testdata/test.csv')
sub_df['watch_label'] = preds_args
sub_df['is_share'] = test_share_df['is_share']
sub_df = reduce(sub_df)
sub_df.to_csv('/opt/sub/zsf-model/submission.csv',index=False)


# In[ ]:




