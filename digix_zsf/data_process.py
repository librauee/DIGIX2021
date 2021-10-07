#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
from collections import Counter,defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import pickle
from gensim.models import Word2Vec
import tqdm
import sys
import os
from utils import reduce, encode_sparse_feature, dense_feature_norm,emb_varlen_feature,emb_pair_feature,group_fea


# In[2]:
def make_feature(df):
    print('开始构造count特征')
    sparse_feats = ['user_id','age', 'gender', 'country', 'province', 'city', 'device_name',] + ['video_id','video_year']
    for f in sparse_feats:
        df[f+'_count'] = df[f].map(df[f].value_counts())
    
    print('开始构造nunique特征')
    nunique_group = []
    key_feature = ['user_id']
    target_feature = ['video_id','video_year']
    
    for feat_1 in key_feature:
        for feat_2 in target_feature:
            if feat_1 + '_' + feat_2 + '_nunique' not in nunique_group:
                nunique_group.append(feat_1 + '_' + feat_2 + '_nunique')
                tmp = group_fea(df, feat_1, feat_2)
                df = df.merge(tmp, on=feat_1, how='left')
                
            if feat_2 + '_' + feat_1 + '_nunique' not in nunique_group:
                nunique_group.append(feat_2 + '_' + feat_1 + '_nunique')
                tmp = group_fea(df, feat_2, feat_1)
                df = df.merge(tmp, on=feat_2, how='left')
    
    key_feature = ['video_id']
    target_feature = ['user_id','age', 'gender', 'country', 'province', 'city', 'device_name',]
    
    for feat_1 in key_feature:
        for feat_2 in target_feature:
            if feat_1 + '_' + feat_2 + '_nunique' not in nunique_group:
                nunique_group.append(feat_1 + '_' + feat_2 + '_nunique')
                tmp = group_fea(df, feat_1, feat_2)
                df = df.merge(tmp, on=feat_1, how='left')
            
            if feat_2 + '_' + feat_1 + '_nunique' not in nunique_group:
                nunique_group.append(feat_2 + '_' + feat_1 + '_nunique')
                tmp = group_fea(df, feat_2, feat_1)
                df = df.merge(tmp, on=feat_2, how='left')
    
    
    print('开始构造ctr特征')
#     mean_iswatch_rate = df[df['period'] < 14]['is_watch'].mean()  # 平均ctr率
#     mean_watchlabel_rate = df[df['period'] < 14]['watch_label'].mean()  # 平均ctr率
#     mean_share_rate = df[df['period'] < 14]['is_share'].mean()  # 平均ctr率
    
    feature_list = ['user_id', 'video_id','device_name','city']
    task_feature = ['is_watch', 'watch_label', 'is_share','is_collect', 'is_comment']  # 
    
    mean_rate_dict = {feat_2: df[df['period'] < 14][feat_2].mean() for feat_2 in task_feature}
    mean_coldu_rate_dict = {feat_2: df[(df['period'] < 14)&(df['coldu']==1)][feat_2].mean() for feat_2 in task_feature}
    mean_coldv_rate_dict = {feat_2: df[(df['period'] < 14)&(df['coldv']==1)][feat_2].mean() for feat_2 in task_feature}
    
    for feat_1 in feature_list:
        res = pd.DataFrame()
        # 各个(特征,pt_d)对应的ctr率
        for period in range(1, 15, 1):
            if period == 0:
                count = df[df['period'] <= period].groupby(feat_1)[task_feature].mean().reset_index()
            #         elif period == 14:
            #             count = df[df['period'] < 14][task_feature].groupby(feat_1)[task_feature].mean().reset_index()
            else:
                count = df[df['period'] < period].groupby(feat_1)[task_feature].mean().reset_index()  # , as_index=False
            count['period'] = period
            res = res.append(count, ignore_index=True)

        # mean_rate 重命名
        res.rename(columns={feat_2: feat_1 + '_' + feat_2 + '_mean_rate' for feat_2 in task_feature}, inplace=True)
        df = pd.merge(df, res, how='left', on=[feat_1, 'period'], sort=False,)  # 生成了新的df

        for feat_2 in task_feature:
            if feat_1 == 'user_id':
                '''update 增加对user_watch_label_mean_rate的调整修正  date 8.25'''
#                 if feat_2=='watch_label':
#                     df[feat_1 + '_' + feat_2 + '_mean_rate'].fillna(mean_coldu_rate_dict[feat_2]*0.67, inplace=True)
#                 elif feat_2=='is_watch':
#                     df[feat_1 + '_' + feat_2 + '_mean_rate'].fillna(mean_coldu_rate_dict[feat_2]*1.2, inplace=True)
#                 else:
                df[feat_1 + '_' + feat_2 + '_mean_rate'].fillna(mean_coldu_rate_dict[feat_2], inplace=True)
            elif feat_1 == 'video_id':
                df[feat_1 + '_' + feat_2 + '_mean_rate'].fillna(mean_coldv_rate_dict[feat_2], inplace=True)
            else:
                df[feat_1 + '_' + feat_2 + '_mean_rate'].fillna(mean_rate_dict[feat_2], inplace=True)
                
        print(feat_1, ' over')

    return df


# In[3]:


user_df = pd.read_csv('/opt/data/traindata/user_features_data/user_features_data.csv',sep='\t')
video_df = pd.read_csv('/opt/data/traindata/video_features_data/video_features_data.csv',sep='\t')
test_df = pd.read_csv('/opt/data/testdata/test.csv')

df_list = []
df19 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210419/part-00000-236b99d5-456a-42b2-bd8d-3cbd61d21cc6-c000.csv',sep='\t')
df20 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210420/part-00000-aad75aa4-b60b-4f5b-8def-c4d60f391fae-c000.csv',sep='\t')
df21 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210421/part-00000-c15f29da-6b1e-48c0-b7d0-2cd560998c3f-c000.csv',sep='\t')
df22 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210422/part-00000-3d97d0f8-2572-45e6-bb60-f367c97e7870-c000.csv',sep='\t')
df23 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210423/part-00000-9809d73a-a55f-4ac2-a59b-9b83cbc5028e-c000.csv',sep='\t')
df24 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210424/part-00000-225e55dc-4504-4c14-b289-322312355b2b-c000.csv',sep='\t')
df25 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210425/part-00000-9d23862b-6bbf-48c6-a598-572df1359737-c000.csv',sep='\t')
df26 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210426/part-00000-0d315342-3ba7-4727-b4a2-123a1a004786-c000.csv',sep='\t')
df27 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210427/part-00000-9132ab46-51c3-4cc3-97de-e7ad5312b852-c000.csv',sep='\t')
df28 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210428/part-00000-fc8c8ca1-e655-4a45-b179-c8d9e2dd804c-c000.csv',sep='\t')
df29 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210429/part-00000-c5dbd994-54d7-4734-adea-0f22d75b23d3-c000.csv',sep='\t')
df30 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210430/part-00000-2da4c3a0-2fcc-422d-8b7c-48940da315ad-c000.csv',sep='\t')
df01 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210501/part-00000-56b6f0ff-57b8-41ac-96ec-c20e3306297e-c000.csv',sep='\t')
df02 = pd.read_csv('/opt/data/traindata/history_behavior_data/20210502/part-00000-d412c602-2f4a-4649-a81f-e56435dd49fd-c000.csv',sep='\t')

for c in set(df19.columns.tolist())-set(test_df.columns.tolist()):
    test_df[c] = 0
test_df['pt_d']=20210503

df_list = [df19, df20, df21, df22, df23, df24, df25, df26,df27,df28,df29 ,df30,df01,df02,test_df]
for i in range(len(df_list)):
    df_list[i]['period'] = i
    
user_set = set()
video_set = set()

for i in range(len(df_list)):
    df = df_list[i]
    cur_user_set, cur_video_set = set(df['user_id'].unique()), set(df['video_id'].unique())
    
    if i<1:
        continue
    
    df['coldu'] = df['user_id'].apply(lambda x: 1 if x not in user_set else 0)
    df['coldv'] = df['video_id'].apply(lambda x: 1 if x not in video_set else 0)
    
    user_set = user_set|cur_user_set
    video_set = video_set|cur_video_set
    

behav_df = pd.concat(df_list,axis=0)


# In[4]:


tmp1 = emb_pair_feature(behav_df,'user_id','video_id',  emb_size=16, min_count=3,)
tmp2 = emb_pair_feature(behav_df,'video_id','user_id',  emb_size=16, min_count=3,)

behav_df = pd.merge(behav_df,tmp1,on='user_id',how='left')
behav_df = pd.merge(behav_df,tmp2,on='video_id',how='left')

emb_feats = [c for c in behav_df.columns.to_list() if 'emb' in c]
# emb 特征缺失值补0
for feat in emb_feats:
    behav_df[feat].fillna(value=0,inplace=True)


# In[5]:


# 修正人潮汹涌
# video_df.loc[video_df['video_id']==28149,'video_duration'] = 60*130


# In[6]:


# 生成embedding特征
tags_feats = ['video_director_list','video_actor_list','video_second_class']
for feat in tags_feats:
    emb_varlen_feature(video_df,feat,emb_size=4)
emb_feats =  [c for c in video_df.columns.to_list() if 'emb' in c]


# In[7]:


tot_df = pd.merge(behav_df,user_df,on=['user_id'],how='left')
video_feats = [ 'video_id','video_release_date','video_duration','video_score'] + emb_feats
tot_df = pd.merge(tot_df,video_df[video_feats],on=['video_id'],how='left')
tot_df = reduce(tot_df)


# In[8]:



tot_df['video_duration'] = tot_df['video_duration'].apply(lambda x : x/60)
# most_value = tot_df['video_duration'].mode()[0]
tot_df['video_duration'].fillna(value=1,inplace=True)
### video_duration分桶处理,还可以更细致
# tot_df['video_duration_type'] = tot_df['video_duration'].apply(lambda x : duration2type(x))

tot_df['video_release_date'] = pd.to_datetime(tot_df['video_release_date'],format="%Y-%m-%d")
tot_df['video_year'] = 2021 - tot_df['video_release_date'].dt.year

# video_year缺失值填充
tot_df['video_year'].fillna(value=-1, inplace=True)
tot_df['video_score'].fillna(value=6,inplace=True)


# emb 特征缺失值补0
for feat in emb_feats:
    tot_df[feat].fillna(value=0,inplace=True)


# In[9]:


user_sparse_feats = ['user_id','age', 'gender', 'country', 'province', 'city',  'device_name',] # 'city_level'
video_sparse_feats = ['video_id','video_year']   # 'video_duration_type',
sparse_feats = user_sparse_feats + video_sparse_feats
# 离散变量编号
encode_sparse_feature(tot_df, cols=sparse_feats)

tot_df = reduce(tot_df)


# In[10]:


# ctr 特征
tot_df = make_feature(tot_df)


# In[11]:


# 数值特征归一化
ctr_feat = [ c  for c in tot_df.columns.to_list() if 'mean_rate' in c ]
count_feat = [ c  for c in tot_df.columns.to_list() if '_count'in c ]
nunique_feat = [ c  for c in tot_df.columns.to_list() if '_nunique'in c ]

dense_feature = ['video_score','video_duration'] + ctr_feat + count_feat + nunique_feat

for f in dense_feature:
    dense_feature_norm(tot_df, f)


# In[12]:


tot_df.drop(columns=['city_level','video_release_date','watch_start_time'],inplace=True)


# In[13]:


# 压缩df
tot_df = reduce(tot_df)

# 保存
# tot_df.to_csv('DATA/total.csv',index=False)

dir_path = '/opt/feature'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

tot_df.to_pickle('/opt/feature/total-emb16.pkl')


# In[ ]:




