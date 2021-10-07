import pandas as pd
from tqdm import tqdm
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import re
import os
from gensim.models import Word2Vec
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# 读取数据
BEHAVIOR_PATH = '/opt/data/traindata/history_behavior_data/'
DATE_PATH = [BEHAVIOR_PATH + i for i in os.listdir(BEHAVIOR_PATH)]
DATE_PATH = sorted(DATE_PATH)
BEHAVIOR_PATH_ = [i + '/' + os.listdir(i)[0] for i in DATE_PATH]

history_behavior_list = []
for i in tqdm(range(len(BEHAVIOR_PATH_))):
    history_behavior = pd.read_csv(BEHAVIOR_PATH_[i], sep='\t')
    history_behavior['pt_d'] = i + 1
    history_behavior_list.append(history_behavior)

test = pd.read_csv('/opt/data/testdata/test.csv')
test['pt_d'] = 15
history_behavior_list.append(test)
history_behavior_df = pd.concat(history_behavior_list)
history_behavior_df.pt_d = history_behavior_df.pt_d.astype('int8')
del history_behavior_list

# w2v 特征
print('w2v started!')
def emb(df, f1, f2):
    emb_size = 16
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=6, min_count=5, sg=0, hs=0, seed=1, iter=5)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv.vocab:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    return tmp

user_df_ = emb(history_behavior_df, f1='user_id', f2='video_id')
user_df_ = reduce_mem_usage(user_df_)

if not os.path.exists('/opt/feature/'):
    os.makedirs('/opt/feature/')
user_df_.to_pickle('/opt/feature/user_video_w2v.pkl')
print('w2v finished!')


# 视频特征
print('video started!')
video_features = pd.read_csv('/opt/data/traindata/video_features_data/video_features_data.csv', sep='\t')
video_features['video_release_date'] = pd.to_datetime(video_features['video_release_date'])
item_varlen_sparse_features = ['video_tags', 'video_director_list', 'video_actor_list', 'video_second_class']
for f in item_varlen_sparse_features:
    video_features[f] = video_features[f].apply(lambda x: x.replace(';', ',').replace(',,', ',').strip(',') if x is not np.NAN else x)

maxlen_dict = {}
vocab_dict = {}
video_tags_dict = {}
count = 1
for i in list(video_features.video_tags):
    if i is not np.NAN:
        for j in i.split(','):
            if j == '':
                print(i)
            if j not in video_tags_dict.keys():
                video_tags_dict[j] = count
                count += 1

video_features.video_tags = video_features.video_tags.apply(lambda x: list(map(lambda y: video_tags_dict[y], x.split(','))) if x is not np.NAN else [])
video_features.video_tags = video_features.video_tags.apply(lambda x: str(x).strip('[').strip(']'))

video_tags_dict = {}
count = 1
for i in list(video_features.video_director_list):
    if i is not np.NAN:
        for j in i.split(','):
            if j == '':
                print(i)
            if j not in video_tags_dict.keys():
                video_tags_dict[j] = count
                count += 1

video_features.video_director_list = video_features.video_director_list.apply(lambda x: list(map(lambda y: video_tags_dict[y], x.split(','))) if x is not np.NAN else [])
video_features.video_director_list = video_features.video_director_list.apply(lambda x: str(x).strip('[').strip(']'))

video_tags_dict = {}
count = 1
for i in list(video_features.video_actor_list):
    if i is not np.NAN:
        for j in i.split(','):
            if j == '':
                print(i)
            if j not in video_tags_dict.keys():
                video_tags_dict[j] = count
                count += 1


video_features.video_actor_list = video_features.video_actor_list.apply(lambda x: list(map(lambda y: video_tags_dict[y], x.split(','))) if x is not np.NAN else [])
video_features.video_actor_list = video_features.video_actor_list.apply(lambda x: str(x).strip('[').strip(']'))

video_tags_dict = {}
count = 1
for i in list(video_features.video_second_class):
    if i is not np.NAN:
        for j in i.split(','):
            if j == '':
                print(i)
            if j not in video_tags_dict.keys():
                video_tags_dict[j] = count
                count += 1


video_features.video_second_class = video_features.video_second_class.apply(lambda x: list(map(lambda y: video_tags_dict[y], x.split(','))) if x is not np.NAN else [])
video_features.video_second_class = video_features.video_second_class.apply(lambda x: str(x).strip('[').strip(']'))

def get_vocab_size(feat):
    return max(video_features[feat].apply(lambda x: max(x) if list(x) else 0)) + 1

def get_maxlen(feat):
    return max(video_features[feat].apply(lambda x: len(x)))

for feat in item_varlen_sparse_features:
    video_features[feat] = video_features[feat].apply(lambda x: np.fromstring(x, 'int', sep=',') if x is not np.NAN else np.array([], dtype=np.int32))
    video_features[feat] = pad_sequences(video_features[feat], maxlen=get_maxlen(feat), padding='post', dtype=np.int32, value=0).tolist()
    maxlen_dict[feat] = get_maxlen(feat)
    vocab_dict[feat] = get_vocab_size(feat)

video_features = video_features[['video_id', 'video_release_date', 'video_score',  'video_duration',
                                 'video_tags', 'video_director_list', 'video_actor_list', 'video_second_class'
                                ]]
video_features['video_year'] = video_features['video_release_date'].dt.year
video_features['video_month'] = video_features['video_release_date'].dt.month
print(maxlen_dict)
print(vocab_dict)

video_features.to_pickle('/opt/feature/video_features.pkl')
print('video finished!')

