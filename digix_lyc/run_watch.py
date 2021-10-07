import os
import pandas as pd
import numpy as np
import argparse
import random
import tensorflow as tf
import keras.backend as K
from time import time
from tqdm import tqdm
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.optimizers import Adam, Adagrad
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tensorflow.python.keras.models import save_model, load_model
from sklearn.preprocessing import label_binarize
from model import AutoInt_, DCN_


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

parser = argparse.ArgumentParser(description='seed')
parser.add_argument("--k", metavar='seed', type=int, default=0)
args = parser.parse_args()
k = args.k

seed = k
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

epochs = 80
batch_size = 9192
embedding_dim = 32
n_classes = 10
score_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 读取训练集
BEHAVIOR_PATH = '/opt/data/traindata/history_behavior_data/'
DATE_PATH = [BEHAVIOR_PATH + i for i in os.listdir(BEHAVIOR_PATH)]
DATE_PATH = sorted(DATE_PATH)
BEHAVIOR_PATH_ = [i + '/' + os.listdir(i)[0] for i in DATE_PATH]

history_behavior_list = []
for i in tqdm(range(len(BEHAVIOR_PATH_))):
    history_behavior = pd.read_csv(BEHAVIOR_PATH_[i], sep='\t')
    history_behavior_list.append(history_behavior)

history_behavior_df = pd.concat(history_behavior_list)
del history_behavior_list

# 训练集采样
train = history_behavior_df
train1 = train[train['watch_label'] != 0]
print(len(train1))
train2 = train[train['watch_label'] == 0].sample(frac=0.1, random_state=k).reset_index(drop=True)
print(len(train2))
train = pd.concat([train1, train2])
del train1, train2

# 读取测试集
test = pd.read_csv('/opt/data/testdata/test.csv')
test['pt_d'] = '20210503'
data = pd.concat([train, test])

# 视频特征拼接
video_features = pd.read_pickle('/opt/feature/video_features.pkl')
data = pd.merge(data, video_features, on='video_id', how='left')

# 日期特征
data['pt_d'] = data['pt_d'].astype('str')
data['pt_d'] = data['pt_d'].apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
data['pt_d'] = pd.to_datetime(data['pt_d'])
data['day'] = (data['pt_d'] - data['video_release_date']).dt.days

# w2v特征
user_features = pd.read_pickle('/opt/feature/user_video_w2v.pkl')
data = pd.merge(data, user_features, on='user_id', how='left')

# 用户属性特征
user_features = pd.read_csv('/opt/data/traindata/user_features_data/user_features_data.csv', sep='\t')
data = pd.merge(data, user_features, on='user_id', how='left')

# 特征数据预处理
data[['video_year', 'video_month']] = data[['video_year', 'video_month']].fillna(0)
data[['video_year', 'video_month']] = data[['video_year', 'video_month']].astype('int32')

target = ['watch_label', 'is_share']
sparse_features = ['province', 'city', 'device_name', 'user_id', 'video_id', 'video_year', 'video_month', 'gender', 'city_level', 'age']
varlen_sparse_features = ['video_tags', 'video_director_list', 'video_actor_list', 'video_second_class']
dense_features = ['video_score', 'video_duration',  'day']

print(len(dense_features))
print(len(sparse_features))
data[dense_features] = data[dense_features].fillna(-1)
scaler = MinMaxScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])
dense_features.extend([f'user_id_video_id_emb_{i}' for i in range(16)])

maxlen_dict = {'video_tags': 81, 'video_director_list': 11, 'video_actor_list': 104, 'video_second_class': 9}
vocab_dict = {'video_tags': 21427, 'video_director_list': 28036, 'video_actor_list': 78291, 'video_second_class': 145}

for f in varlen_sparse_features:
    data[f] = data[f].fillna(0)
    data[f] = data[f].apply(lambda x: [0] * maxlen_dict[f] if x == 0 else x)


train = data[data['pt_d'] <= '2021-05-02']
test = data[data['pt_d'] == '2021-05-03']
print(len(test))
del data

# 构造输入
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=max(train[feat].max(), test[feat].max()) + 1, embedding_dim=embedding_dim)
                          for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

varlen_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=vocab_dict[feat],
                                                      embedding_dim=embedding_dim), maxlen=maxlen_dict[feat], combiner='mean', weight_name=None) for feat in varlen_sparse_features]

dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
linear_feature_columns = fixlen_feature_columns + varlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}

for feat in tqdm(varlen_sparse_features):
    train_model_input[feat] = np.asarray(train_model_input[feat].to_list()).astype(np.int32)
    test_model_input[feat] = np.asarray(test_model_input[feat].to_list()).astype(np.int32)



y_binary = label_binarize(train['watch_label'], classes=range(n_classes))
s = train['watch_label'].value_counts().to_dict()
label_num_list = [s[i] for i in sorted(s.keys())]
total_sample_num = sum(label_num_list)

def multi_category_focal_loss(y_true, y_pred):
    gamma = 2.0
    epsilon = 1.e-7

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    classes_weight = [total_sample_num / label_num for label_num in label_num_list]
    classes_weight_norm = [weight / sum(classes_weight) for weight in classes_weight]
    print(classes_weight_norm)
    classes_weight_norm = tf.cast(classes_weight_norm, tf.float32)
    alpha = tf.expand_dims(classes_weight_norm, 1)

    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    pt = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
    cross_entropy_loss = -K.log(pt)
    weight = tf.pow(tf.subtract(1., pt), gamma)
    focal_loss = tf.matmul(tf.multiply(weight, cross_entropy_loss), alpha)

    return focal_loss


# 模型训练
epochs = 5
model = DCN_(linear_feature_columns, dnn_feature_columns, task='multiclass')
model.compile("adagrad", multi_category_focal_loss)

history = model.fit(train_model_input, y_binary, batch_size=batch_size, epochs=epochs, verbose=1)
# save_model(model, 'dcn_watch' + str(k) + '.h5')
# model.save_weights('dcn_watch_' + str(k) + '.h5')
if not os.path.exists('/opt/sub/'):
    os.makedirs('/opt/sub/')

temp = model.predict(test_model_input, batch_size=batch_size * 4)
np.save('/opt/sub/dcn_watch_' + str(k) + '.npy', temp)
