# import pandas as pd
# import numpy as np
# import os
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR)
#
# weight = [0.66, 0.78, 1.3, 1.73, 1.84, 1.85, 1.98, 2.34, 1.94, 1.0]
# watch = pd.read_csv('/opt/sub/zsf-model/watch/watch-online-emb16.csv')
# watch_lgb = watch[[f'watch_label_{i}' for i in range(10)]].values
#
# watch_nn = np.load('/opt/sub/watch_on.npy')
# watch_mean = watch_lgb + watch_nn
# watch_mean = watch_mean * np.array(weight)
# pred = np.argmax(watch_mean, axis=1)
# test = pd.read_csv('/opt/data/testdata/test.csv')
# test['watch_label'] = pred
# s = test[['user_id', 'video_id', 'watch_label']]
# s['watch_label'] = s['watch_label'].astype('int8')
#
# share = pd.read_csv('/opt/sub/zsf-model/share/share-online-emb16.csv')
# s['is_share'] = share['is_share']
# s.to_csv('/opt/sub/submission.csv', index=False, float_format='%.6f')
# s.head()



import matplotlib.pyplot as plt

name_list = ['PPO', 'V-Learning', 'ORCA']
num_list = [(0.742+0.826)/2, (0.84+0.562)/2, (0.668+0.278)/2]
num_list1 = [(0.576+0.574)/2, 0.505, (0.472+0.164)/2]
x = list(range(len(num_list)))
total_width, n = 0.95, 2
width = total_width / n
font = {
    'size': 16,
    'family': 'serif',
    'serif': 'Times'
}
plt.ylabel('Success Rate', fontdict=font)
plt.xlabel('Algorithms', fontdict=font)

plt.bar(x, num_list, width=width, label='SLI', fc='r')
for i in range(len(x)):
    x[i] = x[i] + width
plt.xticks([i + width/2 for i in range(len(x))] ,name_list, font={'size':12})

plt.bar(x, num_list1, width=width, label='Baseline', fc='b')

plt.legend()
# plt.show()
plt.savefig("gene.pdf")

