import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
# test
test = pd.read_csv('/opt/data/testdata/test.csv')
s = np.zeros((len(test), 10))
for i in range(2017, 2022):
    s += np.load('/opt/sub/dcn_watch_' + str(i) + '.npy') / 5

np.save('/opt/sub/watch_on.npy', s)