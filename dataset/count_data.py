from __future__ import print_function

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

npre = './data/'
npost = '.csv'

filelist = pd.read_csv('datafilelist.csv', header=None)
filelist = filelist.iloc[::-1]
data_count = pd.Series()

print('reading in...')
for _, nseries in tqdm(filelist.iterrows(), total=len(filelist)):
    fname = npre + nseries[0] + npost
    if not os.path.exists(fname):
        fname = npre + '-' + nseries[0] + npost
    data_i = pd.read_csv(fname, header=None, index_col=0, names=[0, 1])
    data_i = data_i.replace(0.0, np.NaN)
    data_count[fname[7:]] = data_i.count().sum()
print('reading done')

print('writing...')
data_count.sort_values(ascending=False, inplace=True, kind='mergesort')
print(data_count)
data_count.to_csv('data_count.csv')
print('writing done')