from __future__ import print_function

import os
import pandas as pd
from tqdm import tqdm

npre = './data/'
npost = '.csv'

filelist = pd.read_csv('datafilelist.csv', header=None)
datalist = []

print('reading in...')
for _, nseries in tqdm(filelist.iterrows(), total=len(filelist)):
    fname = npre + nseries[0] + npost
    if not os.path.exists(fname):
        fname = npre + '-' + nseries[0] + npost
    data_i = pd.read_csv(fname, header=None, index_col=0, names=[(nseries[0],0),(nseries[0],1)])
    datalist.append(data_i)
print('reading done')

print('writing...')
data = pd.concat(datalist, axis=1)
print(data)
data.to_csv('data.csv', chunksize=10)
print('writing done')