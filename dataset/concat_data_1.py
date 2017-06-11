from __future__ import print_function

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

npre = './data/'

filelist = pd.read_csv('data_count.csv', header=None, names=[0,1])
datalist = []

print('reading in...')
for _, nseries in tqdm(filelist.iterrows(), total=len(filelist)):
    fname = npre + nseries[0]
    data_i = pd.read_csv(fname, header=None, index_col=0, names=[(nseries[0],0),(nseries[0],1)])
    datalist.append(data_i)
print('reading done')

data = pd.concat(datalist, axis=1)
data.iloc[:,::2].replace(0, np.NaN, inplace=True)
data.iloc[:,::2].fillna(method='pad', inplace=True)
data.fillna(0, inplace=True)
print(data)
print('writing...')
data.to_csv('data.csv', chunksize=10)
print('writing done')