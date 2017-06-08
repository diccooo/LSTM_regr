from __future__ import division

import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable

from config import Config

class PreData(object):
    def __init__(self):
        self.colnum = Config.colnum
        self.tsnum = Config.nseries
        self.tstotal = Config.tstotal
        self.tspd = Config.tspd
        self.idx_t = self.tstotal // self.tspd
        self.idx_s = self.tsnum * 3 // self.tspd + 1
        self.x, self.y = self.pre_data()

    def sample(self, startidx):
        n1 = self.tsnum // 3
        n0 = self.tsnum - n1 * 2
        idx0 = startidx
        idx3 = startidx + self.tspd * self.idx_s
        idx2 = idx3 - n1
        idx1 = (idx3 + startidx) // 2
        gap0 = (idx1 - idx0) // n0
        gap1 = (idx2 - idx1) // n1
        idxlist = []
        l = [i for i in range(idx0, idx1, gap0)]
        idxlist += l[-n0:]
        l = [i for i in range(idx1, idx2, gap1)]
        idxlist += l[-n1:]
        l = [i for i in range(idx2, idx3)]
        idxlist += l
        return idxlist

    def xy_idx(self):
        xlist = []
        ylist0 = []
        for i in range(self.idx_t - self.idx_s):
            xlist += self.sample(i * 48)
            ylist0.append((i + self.idx_s) * 48 - 1)
        ylist1 = [i + 48 for i in ylist0]
        return xlist, ylist0, ylist1

    def scale(self, df, method=0):
        res = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        if method == 0:
            return res
        if method == 1:
            res = res.apply(lambda x: np.log(x * 1.71828 + 1))
            return res
        if method == 2:
            res.iloc[:,::2] = res.iloc[:,::2].apply(lambda x: np.log(x * 1.71828 + 1))
            return res
        if method == 3:
            res.iloc[:,1::2] = res.iloc[:,1::2].apply(lambda x: np.log(x * 1.71828 + 1))
            return res

    def pre_data(self):
        df = pd.read_csv('./dataset/data.csv', index_col=0, header=[0,1])
        df = df.iloc[:, :self.colnum * 2]
        df_scaled = self.scale(df, Config.scalemethod)
        xlist, ylist0, ylist1 = self.xy_idx()
        xdata = df_scaled.values[xlist]
        ydata = df.values[ylist1] - df.values[ylist0]
        # ydata = df_scaled.values[ylist1] - df_scaled.values[ylist0]
        msk = [i * 2 for i in range(self.colnum)]
        ydata = ydata[:, msk]
        return torch.from_numpy(xdata), torch.from_numpy(ydata)

class Data(object):
    def __init__(self, re_data=Config.re_data):
        if os.path.exists(Config.xpath) and os.path.exists(Config.ypath) and not re_data:
            self.x_ = torch.load(Config.xpath)
            self.y_ = torch.load(Config.ypath)
        else:
            data = PreData()
            torch.save(data.x, Config.xpath)
            torch.save(data.y, Config.ypath)
            self.x_ = data.x
            self.y_ = data.y
        self.x_ = self.x_.view(-1, Config.nseries, Config.ninput)
        self.x_ = self.x_.float()
        self.y_ = self.y_.float()
        self.datasize = self.y_.size(0)
        self.testsize = round(Config.testsize / Config.batsize) * Config.batsize
        self.trainsize = (self.datasize - self.testsize) // Config.batsize * Config.batsize
        self.trainx = self.x_[:self.trainsize]
        self.trainy = self.y_[:self.trainsize]
        self.testx = self.x_[-self.testsize:]
        self.testy = self.y_[-self.testsize:]

def savedata():
    data = PreData()
    torch.save(data.x, Config.xpath)
    torch.save(data.y, Config.ypath)

def test():
    data = Data()
    l1 = [i for i in range(data.datasize)]
    train1 = l1[:data.trainsize]
    test1 = l1[-data.testsize:]
    print(train1[0], train1[-1])
    print(test1[0], test1[-1])
    print(data.trainx.size(), data.trainy.size())
    print(data.testx.size(), data.testy.size())

def get_batch(x, y, i, evaluation=False):
    xbat = Variable(x[i : i+Config.batsize], volatile=evaluation)
    ybat = Variable(y[i : i+Config.batsize])
    return xbat, ybat

if __name__ == '__main__':
    savedata()