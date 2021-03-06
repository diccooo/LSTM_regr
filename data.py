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
        print('reading data...')
        df = pd.read_csv('./dataset/data.csv', index_col=0, header=[0,1])
        print('reading done')
        df = df.iloc[:, :Config.proc_col_num*2]
        df_scaled = self.scale(df, Config.scalemethod)
        # # sorting by var 1----------------------------------------------------
        # msk = [i * 2 for i in range(Config.proc_col_num)]
        # df_tmp = df_scaled.iloc[:, msk]
        # tmplist0 = [i * 48 + 47 for i in range(self.idx_t - 1)]
        # tmplist1 = [i + 48 for i in tmplist0]
        # df_tmp = df_tmp.values[tmplist1] - df_tmp.values[tmplist0]
        # df_tmp = pd.DataFrame(df_tmp)
        # df_dif_var = pd.Series(Config.proc_col_num)
        # for i in range(Config.proc_col_num):
        #     df_dif_var[i] = df_tmp.iloc[:,i].values.var()
        # df_dif_var.sort_values(ascending=Config.var_ascending, inplace=True, kind='mergesort')
        # # print(df_dif_var)
        # idxlist = df_dif_var.index.tolist()
        # idxlist1 = [x * 2 for x in idxlist]
        # idxlist2 = [x + 1 for x in idxlist1]
        # idxlist = [x for cp in zip(idxlist1, idxlist2) for x in cp]
        # idxlist = idxlist[:Config.colnum*2]
        # df_scaled = df_scaled.iloc[:, idxlist]
        # # msk = [i * 2 for i in range(Config.colnum)]
        # # df_tmp = df_scaled.iloc[:, msk]
        # # df_tmp = df_tmp.values[tmplist1] - df_tmp.values[tmplist0]
        # # df_tmp = pd.DataFrame(df_tmp)
        # # for i in range(Config.colnum):
        # #     print(df_tmp.iloc[:,i].values.var())
        # df = df.iloc[:, idxlist]
        # # sorting by var 1----------------------------------------------------
        # sorting by var 2----------------------------------------------------
        df_dif_var = pd.Series(Config.proc_col_num)
        for i in range(Config.proc_col_num):
            df_dif_var[i] = df_scaled.iloc[:,i*2].values.var() * df_scaled.iloc[:,i*2+1].values.var()
        df_dif_var.sort_values(ascending=Config.var_ascending, inplace=True, kind='mergesort')
        # print(df_dif_var)
        idxlist = df_dif_var.index.tolist()
        idxlist1 = [x * 2 for x in idxlist]
        idxlist2 = [x + 1 for x in idxlist1]
        idxlist = [x for cp in zip(idxlist1, idxlist2) for x in cp]
        idxlist = idxlist[:Config.colnum*2]
        df_scaled = df_scaled.iloc[:, idxlist]
        # for i in range(Config.colnum):
        #     print(df_scaled.iloc[:,i*2].values.var() * df_scaled.iloc[:,i*2+1].values.var())
        df = df.iloc[:, idxlist]
        # sorting by var 2----------------------------------------------------
        xlist, ylist0, ylist1 = self.xy_idx()
        xdata = df_scaled.values[xlist]
        ydata = df.values[ylist1] - df.values[ylist0]
        # ydata = df_scaled.values[ylist1] - df_scaled.values[ylist0]
        msk = [i * 2 for i in range(self.colnum)]
        ydata = ydata[:, msk]
        print('processing done')
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
    print('saving done')

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