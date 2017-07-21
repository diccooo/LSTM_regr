class Config(object):
    nseries = 200
    colnum = 300
    ninput = 2 * colnum
    nhidd = ninput
    nout = colnum
    dropout = 0.5
    batsize = 1
    nlayers = 4
    lr = 1
    clip = 0.5
    regular_lambda = 0
    log_interval = 100
    epoch = 2000
    scalemethod = 2
    proc_col_num = 1300
    var_ascending = False
    re_data = False
    re_train = True

    tstotal = 29472         # total time series
    tspd = 48               # time series per day
    seed = 1
    testsize = 100
    xpath = './dataset/x.pkl'
    ypath = './dataset/y.pkl'
    wpath = './weight/weight.pkl'
    