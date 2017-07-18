from __future__ import print_function

import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from config import Config
from data import Data, get_batch
from model import LSTMregrModel
from eval import evaluate

def train(model, criterion, trainx, trainy, lr, epoch):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=Config.regular_lambda)
    total_loss = 0
    r_loss = 0
    np_total = 0
    start_time = time.time()
    for batch, i in enumerate(range(0, trainy.size(0), Config.batsize)):
        xbat, ybat = get_batch(trainx, trainy, i)
        model.zero_grad()
        output = model(xbat)
        np_total += np.sum((output.data.cpu().numpy() * ybat.data.cpu().numpy()) > 0)
        loss = criterion(output, ybat)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), Config.clip)
        optimizer.step()
        total_loss += ybat.size(0) * loss.data
        if batch > 0 and batch % Config.log_interval == 0:
            cur_loss = total_loss[0] / Config.log_interval
            elapsed = time.time() - start_time
            # print(
            #     '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | '
            #     'ms/batch {:6.2f} | loss {:10.5f} |'.format(
            #         epoch, batch, trainy.size(0)//Config.batsize, lr,
            #         elapsed*1000/Config.log_interval, cur_loss
            #     )
            # )
            r_loss += total_loss
            total_loss = 0
            start_time = time.time()
    r_loss += total_loss
    npratio = np_total / (trainy.size(0) * trainy.size(1))
    return r_loss[0] / trainy.size(0), npratio

def main():
    havecuda = torch.cuda.is_available()
    torch.manual_seed(Config.seed)
    if havecuda:
        torch.cuda.manual_seed(Config.seed)

    data = Data()
    trainx, trainy = data.trainx, data.trainy
    testx, testy = data.testx, data.testy
    if havecuda:
        trainx, trainy = trainx.cuda(), trainy.cuda()
        testx, testy = testx.cuda(), testy.cuda()

    model = LSTMregrModel()
    if havecuda:
        model.cuda()
    criterion = torch.nn.MSELoss()
    lr = Config.lr
    best_train_loss = None
    best_test_loss = None
    notupdatedepochs = 0
    if Config.re_train:
        train_npratiolist = []
        train_losslist = []
        test_npratiolist = []
        test_losslist = []
    else:
        with open('./weight/figlisttemp.pkl', 'rb') as f:
            train_npratiolist = pickle.load(f)
            train_losslist = pickle.load(f)
            test_npratiolist = pickle.load(f)
            test_losslist = pickle.load(f)
            lr = pickle.load(f)
        model.load_state_dict(torch.load('./weight/weighttemp.pkl'))
    startepoch = len(train_npratiolist)

    try:
        for epoch in range(startepoch+1, Config.epoch+1):
            epoch_start_time = time.time()
            train_loss, train_npratio = train(model, criterion, trainx, trainy, lr, epoch)
            train_npratiolist.append(train_npratio*100)
            train_losslist.append(train_loss)
            if epoch == 2:
                train_losslist[0] = train_losslist[1]
            fig = plt.figure(figsize=(25, 10))
            ax11 = fig.add_subplot(121)
            ax11.plot(train_npratiolist)
            ax12 = ax11.twinx()
            ax12.plot(train_losslist, 'r')
            print('-' * 80)
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | lr {:02.6f} | '
                'loss {:10.5f} | npratio {:5.2f}% |'.format(
                    epoch, (time.time()-epoch_start_time), lr, train_loss, train_npratio*100
                )
            )
            print('-' * 80)
            if not best_train_loss or train_loss < best_train_loss:
                best_train_loss = train_loss
                notupdatedepochs = 0
            else:
                notupdatedepochs += 1
            if notupdatedepochs > 50 and lr > 0.000002:
                lr /= 10
                notupdatedepochs = 0

            test_loss, test_npratio, dif = evaluate(model, criterion, testx, testy)
            test_npratiolist.append(test_npratio*100)
            test_losslist.append(test_loss)
            ax21 = fig.add_subplot(122)
            ax21.plot(test_npratiolist)
            ax22 = ax21.twinx()
            ax22.plot(test_losslist, 'r')
            plt.savefig('output.png')
            plt.clf()
            print(
                '| test loss {:10.5f} | npratio {:5.2f}% | dif {:2.5f} |'.format(
                    test_loss, test_npratio*100, dif
                )
            )
            print('-' * 80)
            if not best_test_loss or test_loss < best_test_loss:
                torch.save(model.state_dict(), Config.wpath)
                print('weight saved at epoch {:3d}'.format(epoch))
                print('-' * 80)
                best_test_loss = test_loss

            if epoch % 50 == 0:
                torch.save(model.state_dict(), './weight/weighttemp.pkl')
                with open('./weight/figlisttemp.pkl', 'wb') as f:
                    pickle.dump(train_npratiolist, f)
                    pickle.dump(train_losslist, f)
                    pickle.dump(test_npratiolist, f)
                    pickle.dump(test_losslist, f)
                    pickle.dump(lr, f)

        print('training done')
    except KeyboardInterrupt:
        print('-' * 80)
        print('Exiting from training early')
    
if __name__ == '__main__':
    main()
