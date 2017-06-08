from __future__ import print_function

import torch
import numpy as np

from config import Config
from data import Data, get_batch
from model import LSTMregrModel

def evaluate(model, criterion, evalx, evaly):
    model.eval()
    total_loss = 0
    outputlist = []
    for i in range(0, evaly.size(0), Config.batsize):
        xbat, ybat = get_batch(evalx, evaly, i, evaluation=True)
        output = model(xbat)
        total_loss += ybat.size(0) * criterion(output, ybat).data
        outputlist.append(output)
    output = torch.cat(outputlist)
    predval = output.data.cpu().numpy()
    trueval = evaly.cpu().numpy()
    npratio = np.sum((predval * trueval) > 0) / trueval.size
    dif = np.sum(np.square(predval - trueval)) / np.sum(np.square(trueval))
    dif = np.sqrt(dif)
    return total_loss[0] / evaly.size(0), npratio, dif

def main():
    havecuda = torch.cuda.is_available()
    torch.manual_seed(Config.seed)
    if havecuda:
        torch.cuda.manual_seed(Config.seed)

    data = Data(re_data=False)
    testx, testy = data.testx, data.testy
    if havecuda:
        testx, testy = testx.cuda(), testy.cuda()

    model = LSTMregrModel()
    if havecuda:
        model.cuda()
    criterion = torch.nn.MSELoss()
    model.load_state_dict(torch.load(Config.wpath))

    test_loss, npratio, dif = evaluate(model, criterion, testx, testy)
    print('=' * 80)
    print(
        '| test loss {:10.5f} | npratio {:5.2f}% | dif {:2.5f} |'.format(
            test_loss, npratio*100, dif
        )
    )
    print('=' * 80)

if __name__ == '__main__':
    main()