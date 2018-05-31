import argparse

import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader

from segmentation.fcn_model import FCNModel
from segmentation.datasets import PascalVOC2012


def infer(model, dataset, count=1, random=True):
    nonnull = 0
    dataloader = DataLoader(dataset, batch_size=1, shuffle=random)
    for i, data in enumerate(dataloader):
        result = model(data['image'])
        result = nn.Softmax()(result)
        print(result.view(-1))
        out = result.detach().numpy()[0].argmax(axis=0)
        _min, _max, _mean = out.min(), out.max(), out.mean()
        print('{}: min = {}, max = {}, mean = {}'.format(i, _min, _max, _mean))
        print(np.unique(out))

        if _max != 0:
            nonnull += 1

        if i == count -1:
            print('{} out of {} gave something'.format(nonnull, count))
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='Experiment number')
    parser.add_argument('--count', type=int, help='Inference count')
    parser.add_argument('--random', type=bool, help='Pick random image from valid set')

    args = parser.parse_args()

    model = FCNModel(vgg16_pretrained=False)
    model.from_file('/project/fcn_{}.pt'.format(args.n))

    dataset = PascalVOC2012('val')

    infer(model, dataset, args.count, args.random)
