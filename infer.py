import argparse

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader

from segmentation.fcn_model import FCNModel
from segmentation.datasets import PascalVOC2012


def infer(model, dataset, count=1, random=True):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=random)
    for i in range(count):
        data = dataloader[i]
        result = model(data['image'])

        out = result.detach().numpy()[0].argmax(axis=0)
        _min, _max = out.min(), out.max()
        print('{}: min = {}, max = {}'.format(i, _min, _max)

        if _max != 0:
            nonnull += 1
    print('{} out of {} gave something'.format(nonnull, count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='Experiment number')
    parser.add_argument('--count', help='Inference count')
    parser.add_argument('--random', help='Pick random image from valid set')

    args = parser.parse_args()

    model = FCNModel(vgg16_pretrained=False)
    mod.from_file('/project/fcn_{}.pt'.format(args.n))

    dataset = PascalVOC2012('val')

    infer(model, dataset, args.count, args.random)
