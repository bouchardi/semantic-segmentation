import argparse

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader

from segmentation.fcn_model import FCNModel
from segmentation.datasets import PascalVOC2012, CamVid


def train(model, dataset, criterion, optimizer, device, batch_size=8, workers=4, path='/project/fcn.pt', n_epoch=2):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            _input = data['image']
            label = data['label']

            _input = Variable(_input)

            if device != 'cpu':
                _input, label = _input.to(device), label.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(_input)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            current_loss = loss.item()
            running_loss += current_loss
            if i % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), path)
    print('Model saved: {}'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',  help='Dataset', choices=['pascal', 'camvid'], default='camvid')
    parser.add_argument('-o', '--optimizer',  help='Optimizer', choices=['sdg', 'adam'], default='sdg')
    parser.add_argument('-bs', type=int, help='Batch size', default=8)
    parser.add_argument('-lr', type=float, help='Learning rate', default=0.0001)
    parser.add_argument('--n-epochs', type=int, help='Epochs', default=1)

    args = parser.parse_args()

    if args.dataset == 'pascal':
        dataset = PascalVOC2012('train')
    elif args.dataset == 'camvid':
        dataset = CamVid('train')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Start training on {}'.format(device))

    model = FCNModel(n_classes=dataset.get_classes_count(), vgg16_pretrained=True)
    if device != 'cpu':
        model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.get_ignored_class())
    if args.optimizer == 'sdg':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    path = '/project/fcn8s_{}_{}_bs{}_lr{}_e{}.pt'.format(args.dataset, args.optimizer, args.bs, args.lr, args.n_epochs)
    train(model, dataset, criterion, optimizer, device, batch_size=args.bs, path=path, n_epoch=args.n_epochs)
