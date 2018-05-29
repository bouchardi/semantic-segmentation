import torch.optim as optim
import torch.nn as nn
import torch

from segmentation.dummy_model import Dummy
from segmentation.fcn_model import FCNModel
from segmentation.datasets import PascalVOC2012, CIFAR


def train(model, dataset, criterion, optimizer, path='/project/fcn.pt', n_epoch=2):
    for epoch in range(n_epoch):

        running_loss = 0.0
        for i in range(len(dataset)):

            inputs, labels = dataset[i]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), path)
    print('Model saved: {}'.format(path))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = FCNModel()
    dataset = PascalVOC2012('train')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, dataset, criterion, optimizer, n_epoch=1)
