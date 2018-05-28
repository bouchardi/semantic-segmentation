import torch.optim as optim
import torch.nn as nn

from segmentation.dummy_model import Dummy
from segmentation.datasets import PascalVOC2012, CIFAR


def train(model, dataset, criterion, optimizer, n_epoch=2):
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')



if __name__ == '__main__':
    model = Dummy()
    #dataset = PascalVOC2012('train')
    dataset = CIFAR()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, dataset, criterion, optimizer)
