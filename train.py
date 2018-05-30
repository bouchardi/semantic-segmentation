import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader

from segmentation.fcn_model import FCNModel
from segmentation.datasets import PascalVOC2012


def train(model, dataset, criterion, optimizer, device, path='/project/fcn.pt', n_epoch=2):

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), path)
    print('Model saved: {}'.format(path))





#    x = torch.LongTensor(np.asarray(x))
#    x = x.unsqueeze(0)
#
#    x = Variable(x)
#    x = x.unsqueeze(0)




if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Start training on {}'.format(device))

    model = FCNModel()
    if device != 'cpu':
        model.to(device)

    dataset = PascalVOC2012('train')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train(model, dataset, criterion, optimizer, device, n_epoch=1)
