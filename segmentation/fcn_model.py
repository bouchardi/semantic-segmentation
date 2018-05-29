import torch.nn as nn
from torch import randn
import torch.nn.functional as F
from torchvision.models import vgg16

import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.autograd import Variable


class FCNModel(nn.Module):
    def __init__(self):
        super(FCNModel, self).__init__()

        # VGG16 pretrained features
        VGG16 = vgg16(pretrained=True)
        self.features = VGG16.features
        print('vgg16')

        self.conv1 = nn.Conv2d(512, 4096, 7)
        self.conv2 = nn.Conv2d(4096, 4096, 1)
        self.conv3 = nn.Conv2d(4096, 1000, 1)
        self.conv4 = nn.Conv2d(1000, 21, 1)

        self.deconv1 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1)
        self.crop = nn.ConstantPad2d(-2, -2)
        self.deconv2 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1)
        self.deconv7 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1)
        self.deconv8 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1)
        self.deconv9 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1)

    def forward(self, x):
        input_size = x.size()
        print(input_size)
        x = self.features(x)
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = self.conv2(x)
        print(x.size())
        x = self.conv3(x)
        print(x.size())
        x = self.conv4(x)
        print(x.size())
        x = self.deconv1(x)
        print(x.size())
        x = self.crop(x)
        print(x.size())
        x = self.deconv2(x)
        print(x.size())
        x = self.deconv3(x)
        print(x.size())
        x = self.deconv4(x)
        print(x.size())
        x = self.deconv5(x)
        print(x.size())
        x = self.deconv6(x)
        print(x.size())
        return x
