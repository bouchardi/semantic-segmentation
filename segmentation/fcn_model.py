import torch
import torch.nn as nn
from torch import randn
import torch.nn.functional as F
from torchvision.models import vgg16

class FCNModel(nn.Module):
    def __init__(self, vgg16_pretrained=False):
        super().__init__()

        # VGG16 pretrained features
        VGG16 = vgg16(pretrained=vgg16_pretrained)
        # Separate VGG16 after pool layers to connect the skip connections
        self.block_pool0 = VGG16.features[:5]
        self.block_pool1 = VGG16.features[5:10]
        self.block_pool2 = VGG16.features[10:17]
        self.block_pool3 = VGG16.features[17:24]
        self.block_pool4 = VGG16.features[25:]

        self.conv1 = nn.Conv2d(512, 4096, 7)
        self.conv2 = nn.Conv2d(4096, 4096, 1)
        self.conv3 = nn.Conv2d(4096, 1000, 1)
        self.conv4 = nn.Conv2d(1000, 21, 1)

        self.deconv1 = nn.ConvTranspose2d(21, 21, 4, stride=2, padding=1)
        self.crop = nn.ConstantPad2d(-2, -2)
        self.deconv2 = nn.ConvTranspose2d(533, 21, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(533, 21, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(277, 21, 4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(149, 21, 4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(85, 21, 4, stride=2, padding=1)

    def forward(self, x):
        out_pool0 = self.block_pool0(x)
        out_pool1 = self.block_pool1(out_pool0)
        out_pool2 = self.block_pool2(out_pool1)
        out_pool3 = self.block_pool3(out_pool2)
        out_pool4 = self.block_pool4(out_pool3)
        x = F.relu(self.conv1(out_pool4))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.deconv1(x))
        x = self.crop(x)
        x = torch.cat((x, out_pool4), 1)
        x = F.relu(self.deconv2(x))
        x = torch.cat((x, out_pool3), 1)
        x = F.relu(self.deconv3(x))
        x = torch.cat((x, out_pool2), 1)
        x = F.relu(self.deconv4(x))
        x = torch.cat((x, out_pool1), 1)
        x = F.relu(self.deconv5(x))
        x = torch.cat((x, out_pool0), 1)
        x = F.relu(self.deconv6(x))
        return x

    def from_file(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == '__main__':
    model = FCNModel()
    x = randn(1, 3, 512, 512)
    model.forward(x)
