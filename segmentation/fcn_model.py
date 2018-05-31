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
        self.vgg_pool1 = VGG16.features[:5]
        self.vgg_pool2 = VGG16.features[5:10]
        self.vgg_pool3 = VGG16.features[10:17]
        self.vgg_pool4 = VGG16.features[17:24]
        self.vgg_pool5 = VGG16.features[25:]

        # Fully conv
        self.conv1 = nn.Conv2d(512, 4096, 7)
        # TODO: add dropout
        self.conv2 = nn.Conv2d(4096, 4096, 1)
        # TODO: add dropout
        self.conv3 = nn.Conv2d(4096, 21, 1)
        self.deconv1 = nn.ConvTranspose2d(21, 21, 4, stride=2)

        self.conv_0 = nn.Conv2d(512, 21, 1, padding=0)
        # TODO add correct padding value
        self.crop_0 = nn.ZeroPad2d(-5)
        self.deconv_0 = nn.ConvTranspose2d(21, 21, 4, stride=2)

        self.conv_1 = nn.Conv2d(256, 21, 1, padding=0)
        # TODO add correct padding value
        self.crop_1 = nn.ZeroPad2d(-9)
        self.deconv_1 = nn.ConvTranspose2d(21, 21, 16, stride=11)

        self.crop_f = nn.ZeroPad2d((1, 0, 1, 0))
        self.softmax = nn.Softmax()

    def forward(self, x):
        # VGG 16
        out_pool1 = self.vgg_pool1(x)
        out_pool2 = self.vgg_pool2(out_pool1)
        out_pool3 = self.vgg_pool3(out_pool2)
        out_pool4 = self.vgg_pool4(out_pool3)
        out_pool5 = self.vgg_pool5(out_pool4)

        # Fully conv
        x = F.relu(self.conv1(out_pool5))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        upscore2 = self.deconv1(x)

        # First skip connection
        score_pool4 = self.conv_0(out_pool4)
        score_pool4c = self.crop_0(score_pool4)
        fuse_pool4 = score_pool4c + upscore2
        upscore_pool4 = self.deconv_0(fuse_pool4)

        # Second skip connection
        score_pool3 = self.conv_1(out_pool3)
        score_pool3c = self.crop_1(score_pool3)
        fuse_pool3 = score_pool3c + upscore_pool4
        upscore8 = self.deconv_1(fuse_pool3)

        # Crop final result and apply softmax
        x = self.crop_f(upscore8)
        x = self.softmax(x)
        return x

    def from_file(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == '__main__':
    model = FCNModel()
    x = randn(1, 3, 512, 512)
    model.forward(x)
