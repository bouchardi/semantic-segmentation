import torch
import torch.nn as nn
from torch import randn
import torch.nn.functional as F
from torchvision.models import vgg16

class FCNModel(nn.Module):
    def __init__(self, n_classes=21, vgg16_pretrained=False):
        super().__init__()

        self.padding = nn.ZeroPad2d(100)

        # VGG16 pretrained features
        VGG16 = vgg16(pretrained=vgg16_pretrained)
        # Separate VGG16 after pool layers to connect the skip connections
        self.vgg_pool1 = VGG16.features[:5]
        self.vgg_pool2 = VGG16.features[5:10]
        self.vgg_pool3 = VGG16.features[10:17]
        self.vgg_pool4 = VGG16.features[17:24]
        self.vgg_pool5 = VGG16.features[25:]

        # Fully conv
        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(4096, n_classes, 1),
            nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2)
            )


        self.conv_0 = nn.Conv2d(512, n_classes, 1, padding=0)
        self.deconv_0 = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2)

        self.conv_1 = nn.Conv2d(256, n_classes, 1, padding=0)
        self.deconv_1 = nn.ConvTranspose2d(n_classes, n_classes, 16, stride=8)

        for layer in [self.fc[-2], self.conv_0, self.conv_1]:
            self.initialize_weights_to_zero(layer)

    def initialize_weights_to_zero(self, layer):
        layer.weight.data.zero_()
        layer.bias.data.zero_()

    def forward(self, x):
        n,c,h,w = x.size()
        # Padding
        x = self.padding(x)

        # VGG 16
        out_pool1 = self.vgg_pool1(x)
        out_pool2 = self.vgg_pool2(out_pool1)
        out_pool3 = self.vgg_pool3(out_pool2)
        out_pool4 = self.vgg_pool4(out_pool3)
        out_pool5 = self.vgg_pool5(out_pool4)

        # Fully conv
        upscore2 = self.fc(out_pool5)

        # First skip connection
        score_pool4 = self.conv_0(out_pool4)
        score_pool4c = score_pool4[:, :, 5:5+upscore2.size(2),
                                         5:5+upscore2.size(3)]
        fuse_pool4 = score_pool4c + upscore2
        upscore_pool4 = self.deconv_0(fuse_pool4)

        # Second skip connection
        score_pool3 = self.conv_1(out_pool3)
        score_pool3c = score_pool3[:, :, 9:9+upscore_pool4.size(2),
                                         9:9+upscore_pool4.size(3)]
        fuse_pool3 = score_pool3c + upscore_pool4
        output = self.deconv_1(fuse_pool3)

        return output[:, :, 31: (31 + h), 31: (31 + w)].contiguous()

    def from_file(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == '__main__':
    model = FCNModel()
    x = randn(1, 3, 512, 512)
    model.forward(x)
    print(model.eval())
