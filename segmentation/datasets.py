import os
from torch.utils import data

from PIL import Image
from scipy import io
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.autograd import Variable



SETS_INFOS = {
        'train': {
            'base_path': 'benchmark_RELEASE/dataset',
            'image_path': 'img',
            'labels_path': 'cls',
            'list_path': 'train.txt',
            'ext': '.mat'
            },
        'val': {
            'base_path': 'VOC2012',
            'image_path': 'JPEGImages',
            'labels_path': 'SegmentationClass',
            'list_path': 'ImageSets/Segmentation/seg11valid.txt',
            'ext': '.png'
            }
        }

class CIFAR:
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        print(trainset)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
        self.dataiter = iter(self.trainloader)

    def __getitem__(self, i):
        return self.dataiter.next()

    def __len__(self):
        return len(self.trainloader)




class PascalVOC2012(data.Dataset):
    def __init__(self, _set):
        self.images, self.labels = self._get_split_set(_set)
        self.transform_pipeline_input = transforms.Compose([transforms.Resize((512, 512)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225])])
        self.transform_pipeline_target = transforms.Compose([transforms.Resize((512, 512))])

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]

        image = Image.open(image_path).convert('RGB')
        label = self.load_label_as_mask_image(label_path)

        image = self.apply_transform_input(image)
        label = self.apply_transform_target(label)

        return (image, label)

    def __len__(self):
        return len(self.images)

    def apply_transform_target(self, label):
        x = self.transform_pipeline_target(label)
        x = torch.LongTensor(np.asarray(x))
        x = x.unsqueeze(0)
        return x

    def apply_transform_input(self, image):
        x = self.transform_pipeline_input(image)
        x = Variable(x)
        x = x.unsqueeze(0)
        return x


    @staticmethod
    def load_label_as_mask_image(label_path):
        _, ext =os.path.splitext(label_path)
        if ext == '.mat':
            label = io.loadmat(label_path)['GTcls']['Segmentation'][0][0]
            label = Image.fromarray(label.astype(np.uint8))
        elif ext == '.png':
            label = Image.open(label_path)
        return label

    def _get_split_set(self, _set):
        if _set not in SETS_INFOS:
            raise ValueError('Split set must be in {}'.format(SETS_PATH.keys()))

        try:
            filenames = [l.strip('\n') for l in open(self._get_path(_set, 'list_path')).readlines()]
        except Exception:
            raise Exception('Unable to load data for {} set'.format(_set))

        images = [os.path.join(self._get_path(_set, 'image_path'), filename + '.jpg') for filename in filenames]
        labels = [os.path.join(self._get_path(_set, 'labels_path'), filename + self._get_ext(_set)) for filename in filenames]
        return images, labels

    @staticmethod
    def _get_ext(_set):
        return '%s' % SETS_INFOS.get(_set).get('ext')

    @staticmethod
    def _get_path(_set, path_type):
        return os.path.join('/datasets', SETS_INFOS.get(_set).get('base_path', ''), SETS_INFOS.get(_set).get(path_type))

if __name__ == '__main__':
    cifar = CIFAR()
    print(len(cifar))
