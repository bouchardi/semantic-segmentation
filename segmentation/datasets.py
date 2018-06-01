import os
from torch.utils import data

from PIL import Image
from scipy import io
import numpy as np
from scipy.misc import imresize
from skimage import io as sio

import torch
import torchvision
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision.transforms import functional as F


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
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def pre_process(image, label):
    image = np.array(image)
    label = np.array(label)

    image = imresize(image, (512, 512, 3), interp='bilinear')
    label = imresize(label, (512, 512), interp='nearest')

    # Pytorch compatible
    image = np.transpose(image, (2, 0, 1))

    image = torch.FloatTensor(image)
    label = torch.LongTensor(label)

    image = F.normalize(image, _MEAN, _STD)
    return image, label


class PascalVOC2012(data.Dataset):
    def __init__(self, _set):
        self.images, self.labels = self._get_split_set(_set)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]

        image = Image.open(image_path).convert('RGB')
        label = self.load_label_as_mask_image(label_path)

        image, label = pre_process(image, label)
        return {'image': image,
                'label': label}

    def __len__(self):
        return len(self.images)

    def get_classes(self):
       return ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    def get_ignored_class(self):
        return 255

    def get_classes_count(self):
        return len(self.get_classes())

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
        return os.path.join('/datasets/pascal', SETS_INFOS.get(_set).get('base_path', ''), SETS_INFOS.get(_set).get(path_type))



class CamVid(data.Dataset):

    def __init__(self, _set='val'):
        self.images, self.labels = self._get_split_set(_set)


    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]

        image = sio.imread(image_path)
        label = sio.imread(label_path)

        image, label = pre_process(image, label)

        return {'image': image,
                'label': label}

    def __len__(self):
        return len(self.images)

    def get_classes(self):
        return ['sky', 'building', 'column_pole', 'road', 'sidewalk', 'tree',
                'sign', 'fence', 'car', 'pedestrian', 'byciclist']

    def get_ignored_class(self):
        return 11

    def get_classes_count(self):
        # -1 for the ignored class
        return len(self.get_classes())

    def _get_split_set(self, _set):
        base_path = '/datasets/camvid'
        # Get file names for this set and year
        images = []
        labels = []
        with open(os.path.join(base_path, _set + '.txt')) as f:
            for fi in f.readlines():
                raw_name = fi.strip()
                raw_name = raw_name.split("/")[4]
                raw_name = raw_name.strip()
                images.append(os.path.join(base_path, '{}'.format(_set), raw_name))
                labels.append(os.path.join(base_path, '{}annot'.format(_set), raw_name))
        return images, labels
