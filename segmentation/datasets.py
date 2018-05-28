import os
from torch.utils import data

from PIL import Image
from scipy import io
import numpy as np

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

class PascalVOC2012(data.Dataset):
    def __init__(self, _set):
        self.images, self.labels = self._get_split_set(_set)


    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]

        image = Image.open(image_path).convert('RGB')
        label = self.load_label_as_mask_image(label_path)

        return {'images': [image],
                'labels': [label],
                'index': index}

    def __len__(self):
        return len(self.images)

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