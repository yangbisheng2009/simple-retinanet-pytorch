from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2

class VocDataset(Dataset):
    def __init__(self, voc_path, classes, split='train', transform=None):
        """
        init data structure, as fllows:
        name2id: key=labelname, value=labelid
        id2name: x
        annos: key=the path of image, value=all image info
        image_paths: all image path list
        """
        self.name2id, self.id2name = self._load_class(classes)
        self.annos = {}
        self._load_annos(voc_path, split)
        self.image_paths = list(self.annos.keys())
        self.transform = transform

    def _load_class(self, classes):
        """load classes.txt, get two dict of name2id,id2name"""
        name2id = {}
        id2name = {}

        for idx, cls in enumerate(classes):
            name2id[cls] = idx
            id2name[idx] = cls

        return name2id, id2name

    def _load_annos(self, voc_path, split='train'):
        """load all anno file, get bbox, label and so on"""
        keys = [line.strip() for line in open(os.path.join(voc_path, 'ImageSets/Main', split+'.txt'))]

        for x in keys:
            xml_path = os.path.join(voc_path, 'Annotations', x+'.xml')
            anno = self._parse_single_xml(xml_path)
            jpg_path = os.path.join(os.path.join(voc_path, 'JPEGImages', x+'.jpg'))
            self.annos[jpg_path] = anno

    def _parse_single_xml(self, xml_path, use_difficult=False):
        """process single voc xml, get bbox,label and so on"""
        xml_anno = ET.parse(xml_path)
        anno = []
        difficult = []

        for obj in xml_anno.findall('object'):
            if not use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            xmin = int(bndbox_anno.find('xmin').text)
            ymin = int(bndbox_anno.find('ymin').text)
            xmax = int(bndbox_anno.find('xmax').text)
            ymax = int(bndbox_anno.find('ymax').text)
            class_name = obj.find('name').text.lower().strip()
            anno.append({'x1': xmin, 'x2': xmax, 'y1': ymin, 'y2': ymax, 'class': class_name})
        return anno

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        prefix = self.load_prefix(idx)
        sample = {'img': img, 'annot': annot, 'prefix': prefix}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_prefix(self, idx):
        return self.image_paths[idx].split('/')[-1][:-4]

    def load_image(self, image_index):
        #img = skimage.io.imread(self.image_paths[image_index])
        img = cv2.imread(self.image_paths[image_index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        '''
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        '''

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        annotation_list = self.annos[self.image_paths[image_index]]
        annotations = np.zeros((0, 5))

        if len(annotation_list) == 0:
            return annotations

        for idx, a in enumerate(annotation_list):
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name2id[a['class']]
            annotations = np.append(annotations, annotation, axis=0)

        return annotations


    def image_aspect_ratio(self, idx):
        img = cv2.imread(self.image_paths[idx])
        height, width, channel = img.shape
        return float(width) / float(height)

    def num_classes(self):
        return max(self.name2id.values()) + 1

    def name_to_id(self, name):
        return self.name2id[name]


def collater(data):
    """
    1.change data type from List to Dict
    2.change image paded
    """
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    prefix = [s['prefix'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'prefix': prefix}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots, prefix = sample['img'], sample['annot'], sample['prefix']

        # get scale
        H, W, C = image.shape
        scale1 = min_side / min(H, W)
        scale2 = max_side / max(H, W)
        scale = min(scale1, scale2)
        # resize the image with the computed scale
        #image = skimage.transform.resize(image, (int(round(H*scale)), int(round((W*scale)))))
        image = cv2.resize(image, (int(round((W*scale))), int(round(H*scale))), interpolation=cv2.INTER_LINEAR)

        # get new H, W, C
        H, W, C = image.shape
        pad_w = 32 - H % 32
        pad_h = 32 - W % 32
        new_image = np.zeros((H + pad_w, W + pad_h, C)).astype(np.float32)
        new_image[:H, :W, :] = image.astype(np.float32)
        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'prefix': prefix}


class Augmenter(object):
    """convert image by X"""
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots, prefix = sample['img'], sample['annot'], sample['prefix']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots, 'prefix': prefix}

        return sample


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots, prefix = sample['img'], sample['annot'], sample['prefix']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots, 'prefix': prefix}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        order = list(range(len(self.data_source)))
        #random.shuffle(order)
        # order the list by the width/height
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
