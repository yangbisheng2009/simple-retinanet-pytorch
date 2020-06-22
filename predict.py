import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import yaml
import sys
import cv2
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from utils.dataloader import VocDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer, UnResizer
from utils import model


assert torch.__version__.split('.')[0] == '1'

os.environ["LRU_CACHE_CAPACITY"] = "1"
print('CUDA available: {}'.format(torch.cuda.is_available()))

parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
parser.add_argument('-p', '--project_path', default='./configs/video1.yml', help='project path')
parser.add_argument('--input-images', default='./input-images/video1', help='input images')
parser.add_argument('--output-images', default='./output-images/video1', help='output images')
parser.add_argument('--checkpoint', default='checkpoints/video1/retinanet_28.pth', help='Path to model (.pt) file.')
parser.add_argument('--backbone', default='resnet101', help='backbone.')
parser.add_argument('--use-gpu', action='store_true', help='if use gpu')
args = parser.parse_args()

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def get_all_colors(classes_num, seed=1):
    class_colors = []
    random.seed(seed)
    for _ in range(classes_num):
        class_colors.append(tuple([random.randint(0, 255) for _ in range(3)]))

    return class_colors

def main():
    params = Params(args.project_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_colors = get_all_colors(len(params.classes))

    if not os.path.exists(args.output_images):
        os.mkdir(args.output_images)

    unnormalize = UnNormalizer()
    unresizer = UnResizer()

    # 1. preprocess data
    tsfm = transforms.Compose([Normalizer(), Resizer()])

    # 2. load checkpoint
    retinanet = model.resnet(len(params.classes), pretrained=False, backbone=args.backbone)
    if torch.cuda.is_available():
        retinanet.load_state_dict(torch.load(args.checkpoint))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()

    st = time.time()
    for f in os.listdir(args.input_images):
        with torch.no_grad():
            per_st = time.time()
            image = cv2.imread(os.path.join(args.input_images, f))
            image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_ = image_.astype(np.float32) / 255.0

            sample = {'img': image_, 'annot': np.zeros((1, 5)), 'prefix': ''}
            sample = tsfm(sample)
            sample = collater([sample])

            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(sample['img'].cuda().float())
            else:
                #scores, classification, transformed_anchors = retinanet(sample['img'].float())
                scores, classification, transformed_anchors = retinanet(sample['img'].to(device))
            print('{} cost {}'.format(f, time.time() - per_st))
            
            idxs = np.where(scores.cpu() > 0.5)
            img = np.array(255 * unnormalize(sample['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            scale = unresizer.get_scale(image)
            for j in range(idxs[0].shape[0]):
                text = params.classes[classification[j].item()]
                color = class_colors[classification[j].item()]
                bbox = transformed_anchors[idxs[0][j], :]
                #xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                xmin, ymin, xmax, ymax = unresizer([bbox[0], bbox[1], bbox[2], bbox[3]], scale)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
                cv2.putText(image, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.imwrite(os.path.join(args.output_images, f), image)

    print(time.time() - st)

if __name__ == '__main__':
    main()
