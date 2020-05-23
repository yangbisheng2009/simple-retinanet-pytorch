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

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from utils.dataloader import VocDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
parser.add_argument('--project_path', default='./configs/video1.yml', help='project path')
parser.add_argument('--input-images', default='./input-images/video1', help='input images')
parser.add_argument('--output-images', default='./output-images/video1', help='output images')
parser.add_argument('--checkpoint', default='checkpoints/video1/retinanet_28.pth', help='Path to model (.pt) file.')
parser.add_argument('--use-gpu', action='store_true', help='if use gpu')
args = parser.parse_args()

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def baseline():
    params = Params(args.project_path)

    dataset_val = VocDataset(params.voc_path, params.classes, split='val', transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = torch.load(args.checkpoint)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()

    unnormalize = UnNormalizer()

    ast = time.time()
    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            st = time.time()
            #print(data)
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            #print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores.cpu() > 0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
            print(img)

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                text = params.classes[classification[j].item()]

                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                #cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            cv2.imwrite(os.path.join(args.output_images, str(idx)+'.jpg'), img)
    print(time.time() - ast)

def main():
    params = Params(args.project_path)

    if not os.path.exists(args.output_images):
        os.mkdir(args.output_images)

    unnormalize = UnNormalizer()

    # 1. preprocess data
    tsfm = transforms.Compose([Normalizer(), Resizer()])

    # 2. load checkpoint
    retinanet = torch.load(args.checkpoint)
    '''
    if args.use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)
    '''
    retinanet = torch.nn.DataParallel(retinanet)
    retinanet.eval()

    st = time.time()
    for f in os.listdir(args.input_images):
        image = cv2.imread(os.path.join(args.input_images, f))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        sample = {'img': image, 'annot': np.zeros((1, 5)), 'prefix': ''}
        sample = tsfm(sample)
        sample = collater([sample])

        '''
        if torch.cuda.is_available():
            scores, classification, transformed_anchors = retinanet(sample['img'].cuda().float())
        else:
            scores, classification, transformed_anchors = retinanet(sample['img'].float())
        '''
        scores, classification, transformed_anchors = retinanet(sample['img'].float())

        #print('Elapsed time: {}'.format(time.time()-st))
        idxs = np.where(scores.cpu() > 0.5)
        img = np.array(255 * unnormalize(sample['img'][0, :, :, :])).copy()

        img[img < 0] = 0
        img[img > 255] = 255

        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        for j in range(idxs[0].shape[0]):
            text = params.classes[classification[j].item()]

            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imwrite(os.path.join(args.output_images, f), img)
    print(time.time() - st)
if __name__ == '__main__':
    #baseline()
    main()
