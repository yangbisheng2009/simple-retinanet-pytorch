import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2
import skimage
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from utils.dataloader import collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torchvision import transforms as T
assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

# threshold for class score
threshold = 0.5

def main():
    parser = argparse.ArgumentParser(description='retinanet predict.')
    parser.add_argument('--input', default='/home/work/yangfg/corpus/traffic-cars-count/test/', help='input images')
    parser.add_argument('--output', default='./output', help='output images')
    parser.add_argument('--backbone', default='resnet101', help='backbone')
    parser.add_argument('--class-path', default='./configs/classes.txt', help='class path')
    parser.add_argument('--checkpoint', default='./checkpoints/retinanet_3.pth', help='checkpoint')
    args = parser.parse_args()

    retinanet = torch.load(args.checkpoint)
    retinanet = retinanet.cuda()
    retinanet.eval()
    transforms = T.Compose([Normalizer(), Resizer()])

    for f in os.listdir(args.input):
        file_path = os.path.join(args.input, f)
        image = skimage.io.imread(file_path)
        print(image.shape)

        sampler = {"img": image.astype(np.float32)/255.0, "annot": np.empty(shape=(5,5)), 'prefix': f[:-4]}
        image_tf = transforms(sampler)
        scale = image_tf["scale"]
        new_shape = image_tf['img'].shape
        x = torch.autograd.Variable(image_tf['img'].unsqueeze(0).transpose(1,3), volatile=True)
        with torch.no_grad():
            scores,_,bboxes = retinanet(x.cuda().float())
            bboxes /= scale
            scores = scores.cpu().data.numpy()
            bboxes = bboxes.cpu().data.numpy()
            # select threshold
            idxs = np.where(scores > threshold)[0]
            scores = scores[idxs]
            bboxes = bboxes[idxs]
            #embed()

            #image =  cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            image = cv2.imread(file_path)
            for i,box in enumerate(bboxes):
                 cv2.rectangle(image,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),color=(0,0,255),thickness=2)
            print("Predicting image: {}".format(x))
            print(image.shape)
            print(type(image))
            print(bboxes)

            cv2.imwrite(os.path.join(args.output, f), image)
if __name__ == "__main__":
    main()
