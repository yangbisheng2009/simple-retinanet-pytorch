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
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from utils.dataloader import collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torchvision import transforms as T
assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

# threshold for class score
threshold = 0.05

def main():
    parser = argparse.ArgumentParser(description='retinanet predict.')
    parser.add_argument('--input', default='/home/work/yangfg/corpus/traffic-cars-count/test/', help='input images')
    parser.add_argument('--output', default='./output', help='output images')
    parser.add_argument('--backbone', default='resnet101', help='backbone')
    parser.add_argument('--checkpoint', default='./checkpoints/retinanet_4.pth', help='checkpoint')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    retinanet = torch.load(args.checkpoint)
    retinanet = retinanet.cuda()
    retinanet.eval()
    transforms = T.Compose([Normalizer(), Resizer()])

    for f in os.listdir(args.input):
        file_path = os.path.join(args.input, f)
        #image = skimage.io.imread(file_path)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sampler = {"img": image.astype(np.float32)/255.0, "annot": np.empty(shape=(5,5)), 'prefix': f[:-4]}
        image_tf = transforms(sampler)
        scale = image_tf["scale"]
        new_shape = image_tf['img'].shape
        x = torch.autograd.Variable(image_tf['img'].unsqueeze(0).transpose(1,3), volatile=True)
        with torch.no_grad():
            scores,_,bboxes = retinanet(x.cuda().float())
            bboxes /= scale
            scores = scores.cpu().data.numpy()
            print(len(scores))
            print(scores)
            bboxes = bboxes.cpu().data.numpy()
            # select threshold
            idxs = np.where(scores > threshold)[0]
            scores = scores[idxs]
            bboxes = bboxes[idxs]
            print(len(bboxes))
            #embed()

            #image =  cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
            image = cv2.imread(file_path)
            for i,box in enumerate(bboxes):
                 cv2.rectangle(image,(int(box[1]),int(box[0])),(int(box[3]),int(box[2])),color=(0,0,255),thickness=2)

            cv2.imwrite(os.path.join(args.output, f), image)
if __name__ == "__main__":
    main()
