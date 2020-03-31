import os
import argparse
import numpy as np
import time
import copy
import pdb
import time
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import VocDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--voc-path', default='/home/work/yangfg/corpus/train-driver', help='data path')
    parser.add_argument('--class-path', default='./configs/classes.txt', help='class path')
    parser.add_argument('--checkpoint', default='./checkpoints/retinanet_9.pth', help='Path to model (.pt) file.')
    parser.add_argument('--output', default='./output', help='output images')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    dataset_val = VocDataset(args.voc_path, args.class_path, split='val',
                             transform=transforms.Compose([Normalizer(), Resizer()]))

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

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores.cpu()>0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.id2name[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                #print(label_name)

            print(data['prefix'])
            cv2.imwrite(os.path.join(args.output, data['prefix'][0]+'.jpg'), img)


if __name__ == '__main__':
 main()
