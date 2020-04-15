import os
import argparse
import collections
import numpy as np
import traceback

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import model
from utils import csv_eval
from utils.dataloader import VocDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer

# support pytorch 1.0+
assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--voc-path', default='/home/work/yangfg/corpus/mouse', help='data path')
    parser.add_argument('--class-path', default='./configs/classes.txt', help='class path')
    parser.add_argument('--backbone', default='resnet101', help='backbone')
    parser.add_argument('--epochs', default=100, type=int, help='epoch number')
    parser.add_argument('--checkpoints', default='./checkpoints', help='checkpoints')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    args = parser.parse_args()

    dataset_train = VocDataset(args.voc_path, args.class_path, split='train',
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = VocDataset(args.voc_path, args.class_path, split='val',
                             transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=args.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler, shuffle=True)
    #dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = model.resnet(num_classes=dataset_train.num_classes(), pretrained=True, backbone=args.backbone)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        # retinanet = torch.nn.DataParallel(retinanet)
        pass

    #retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(args.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except:
                traceback.print_exc()

        print('Evaluating dataset')
        mAP = csv_eval.evaluate(dataset_val, retinanet)
        scheduler.step(np.mean(epoch_loss))

        print('===> save one epoch')
        torch.save(retinanet.module, os.path.join(args.checkpoints, 'retinanet_{}.pth'.format(epoch_num)))

    retinanet.eval()
    torch.save(retinanet, 'model_final.pth')


if __name__ == '__main__':
    main()
