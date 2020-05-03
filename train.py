import os
import argparse
import collections
import numpy as np
import traceback
import yaml

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import model
from utils import csv_eval
from utils.dataloader import VocDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from utils.vistool import VisTool

# support pytorch 1.0+
assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
parser.add_argument('-p', '--project-path', default='./configs/mouse.yml', help='project file')
parser.add_argument('--backbone', default='resnet101', help='backbone')
parser.add_argument('--epochs', default=100, type=int, help='epoch number')
parser.add_argument('--checkpoints', default='./checkpoints', help='checkpoints')
parser.add_argument('--batch-size', default=8, type=int, help='batch size')
parser.add_argument('--plot-freq', default=1, type=int, help='plot one image every freq batchs')
parser.add_argument('--train-vision', action='store_true', help='if plot image')
parser.add_argument('--plot-env', default='default', help='plot image title')
args = parser.parse_args()

print(args)

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def main():
    params = Params(args.project_path)
    project_name = params.project_name
    
    save_path = os.path.join(args.checkpoints, project_name)
    os.makedirs(save_path, exist_ok=True)

    dataset_train = VocDataset(params.voc_path, params.classes, split='train',
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = VocDataset(params.voc_path, params.classes, split='val',
                             transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=args.batch_size, drop_last=False)
    #dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler, shuffle=True)
    dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=4, collate_fn=collater, batch_sampler=sampler_val)

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

    if args.train_vision:
        vistool = VisTool(args.plot_env, params.classes)

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

                print('Epoch: {} / {} | Iteration: {} / {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | '
                      'Running loss: {:1.5f}'.format(epoch_num, args.epochs, iter_num, len(sampler.groups),
                      float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except:
                traceback.print_exc()

        print('Evaluating dataset')

        if args.train_vision:
            mAP = csv_eval.evaluate(dataset_val, retinanet, vistool=vistool)
        else:
            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        print('*** SAVE ONE EPOCH ***')
        torch.save(retinanet.module, os.path.join(save_path, 'retinanet_{}.pth'.format(epoch_num)))

    retinanet.eval()
    torch.save(retinanet, 'model_final.pth')


if __name__ == '__main__':
    main()
