import argparse
import yaml

import torch
from torchvision import transforms

from utils import model
from utils.dataloader import VocDataset, Resizer, Normalizer
from utils import voc_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
parser.add_argument('-p', '--project-path', default='./configs/video1.yml', help='project path')
parser.add_argument('--checkpoint', default='./checkpoints/video1/retinanet_28.pth', help='Path to model (.pt) file.')
args = parser.parse_args()

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def main():
    params = Params(args.project_path)

    # 1. prepare data
    dataset_val = VocDataset(params.voc_path, params.classes, split='val',
                             transform=transforms.Compose([Normalizer(), Resizer()]))

    # 2. prepare model
    retinanet = torch.load(args.checkpoint)
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)
    retinanet.eval()

    mAP = voc_eval.evaluate(dataset_val, retinanet)


if __name__ == '__main__':
    main()