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
parser.add_argument('-p', '--project-path', help='project path')
parser.add_argument('--checkpoint', help='Path to model', type=str)
args = parser.parse_args()

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def main():
    params = Params(args.project_path)

    dataset_val = VocDataset(params.voc_path, params.classes, split='val',
                             transform=transforms.Compose([Normalizer(), Resizer()]))


    # Create the model
    retinanet = model.resnet(num_classes=dataset_val.num_classes(), pretrained=True, backbone=params.backbone)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(args.checkpoint))
        retinanet = torch.load(args.checkpoint)
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(args.checkpoint))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    #coco_eval.evaluate_coco(dataset_val, retinanet)
    mAP = voc_eval.evaluate(dataset_val, retinanet)


if __name__ == '__main__':
    main()