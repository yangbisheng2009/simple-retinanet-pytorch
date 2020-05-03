import os
import sys
import argparse
import cv2
import numpy as np
import visdom
import matplotlib.pyplot as plt


'''
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

parser = argparse.ArgumentParser('Create a train vis project.')
parser.add_argument('--env', default='default', help='enviroment')
parser.add_argument('--images', default='./images', help='input test image')
args = parser.parse_args()

ALL_LABELS = ['car', 'person']
'''


class VisTool(object):
    """
    visdom后端画图类
    """
    def __init__(self, env, labels, **kwargs):
        """
        初始化visdom，得到visdom类，命名
        """
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self.vis.text(labels, 'labels')

    def draw(self, img, boxes, clss, title, **kwargs):
        """
        img: opencv打开图像的原始格式，numpy类型， HWC、BGR原始格式，
        boxes: list格式，二维数组，单图片对应的所有box
        clss: 类别list
        title: window名称，显示名称
        """
        for idx, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = [int(coo) for coo in box]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=1)
            #cv2.putText(img, clss[idx], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            #cv2.putText(img, clss[idx], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:, :, ::-1].transpose((2, 0, 1))

        self.vis.images(img, win=title, opts={'title': title}, **kwargs)

'''
if __name__ == '__main__':
    vistool = VisTool(args.env, ALL_LABELS)

    for image_name in os.listdir(args.images):
        path = os.path.join(args.images, image_name)
        img = cv2.imread(path)

        boxes = [[20, 60, 80, 100], [60, 100, 100, 200]]
        clss = ['car', 'person']
        title = 'gt_img'
        vistool.draw(img, boxes, clss, title)
'''
