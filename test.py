import os
import numpy as np
import random

def gen_test_data(num_data, num_classes):
    mat = [[np.zeros(0) for i in range(num_classes)] for j in range(num_data)]

    for x in range(num_data):
        p = random.random()
        
        # 高概率有框
        if p < 0.9:
            for xx in range(num_classes):
                p_ = random.randint(1, 15)

                # 低概率有框
                if p_ < 4:
                    #mat[x][xx] = np.random.uniform(1, 600, (p_, 4))
                    mat[x][xx] = np.concatenate([np.random.uniform(1,200, (p_, 2)), np.random.uniform(300, 400, (p_, 2))], axis=1)
                # 高概率无框
                else:
                    continue
        # 没框直接赋值为空
        else:
            continue
        
    return mat

def gen_gt_data(num_data, num_classes):
    mat = [[np.zeros(0) for i in range(num_classes)] for j in range(num_data)]

    for x in range(num_data):
        p = random.random()
        
        ## 高概率有框
        #if p < 0.9:
        if True:
            for xx in range(num_classes):
                p_ = random.randint(1, 15)

                # 低概率有框
                #if p_ < 4:
                if True:
                    #mat[x][xx] = np.random.uniform(1, 600, (p_, 4))
                    mat[x][xx] = np.concatenate([np.random.uniform(1,200, (p_, 2)), np.random.uniform(300, 400, (p_, 2))], axis=1)
                # 高概率无框
                else:
                    continue
        # 没框直接赋值为空
        else:
            continue
        
    return mat

def evaluate(preds, gts, iou_thres=0.5, max_detections=100):
    num_cls = len(preds[0])

    precision_molecule = np.zeros(num_cls)
    precision_denominator = np.zeros(num_cls)

    recall_molecule = np.zeros(num_cls)
    recall_denominator = np.zeros(num_cls)
    overlap_molecule = np.zeros(num_cls)

    for i, one_data in enumerate(preds):
        for j, one_label in enumerate(one_data):
            pred_num = one_label.shape[0]
            gt_num = gts[i][j].shape[0]


            precision_denominator[j] += pred_num
            recall_denominator[j] += gt_num

            if gt_num == 0 and pred_num > 0:
                pass
            elif gt_num > 0 and pred_num == 0:
                pass
            elif gt_num == 0 and pred_num == 0:
                continue
            elif gt_num > 0 and pred_num > 0:
                # create a confusion matrix by pred box and gt box
                # just like this:
                #     preds
                #---------------
                # gts|  1,0,1,1
                #    |  0,1,0,0
                #---------------
                # precision
                confusion_matrix = [[bbox_iou(one_pred_box[:4], one_gt_box[:4]) for one_pred_box in one_label] for one_gt_box in gts[i][j]]
                #print(confusion_matrix)

                for x in confusion_matrix:
                    # prec ++
                    for xx in x:
                        if xx >= iou_thres:
                            precision_molecule[j] += 1

                    # recal ++
                    if max(x) > iou_thres:
                        recall_molecule[j] += 1

                    # overlap ++
                    overlap_cnt = len([xx for xx in x if xx >= iou_thres]) - 1
                    if overlap_cnt >= 1:
                        overlap_molecule[j] += overlap_cnt

    precision = precision_molecule / precision_denominator
    recall = recall_molecule / recall_denominator
    overlap = overlap_molecule / recall_denominator

    print(f'precision_molecule: {precision_molecule}, precision_denominator: {precision_denominator}')
    print(f'recall_molecule:{recall_molecule}, recall_molecule: {recall_molecule}')
    print(f'overlap_molecule: {overlap_molecule}, recall_denominator: {recall_denominator}')

    print(precision, recall, overlap)
    return precision, recall

def bbox_iou(bbox_a, bbox_b):
    assert bbox_a.shape[0] == 4
    assert bbox_b.shape[0] == 4

    # top left
    tl = np.maximum(bbox_a[:2], bbox_b[:2])
    # bottom right
    br = np.minimum(bbox_a[2:], bbox_b[2:])


    area_i = np.prod(br - tl) * (tl <= br).all()
    area_a = np.prod(bbox_a[2:] - bbox_a[:2])
    area_b = np.prod(bbox_b[2:] - bbox_b[:2])
    iou = area_i / (area_a + area_b - area_i)
    return iou


def gen_():
    preds =  [[np.array([[601, 604, 620, 623], [590, 610, 633, 633]]), np.array([])], 
              [np.array([]), np.array([[301, 420, 399, 564], [290, 458, 430, 600]])], ]
    gts =  [[np.array([[600, 601, 623, 623]]), np.array([])], 
            [np.array([]), np.array([[300, 421, 400, 566]])],]
   
    return preds, gts

if __name__ == '__main__':
    preds = gen_test_data(5, 1)
    gts = gen_gt_data(5, 1)
    
    #print(bbox_iou(np.array([235.34687846,277.88420236,195.06307235,563.12906109]), np.array([235.34687846,277.88420236,195.06307235,563.12906109])))
    #print(bbox_iou(np.array([1,1,4,4]), np.array([2,2,3,3])))
    #print(f'preds:\n{preds}\ngts:\n{gts}\n')
    preds, gts = gen_()
    evaluate(preds, gts)

    #evaluate(preds, preds)
