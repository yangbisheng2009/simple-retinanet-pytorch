# simple retinanet pytorch version

[中文](./README_cn.md) | [English](./README.md)

## What's simple?


## Useage
```
# train
nohup python train.py --project-path confgs/xxx.yml --batch-size 16 --backbone resnet101 --train-vision --plt-env mouse_detect --checkpoints checkpoints/ --epochs 100
# predict
python predict.py --input intput-images --output output-images --checkpoint checkpoints/xxx/retinanet_99.pth --project-file confgs/xxx.yml
```
## References
https://github.com/chenyuntc/simple-faster-rcnn-pytorch  
https://github.com/yhenon/pytorch-retinanet  

## ChangeLog
2020-03-06: First commit.  
2020-03-10: Modify dataloader(COCO to VOC).  
2020-04-20: Add yaml config.  
2020-05-26: Modify the method of save model so that I can load the model everywhere. torch.save(model) -> torch.save(model.state_dict())  
2020-06-20: Modify predict resize method to keep the same size between input images and output images.  
