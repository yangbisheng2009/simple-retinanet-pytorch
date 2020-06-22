# simple retinanet pytorch version

[中文](./README_cn.md) | [English](./README.md)

## What's simple?
I do not want to show COCO performance. For many project can get high score in COCO dataset. But when using them in actual scene, usually work not very well.  
Why? Parameters are just suit for COCO dataset(classes number or highyperparameter).  
So, I create this project which can suit almost all scenes. If you are new to object detection, this project is suitable for you. You needn't to adjust the parameters for a long time.  

## Useage
```
# train
nohup python train.py --project-path confgs/xxx.yml --batch-size 16 --backbone resnet101 --train-vision --plt-env mouse_detect --checkpoints checkpoints/ --epochs 100
# predict
python predict.py --input intput-images --output output-images --checkpoint checkpoints/xxx/retinanet_99.pth --project-file confgs/xxx.yml
```
## Examples
### Mouse Detection(single class)
<p align="left">
    <img width=350 height=250 src="https://github.com/yangbisheng2009/simple-retinanet-pytorch/blob/master/images/1.jpg">
    <img width=350 height=250 src="https://github.com/yangbisheng2009/simple-retinanet-pytorch/blob/master/images/2.jpg">
</p>
<p align="left">
    <img width=350 height=250 src="https://github.com/yangbisheng2009/simple-retinanet-pytorch/blob/master/images/3.jpg">
    <img width=350 height=250 src="https://github.com/yangbisheng2009/simple-retinanet-pytorch/blob/master/images/4.jpg">
</p>
<p align="left">
    <img width=350 height=250 src="https://github.com/yangbisheng2009/simple-retinanet-pytorch/blob/master/images/5.jpg">
    <img width=350 height=250 src="https://github.com/yangbisheng2009/simple-retinanet-pytorch/blob/master/images/6.jpg">
</p>

## Tricks
1. Prefer to use single GPU.
2. The larger the batch size, the better.
3. In your work, you can try resnet50 and resnet101 at the same time(for single GPU).

## References
https://github.com/chenyuntc/simple-faster-rcnn-pytorch  
https://github.com/yhenon/pytorch-retinanet  

## ChangeLog
[2020-03-06]: First commit.  
[2020-03-10]: Modify dataloader(COCO to VOC).  
[2020-04-20]: Add yaml config.  
[2020-05-26]: Modify the method of save model so that I can load the model everywhere. torch.save(model) -> torch.save(model.state_dict())  
[2020-06-20]: Modify predict resize method to keep the same size between input images and output images.  
[2020-06-22]: Solve CPU memory leak problem.  

## TODO
1. Create a web server which can run my model.
2. Add industry object detection examples continously.