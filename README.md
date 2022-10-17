# 1.项目介绍
  第一次使用Github，上传有关人工智能安全的第一次作业，本次作业选用resnet18的模型，在CIFAR10数据集上完成训练和测试，根据所有数据集的特征进行分类。最终的精确度在81%左右
  
# 2.参数设置
  **本次作业中涉及到的超参分别有num_epochs 训练周期, lr 学习率, wd 正则化参数，lr_period 学习周期, lr_decay 学习衰减率，net选用模型**，经过调参后将它们的值分别设置为
  >num_epochs, lr, wd = try_all_gpus(), 15, 1e-3, 5e-3
  
  >lr_period, lr_decay, net = 5, 0.9, get_net()
  时模型能达到一个较好的训练效果
  
# 3.项目运行
  下载main.py文件并安装对应的拓展库后点击运行即可，如果没有数据集请将代码第32行处的download参数改为True

# 4.遇到问题
  在把文件上传至Github时需要先删除数据集，否则会提示上传文件过大，导致上传失败的错误
