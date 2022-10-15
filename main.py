import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch import nn
from torchvision import models
from torch.autograd import Variable

if __name__ == '__main__':
    transform_train = torchvision.transforms.Compose([
        # 在高度和宽度上将图像放大到40像素的正方形
        torchvision.transforms.Resize(40),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64到1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                 ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])])

    batch_size = 128  # 读取小批量

    trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2, drop_last=True)
    #num_worker 使用两个进程来读取数据
    testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                           download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2, drop_last=True)

    def try_all_gpus():
        """返回所有可用的GPU,如果没有GPU,则返回[cpu(),]"""
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        return devices if devices else [torch.device('cpu')]


    def get_net():
        net = models.resnet18(pretrained=True)  # 选用resnet18模型进行训练
        # for param in net.parameters():
        # param.requires_grad = False
        net_fit = net.fc.in_features
        net.fc = torch.nn.Linear(net_fit,10)
        # net = nn.Sequential(nn.Linear(net_fit, 10), nn.Softmax(dim=1))
        return net


    # net = get_net()

    # devices=torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
    # devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

    def accuracy(pred, target):
        correct = 0.
        pred_label = torch.argmax(pred, 1)  # 返回每张照片10个预测值里面最大数的标签
        # 如果这个标签等于真实标签，则将数值转化为个数，转化为float类型并返回给correct
        correct += (pred_label == target).to(torch.float).sum().item()
        return correct, len(pred)  # 返回正确的个数


    acc = {'train': [], "val": []}
    loss_all = {'train': [], "val": []}

    loss = nn.CrossEntropyLoss(reduction="none")
    best_acc = 0


    def evaluate_accuracy_gpu(net, data_iter, device=None):
        """使用GPU计算模型在数据集上的精度"""
        if isinstance(net, nn.Module): # 判断一个对象是否是已知类型
            net.eval()                 # 设置为评估模式
            if not device:
                device = next(iter(net.parameters())).device
        valid_loss, vaild_acc_sum, valid_prednum_sum = 0., 0., 0.
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(X, list):
                    # BERT微调所需的（之后将介绍）
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)
                outputs = net(X)
                l = loss(outputs, y)

                vaild_acc, valid_prednum = accuracy(outputs, y)
                vaild_acc_sum += vaild_acc
                valid_prednum_sum += valid_prednum
                valid_loss += l.sum()
        return valid_loss, vaild_acc_sum, valid_prednum_sum


    def train_batch_ch13(net, X, y, loss, trainer, devices):
        """用多GPU进行小批量训练"""
        if isinstance(X, list): #判断 X 是否为 List 类型
            X = [x.to(devices[0]) for x in X]
        else:
            X = X.to(devices[0])
        y = y.to(devices[0])
        net.train()
        trainer.zero_grad()
        pred = net(X)
        l = loss(pred, y)
        l.sum().backward() #反向传播
        # 目的就是计算权重、偏置等超参数的梯度，方便用优化算法更新参数时用到
        trainer.step()     #更新模型
        train_loss_sum = l.sum()
        with torch.no_grad():
            train_acc_sum, train_prednum = accuracy(pred, y)
        return train_loss_sum, train_acc_sum, train_prednum


    def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):

        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
        # weight_decay为权重衰减  amsgrad=True
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        # 优化lr,每隔 lr_period个epoch就给当前的lr乘以lr_decay
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period,lr_decay)

        for epoch in range(num_epochs):
            print('epoch', epoch + 1, '*******************************')
            # net.train()
            train_total_loss, train_correctnum, train_prednum = 0., 0., 0.
            for i, (features, labels) in enumerate(train_iter):

                l, acc, prednum = train_batch_ch13(net, features, labels, loss, optimizer, devices)
                train_total_loss += l
                train_correctnum += acc
                train_prednum += prednum
                if i % 32 == 31 or i == batch_size - 1:
                    print('train: [%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                          % (i, len(train_iter), train_total_loss / train_prednum, 100. * train_correctnum / train_prednum,
                             train_correctnum, train_prednum))

            if valid_iter is not None:
                valid_total_loss, valid_correctnum, valid_prednum = 0., 0., 0.
                valid_total_loss, valid_correctnum, valid_prednum = evaluate_accuracy_gpu(net, valid_iter)
                print('test: [%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (i, len(valid_iter), valid_total_loss / valid_prednum, 100. * valid_correctnum / valid_prednum,
                         valid_correctnum, valid_prednum))

            scheduler.step()
            train_loss = train_total_loss / train_prednum
            valid_loss = valid_total_loss / valid_prednum

            print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, valid_loss))
            print('Training Acc: {:.6f} \tValidation Acc: {:.6f}'.format(train_correctnum / train_prednum,
                                                                         valid_correctnum / valid_prednum))


    devices, num_epochs, lr, wd = try_all_gpus(), 15, 1e-3, 5e-3
    lr_period, lr_decay, net = 5, 0.9, get_net()
    train(net, trainloader, testloader, num_epochs, lr, wd, devices, lr_period, lr_decay)
