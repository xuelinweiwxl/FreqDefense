'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-21 14:56:44
LastEditTime: 2024-04-23 16:45:22
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/datasets/datautils.py
'''

'''
If you need to load your own dataset, all parameter related to the dataset should be defined in this file.
'''

import os
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader

def getNormalizeParameter(datasetname):
    if datasetname == '20-imagenet':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if datasetname == 'imagenet':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if datasetname == 'celeba':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if datasetname == 'cifar10':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise Exception('unknown dataset')

def getImageSize(datasetname):
    if datasetname == '20-imagenet':
        return 3, 224
    if datasetname == 'imagenet':
        return 3, 224
    if datasetname == 'celeba':
        return 3, 224
    if datasetname == 'cifar10':
        return 3, 32
    else:
        raise Exception('unknown dataset')

def getTransforms(datasetname):
    mean, std = getNormalizeParameter(datasetname)
    _, size = getImageSize(datasetname)
    trans = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    return trans

def getDataSet(datasetname, root, train=True):
    if datasetname == '20-imagenet':
        if train:
            root = root + '/20-imagenet/train'
            assert os.path.exists(root), f'path {root} not exists'
            dataset = ImageFolder(root, transform=getTransforms(datasetname))
        else:
            root = root + '/20-imagenet/val'
            assert os.path.exists(root), f'path {root} not exists'
            dataset = ImageFolder(root, transform=getTransforms(datasetname))
        return dataset
    if datasetname == 'imagenet':
        if train:
            root = root + '/imagenet/train'
            assert os.path.exists(root), f'path {root} not exists'
            dataset = ImageFolder(root, transform=getTransforms(datasetname))
        else:
            root = root + '/imagenet/val'
            assert os.path.exists(root), f'path {root} not exists'
            dataset = ImageFolder(root, transform=getTransforms(datasetname))
        return dataset
    if datasetname == 'celeba':
        if train:
            root = root + '/celeba/train'
            assert os.path.exists(root), f'path {root} not exists'
            dataset = ImageFolder(root, transform=getTransforms(datasetname))
        else:
            root = root + '/celeba/test'
            assert os.path.exists(root), f'path {root} not exists'
            dataset = ImageFolder(root, transform=getTransforms(datasetname))
        return dataset
    if datasetname == 'cifar10':
        dataset = CIFAR10(root, train=train, transform=getTransforms(datasetname), download=True)
        return dataset
    else:
        raise Exception('unknown dataset') 

def getDataloader(datasetname, root, batch_size, num_works=4, train=True):
    dataset = getDataSet(datasetname, root, train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_works, pin_memory=True)
    return dataloader

def test():
    datasetname = 'cifar10'
    root = '../data'
    dataloader = getDataloader(datasetname, root, 200, 4, True)
    testdataloader = getDataloader(datasetname, root, 200, 4, False)
    print(len(dataloader.dataset), len(testdataloader.dataset))
    from tqdm import tqdm
    toPIL = T.ToPILImage()
    trans = T.Compose([
        T.Resize(32),
        T.CenterCrop(32),
        T.ToTensor()
    ])
    dataloader.dataset.transform = trans
    for x, y in tqdm(dataloader):
        x = x[111,:,:,:]
        x = toPIL(x)
        x.save('../test3.png')
        break
if __name__ == '__main__':
    test()