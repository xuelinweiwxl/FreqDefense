'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-21 14:56:44
LastEditTime: 2024-03-21 15:33:32
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/datasets/datautils.py
'''
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader

def getTransforms(datasetname):
    if datasetname == '20-imagenet':
        mean, std = getNormalizeParameter(datasetname)
        trans = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        return trans
    elif datasetname == 'imagenet':
        mean, std = getNormalizeParameter(datasetname)
        trans = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        return trans
    else:
        raise Exception('unknown dataset')

def getNormalizeParameter(datasetname):
    if datasetname == '20-imagenet':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if datasetname == 'imagenet':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise Exception('unknown dataset')

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
    if datasetname == 'imagnet':
        if train:
            root = root + '/imagnet/train'
            assert os.path.exists(root), f'path {root} not exists'
            dataset = ImageFolder(root, transform=getTransforms(datasetname))
        else:
            root = root + '/imagnet/val'
            assert os.path.exists(root), f'path {root} not exists'
            dataset = ImageFolder(root, transform=getTransforms(datasetname))
        return dataset
    else:
        raise Exception('unknown dataset') 

def getDataloader(datasetname, root, batch_size, num_works=4, train=True):
    dataset = getDataSet(datasetname, root, train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_works, pin_memory=True)
    return dataloader


def test():
    datasetname = '20-imagenet'
    root = '../data'
    dataloader = getDataloader(datasetname, root, 64, 4, True)
    testdataloader = getDataloader(datasetname, root, 64, 4, False)
    print(len(dataloader.dataset), len(testdataloader.dataset))
    from tqdm import tqdm
    for x, y in tqdm(dataloader):
        print(x.shape, y.shape, end='\r')
if __name__ == '__main__':
    test()