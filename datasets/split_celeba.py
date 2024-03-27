'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-27 19:31:33
LastEditTime: 2024-03-27 20:37:47
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/datasets/split_celeba.py
Comment: Seperate the CelebA dataset into training and testing set.
'''

import os
import shutil

celeba_root = '../data/img_align_celeba'
split_root = '../data/celeba'
# list all files in the root directory
all_files = os.listdir(celeba_root)
with open('../data/list_eval_partition.txt', 'r') as f:
    line = f.readline()
    while line:
        img_name, partition = line.strip().split(' ')
        if partition == '0':
            target_dir = os.path.join(split_root, 'train', '0')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.copy(os.path.join(celeba_root, img_name), os.path.join(target_dir, img_name))
        elif partition == '1':
            target_dir = os.path.join(split_root, 'val', '0')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.copy(os.path.join(celeba_root, img_name), os.path.join(target_dir, img_name))
        elif partition == '2':
            target_dir = os.path.join(split_root, 'test', '0')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            shutil.copy(os.path.join(celeba_root, img_name), os.path.join(target_dir, img_name))
        else:
            raise Exception('unknown partition')
        line = f.readline()