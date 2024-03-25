'''
Author: Xuelin Wei
Email: xuelinwei@seu.edu.cn
Date: 2024-03-25 10:36:30
LastEditTime: 2024-03-25 10:36:44
LastEditors: xuelinwei xuelinwei@seu.edu.cn
FilePath: /FreqDefense/utils/utils.py
'''
class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)