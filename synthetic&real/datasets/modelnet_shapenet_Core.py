'''
Author: your name
Date: 2021-03-30 11:02:39
LastEditTime: 2021-04-23 10:52:25
LastEditors: ze bai
Description: In User Settings Edit
FilePath: /multimodel/datasets/modelnet_shapenet_Core.py
'''
import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os


class _ShapeNetCore(Dataset):
    def __init__(self, dataroot, pattern):
        super().__init__()
        self.dataroot = dataroot
        splitfile = os.path.join(dataroot, 'train_test_split', f'shuffled_{pattern}_file_list.json')
        classnum2labelfile = os.path.join(dataroot, 'shapenet_classnum2label.json')
        # read split file
        with open(splitfile, 'r') as f:
            load_dict = json.load(f)
        # filename add to self.file
        self.filename = load_dict
        with open(classnum2labelfile, 'r') as f:
            self.classnum2label = json.load(f)
    
    def __getitem__(self, index):
        #self.file
        #return a dict
        #for modelnet40: point, normal, classlabel
        #for shapenet: point, normal, classlabel, denselabel
        sevenD= np.loadtxt(os.path.join(self.dataroot, self.filename[index][11:] + '.txt'), delimiter=' ')
        classlabel = self.classnum2label[self.filename[index].split('/')[1]]
        denselabel = sevenD[:, -1]
        point = sevenD[:, :3]
        normal = sevenD[:, 3:6]
        return {'point':point, 'normal':normal, 'classlabel':classlabel, 'denselabel':denselabel}
    def __len__(self):
        return len(self.filename)

class _ModelNet40Core(Dataset):
    def __init__(self, dataroot, pattern):
        super().__init__()
        self.dataroot = dataroot
        splitfile = os.path.join(dataroot, f'modelnet40_{pattern}.json')
        classnum2labelfile = os.path.join(dataroot, 'modelnet40_classnum2label.json')
        # read split file
        with open(splitfile, 'r') as f:
            load_dict = json.load(f)
        # filename add to self.file
        self.filename = load_dict
        with open(classnum2labelfile, 'r') as f:
            self.classnum2label = json.load(f)
    
    def __getitem__(self, index):
        #self.file
        #return a dict
        #for modelnet40: point, normal, classlabel
        #for shapenet: point, normal, classlabel, denselabel
        sixD = np.loadtxt(os.path.join(self.dataroot, self.filename[index]), delimiter=',')
        classlabel = self.classnum2label[self.filename[index].split('/')[0]]
        point = sixD[:, :3]
        normal = sixD[:, 3:6]
        return {'point':point, 'normal':normal, 'classlabel':classlabel}
    def __len__(self):
        #return 100
        return len(self.filename)

class _ModelNet40UnseenCore(_ModelNet40Core):
    def __init__(self, dataroot, pattern):
        self.dataroot = dataroot
        splitfile = os.path.join(dataroot, f'modelnet40_unseen_{pattern}.json')
        classnum2labelfile = os.path.join(dataroot, 'modelnet40_classnum2label.json')
        # read split file
        with open(splitfile, 'r') as f:
            load_dict = json.load(f)
        # filename add to self.file
        self.filename = load_dict
        with open(classnum2labelfile, 'r') as f:
            self.classnum2label = json.load(f)