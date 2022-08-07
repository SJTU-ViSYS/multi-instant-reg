'''
Author: your name
Date: 2021-03-30 11:50:27
LastEditTime: 2022-08-07 12:56:44
LastEditors: Gilgamesh666666 fengyujianchengcom@163.com
Description: In User Settings Edit
FilePath: /mm/datasets/multi_3dcorr_oneTomore.py
'''
'''
Author: bai ze
Date: 2021-03-03 21:37:07
LastEditTime: 2021-03-30 11:50:12
LastEditors: Please set LastEditors
Description: ShapeNetCore Dataset
FilePath: /undefined/home/zebai/dataset.py
'''
import sys
import os

from torch._C import dtype
sys.path.append(os.getcwd())
import open3d as o3d
from o3d_tools.visualize_tools import *
from o3d_tools.operator_tools import *
import torch
from torch.utils.data import DataLoader,Dataset
import json
import numpy as np
import os
import torchvision
import copy
class RandomJitter:
    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip
    def __call__(self, input_tensor):
        # [n,m]
        noise = np.clip(np.random.normal(loc=0.0,scale=self.scale,size=input_tensor.shape), a_min=-self.clip, a_max=self.clip)
        return input_tensor + noise

class RandomSampler:
    def __init__(self, num, axis=0):
        self.num = num
        self.axis = axis
    '''
    description: 
    param {*} input_tensor
    return {*} if axis=0
                    return (num,:)tensor
               else:
                    return (num,)idx
    '''    
    def __call__(self, input_tensor):
        assert self.axis < len(input_tensor.shape)
        N = input_tensor.shape[self.axis]
        idx = np.random.choice(N, self.num, replace=(N < self.num))
        if self.axis == 0:
            return input_tensor[idx], idx
        else:
            return idx

class _MultiRot_oneTomore_multi(Dataset):
    def __init__(self, dataset, max_amp=3, max_obj=20, size_scale=2, transform=None, random_obj=False):
        super().__init__()
        # initialize a dataset under pattern
        if (not isinstance(max_obj,int)) or (not max_obj):
            print('max_obj must be int which larger than 0!')
            raise ValueError
        else:
            self.max_obj = max_obj
        self.dataset = dataset
        self.max_amp = max_amp
        self.transform = transform
        self.random_obj = random_obj
        self.size_scale = size_scale
        
    def __getitem__(self, index):
        #random select k
        np.random.seed(index)

        
        #random select k pointcloud(no replace)
        if self.max_obj > 10:
            max_num_cls = np.random.randint(5, int(self.max_obj/2))
            # #print(max_num_cls, int(self.max_obj/2))
            num_cls = np.random.randint(1, max_num_cls)
        else:
            num_cls = 2
        choice = np.random.choice(np.arange(num_cls, dtype=np.int16), self.max_obj, replace=True)
        #print(num_cls, choice)
        _, count = np.unique(choice, return_counts=True)
        
        final_num_cls = len(count)
        total_cls_num = len(self.dataset)
        cls_indices = np.random.choice(np.arange(total_cls_num, dtype=np.int16), final_num_cls, replace=False)
        
        src_cls_idx = np.random.choice(cls_indices, 1, replace=False)[0]
        # source
        source_datapack = self.dataset[src_cls_idx]
        source_pt = source_datapack['point']#*self.size_scale#[n,3]
        raw_source_pt = copy.deepcopy(source_pt)

        raw_source_pt_idx = np.arange(source_pt.shape[0])
        if self.transform is not None:   
            #print(f'_MultiRot begin source transform')
            for tf in self.transform:
                if isinstance(tf, RandomSampler):
                    source_pt, raw_source_pt_idx = tf(source_pt)
                else:
                    source_pt = tf(source_pt)
        
        # target:
        # for _ in range(100):
        #print('_MultiRot begin target')
        datapack = {}
        #datapack['source_point'] = source_pt
        datapack['target_point'] = []
        datapack['raw_target_point'] = []
        datapack['raw_target_point_idx'] = []
        datapack['relative_target_trans'] = []
        datapack['labels'] = []
        datapack['raw_target_point_label'] = []
        c = 0
        point_list_for_check = []
        #print(count)
        cccc = 0
        for cls_idx, k in zip(cls_indices, count):
            #print(cls_idx, k)
            tgt_datapack = self.dataset[cls_idx]
            points = tgt_datapack['point']*self.size_scale#[n,3]
            for _ in range(k):
                
                tgt_trans, target_pt = random_rotation(points, max_degree=360, max_amp=self.max_amp, minamp=self.max_amp*0.25)#0.3

                raw_target_pt = target_pt.shape[0]
                
                datapack['raw_target_point'].append(target_pt)
                
                if self.transform is not None:
                    for tf in self.transform:
                        if isinstance(tf, RandomSampler):
                            target_pt, raw_target_pt_idx = tf(target_pt)
                            
                            #print(raw_target_pt_idx.shape)

                            #print(raw_target_pt_idx.shape[0]*cccc)
                            datapack['raw_target_point_idx'].append(raw_target_pt_idx + raw_target_pt*cccc)
                            
                        else:
                            target_pt = tf(target_pt)
                    
                    #target_nm = get_normals(target_pt)
                datapack['target_point'].append(target_pt)
                cccc += 1

                if cls_idx == src_cls_idx:
                    c += 1
                    datapack['relative_target_trans'].append(tgt_trans)
                    datapack['labels'].append(np.ones(target_pt.shape[0], dtype=np.int16)*c)
                    datapack['raw_target_point_label'].append(np.ones(raw_target_pt, dtype=np.int16)*c)
                else:
                    datapack['labels'].append(np.zeros(target_pt.shape[0], dtype=np.int16))
                    datapack['raw_target_point_label'].append(np.zeros(raw_target_pt, dtype=np.int16)*c)
                #datapack['target_normal'] = target_nm
                point_list_for_check.append(target_pt)
            # if not self.check_overlap(point_list_for_check):
            #     break
        #noise = np.random.rand(1000,3)*2*self.max_amp - self.max_amp
        noise = np.random.rand(1000,3)*self.max_amp
        datapack['target_point'].append(noise)
        datapack['labels'].append(np.zeros(1000, dtype=np.int16))
        datapack['raw_target_point_idx'].append(np.arange(1000, dtype=np.int16) + raw_target_pt*cccc)
        datapack['raw_target_point'].append(noise)
        datapack['raw_target_point_label'].append(np.zeros(1000, dtype=np.int16))


        return {
            'source_point':source_pt,
            #'source_normal':np.concatenate(source_normals_list),
            'target_point':np.concatenate(datapack['target_point']),
            #'target_normal':np.concatenate(target_normals_list),
            'gt_trans':np.stack(datapack['relative_target_trans']),
            'gt_labels':np.concatenate(datapack['labels']),
            'raw_source_pt':raw_source_pt,
            'raw_target_point':np.concatenate(datapack['raw_target_point']),
            'raw_source_pt_idx':raw_source_pt_idx,
            'raw_target_point_idx':np.concatenate(datapack['raw_target_point_idx']),
            'raw_target_point_labels':np.concatenate(datapack['raw_target_point_label']),
        }
        
    '''
    description: check if k point cloud are overlapping with each other
    param {source_points_list = [n1*3, n2*3..]} list
    return {bool}
    '''    
    def check_overlap(self, pointcloud_list):
        #print(f'_MultiRot begin check_overlap')
        def square_min_dist(pc1, pc2):
            # [n1,3], [n2,3]
            return np.min(square_dist(pc1, pc2))
        for i in range(len(pointcloud_list)):
            for j in range(i+1,len(pointcloud_list),1):
                if square_min_dist(pointcloud_list[i], pointcloud_list[j]) < 1e-3:
                    
                    return True
        return False
    def __len__(self):
        return int(len(self.dataset)/4)

        
def visual(source_point, target_point, corrs, labels):
    corr_list = []
    labels += 1
    for i in range(labels.max()+1):
        mask = (labels == i)
        print(f'label {i} ratio: {mask.mean()}')
        if i == 0:
            color = [1,0,0]
        else:
            color = COLOR_MAP_NROM[(i*3)%26]
        corr = visualize_correspondences_official(source_point, target_point, corrs[mask], color)
        corr_list.append(corr)
    source_pcd = make_o3d_PointCloud(source_point)
    target_pcd = make_o3d_PointCloud(target_point)
    source_pcd.paint_uniform_color(get_blue())
    target_pcd.paint_uniform_color(get_green())
    o3d.visualization.draw_geometries([source_pcd, *corr_list, target_pcd])#