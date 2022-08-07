'''
Date: 2021-06-23 18:15:30
LastEditors: Gilgamesh666666 fengyujianchengcom@163.com
LastEditTime: 2022-08-06 15:30:33
FilePath: /new_OverlapPredator_kl/datasets/multiRot.py
'''

from datasets.modelnet_shapenet_Core import _ModelNet40Core, _ShapeNetCore

import os,sys,glob,torch
import numpy as np

from torch.utils.data import Dataset
import open3d as o3d
import o3d_tools.visualize_tools as vt

from datasets.multi_3dcorr_oneTomore_multi import _MultiRot_oneTomore_multi, RandomJitter, RandomSampler

def ABfind_nndist(A,B):
    from sklearn.neighbors import KDTree
    # [n,3], [m,3]
    tree = KDTree(B)
    nndist, idx = tree.query(A, return_distance=True)
    return nndist, idx #[n,], [n,]

mn40_dataroot = '/media/zebai/T7/Datasets/modelnet40_normal_resampled'
sn_dataroot = '/media/zebai/T7/Datasets/shape_data'

class MultiRotDataset_multi(Dataset):
    def __init__(self, config, inlier_thresh=0.25, corr_num=256, split='train', with_normal=False):
        self.split = split
        self.config = config
        
        jitter_scale, jitter_clip = 0.01, 0.05
        #transform = torchvision.transforms.Compose([RandomSampler(self.config.num_point_per_model), RandomJitter(scale=jitter_scale, clip=jitter_clip)])
        transform = [RandomSampler(self.config.num_point_per_model), RandomJitter(scale=jitter_scale, clip=jitter_clip)]
        mn40_dataroot = '/media/zebai/T7/Datasets/modelnet40_normal_resampled'
        sn_dataroot = '/media/zebai/T7/Datasets/shape_data'
        modelcore = _ModelNet40Core
        modelcore_root = mn40_dataroot
        # modelcore = _ShapeNetCore
        # modelcore_root = sn_dataroot
        self.dataset = _MultiRot_oneTomore_multi(modelcore(dataroot=modelcore_root, pattern=split), max_amp=10, size_scale=1, max_obj=self.config.max_obj, transform=transform, random_obj=self.config.random_obj)

        self.inlier_thresh = 1.5*jitter_clip
        
        self.corr_num = corr_num
        self.cache = {}
    def __getitem__(self, index):
        #pcs[view_sel, ...].astype(np.float32), segms[view_sel, ...].astype(np.int16), new_trans_dict, all_part_trans
        #[view,n,3/6], [view,n], dict{'cam':[view, 4, 4], 'part(obj)_id':[view, 4, 4]}, dict{'i_j':{'part(obj)_id':[4,4]}}
        #pcs, segms, new_trans_dict, all_part_trans = self.dataset[index]
        datapack = self.dataset[index]
        src = datapack['source_point'] #numpy [obj_num*n, 3]
        tgt = datapack['target_point'] #numpy [obj_num*n, 3]

        raw_source_pt = datapack['raw_source_pt'].astype(np.float32)
        raw_target_point = datapack['raw_target_point'].astype(np.float32)
        raw_source_pt_idx = datapack['raw_source_pt_idx'].astype(np.float32)
        raw_target_point_idx = datapack['raw_target_point_idx']
            
        src_pcd,tgt_pcd = src, tgt
        #raw_src = np.concatenate((src, get_normals(src, radius=0.1)), axis=1)
        #raw_tgt = np.concatenate((tgt, get_normals(tgt, radius=0.1)), axis=1)
        
        part_trans = datapack['gt_trans'] #[obj_num, 4, 4]
        
        src_n = src_pcd.shape[0]
        n_parts = len(part_trans)
        
        labels = datapack['gt_labels']

        # tgt = np.concatenate((tgt, tgt.min(axis=0) + np.random.rand(1000,3)*(tgt.max(axis=0)-tgt.min(axis=0))), axis=0)
        # labels = np.concatenate((labels, np.zeros(1000, dtype=np.int16)), axis=0)
        #target_normal = datapack['target_normal']
        
        ##################### labels and inlier_mask ####################
        # labels = {1, 2, 6} #...
        # slice the source_point by k point_num_list
        gt_trans = datapack['gt_trans'] # numpy

        trans_src = []
        for trans in gt_trans:
            R = trans[:3, :3]
            t = trans[:3, 3]
            trans_src.append(np.dot(src, R.T) + t.T)
        trans_src = np.concatenate(trans_src)

        #* Generate labels -----------------------------------------------------------------------------
        if self.split != 'train':
            
            tgt_pcd_with_labels = np.concatenate((tgt_pcd,labels[:, None]), axis=1)
            
            #vs_ls = []
            #print(tgt_pcd_labels.max(),part_trans.shape[0])
            # for i in range(tgt_label.max()+1):
            #     if (tgt_label==i).sum():
            #         print(i)
            #         vs_ls.append(vt.make_o3d_PointCloud(tgt_pcd[tgt_label==i, :3], color=vt.COLOR_MAP_NROM[i*5]))
            # o3d.visualization.draw_geometries(vs_ls)
        else:
            tgt_pcd_with_labels = tgt_pcd
        #*----------------------------------------------------------------------------------------------
        
        #trans_src = np.dot(trans_src, fix_T[:3,:3].T) + fix_T[:3,3]
        #manual_registration(vt.make_o3d_PointCloud(trans_src, color=vt.SRC_COLOR), o3dpcd)
        
        #o3d.visualization.draw_geometries([vt.make_o3d_PointCloud(trans_src, color=vt.SRC_COLOR), o3dpcd])
        #registration(trans_src, tgt)
        src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

        if index in self.cache.keys():
            nndist, idx = self.cache[index]
        else:
            nndist, idx = ABfind_nndist(trans_src[:, :3], tgt_pcd)
            self.cache[index] = (nndist, idx)
        #print(idx.shape)
        corr = np.concatenate((np.arange(idx.shape[0])[:, None], idx), axis=1)
        correspondences = corr[np.squeeze(nndist) < self.inlier_thresh] # [m, 2]
        if self.split=='train':
            idx = np.random.choice(corr.shape[0], self.corr_num, replace=(corr.shape[0]<self.corr_num))
            corr = corr[idx]
        
        corr_idx = np.arange(correspondences.shape[0])
        np.random.shuffle(corr_idx)
        
        correspondences = torch.from_numpy(correspondences[corr_idx])


        if self.split == 'test':
            rot = part_trans.astype(np.float32)
        else:
            rot = part_trans[:, :3, :3].astype(np.float32)
        trans = part_trans[:, :3, 3].astype(np.float32)

        src_bbox = vt.make_o3d_PointCloud(src_pcd).get_oriented_bounding_box()
        R, center, extent_ori = src_bbox.R, src_bbox.center, src_bbox.extent
        rr = np.eye(4)
        rr[:3, :3] = R
        extent = [i/2 for i in extent_ori]
        idx = np.argmin(extent)
        extent[idx] = max(0.1, extent[idx])
        src_pcd_box = np.asarray([[1, 0, 0, center[0]], [0, 1, 0, center[1]], [0, 0, 1, center[2]], [0, 0, 0, 1]]).dot(rr).dot(np.asarray([[extent[0], 0, 0, 0], [0, extent[1], 0, 0], [0, 0, extent[2], 0], [0, 0, 0, 1]]))

        bbox = []
        for ttt in part_trans:
            bbox.append(ttt.dot(src_pcd_box))
           
        raw_source_pt = datapack['raw_source_pt'].astype(np.float32)
        raw_target_pt = datapack['raw_target_point'].astype(np.float32)
        raw_source_pt_idx = datapack['raw_source_pt_idx'].astype(np.float32)
        raw_target_pt_idx = datapack['raw_target_point_idx']
        raw_target_pt_label = datapack['raw_target_point_labels']
        return src_pcd,tgt_pcd,src_feats,tgt_feats,rot,trans, correspondences, trans_src, tgt_pcd_with_labels, torch.ones(1), (bbox, src_pcd_box, raw_source_pt, raw_target_pt, raw_source_pt_idx, raw_target_pt_idx, raw_target_pt_label)

    def __len__(self):
        return len(self.dataset)