'''
Date: 2021-07-17 10:35:19
LastEditors: Gilgamesh666666 fengyujianchengcom@163.com
LastEditTime: 2022-08-07 11:32:33
FilePath: /new_OverlapPredator_kl/datasets/scan2cadDatasets_box.py
'''
#import PCLKeypoint
from enum import Flag
import enum
from genericpath import exists
import os
from pickle import FALSE
from numpy.core.getlimits import _register_type
from numpy.random.mtrand import shuffle
import open3d as o3d
from torch._C import Value
import JSONHelper
import o3d_tools.visualize_tools as vt
import o3d_tools.operator_tools as ot
import numpy as np
import pymeshlab
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
import shutil

def render(pcd):
    # [n,3]
    from scipy.spatial.transform import Rotation as Rotation
    view_num = 8
    W = ((2*np.pi - np.arange(0, 2*np.pi, 2*np.pi/view_num))[:, None, None]*np.eye(3)[None, :, :]).reshape(-1, 3) #[view_num*3, 3]
    
    Rs = Rotation.from_rotvec(W).as_matrix() #[view_num*3, 3]->#[view_num*3, 3, 3]
    #print(Rs)
    pcd_center = pcd - np.mean(pcd, axis=0) # [n,3]
    radius = np.max(np.linalg.norm(pcd_center, axis=1))
    
    new_pcd = np.einsum("bik,bkj->bij", pcd_center[None, :, :], Rs.transpose(0, 2, 1))#[view_num*3, n, 3]
    grid_num = 20
    voxel_size = 2*radius/grid_num
    #print(voxel_size)
    new_xy = np.floor(new_pcd[..., :2]/voxel_size)
    hashkey = new_xy[..., 0] + new_xy[..., 1]*grid_num #[view_num*3, n]
    #print(hashkey)
    hashval = new_pcd[..., 2]#[view_num*3, n]
    indices = np.argsort(hashkey, axis=1)#[view_num*3, n]
    
    #[view_num*3, n]
    proj_pcd = []
    for key, val, pc, idx, R in zip(hashkey, hashval, new_pcd, indices, Rs):
        key, val, pc = key[idx], val[idx], pc[idx]
        #vt.visualize_pcd(pc, visual=True)
        _, count = np.unique(key, return_counts=True) #[n]->[m]
        #print(count.shape)
        start = 0
        points = []
        for i in count:
            end = start + i
            #print(start, end)
            z_idx = np.argmin(val[start:end])
            points += [pc[z_idx+start,:]]
            start = end
        proj_pcd.append(np.dot(np.asarray(points), R))
        #print(np.asarray(points))
        #vt.visualize_pcd(np.asarray(points), visual=True)
    proj_pcd = np.concatenate(proj_pcd,axis=0)
    return proj_pcd

def resample_mesh_meshlab(objfile, pts_num):
    # lines needed to run this specific example
    
    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # load mesh
    ms.load_new_mesh(objfile)
    
    # apply convex hull filter to the current selected mesh (last loaded)
    ms.montecarlo_sampling(samplenum=pts_num)

    # get a reference to the current selected mesh
    m = ms.current_mesh()
    points = m.vertex_matrix()
    return points.astype(np.float32)

def read_ply_meshlab(objfile):
    # lines needed to run this specific example
    
    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # load mesh
    ms.load_new_mesh(objfile)
    
    # get a reference to the current selected mesh
    m = ms.current_mesh()
    #m.show()
    points = m.vertex_matrix()
    return points.astype(np.float32)

def read_cad(shapenet_dir, catid_cad, id_cad, pts_num=10000, ds_vs=None, return_normals=False):
    #shapenet_dir = '/home/zebai/Scan2CAD/Assets/shapenet-sample'
    cad_file = os.path.join(shapenet_dir, catid_cad, id_cad, "models/model_normalized.obj")

    savefile = cad_file.replace('.obj', f'_{pts_num}.npy')
    if os.path.exists(savefile) and np.load(savefile).shape[0] == pts_num:
        sample_points = np.load(savefile)
        #print(f'read {savefile}')
    else:
        sample_points = resample_mesh_meshlab(cad_file, pts_num=pts_num)
        np.save(savefile, sample_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample_points)
    # vt.visualize_pcd(sample_points, visual=True)
    # vt.visualize_pcd(render(sample_points), visual=True)

    if ds_vs !=None:
        pcd = pcd.voxel_down_sample(ds_vs)
    
    if return_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.1))
        pcd.orient_normals_towards_camera_location()
        pcd.normalize_normals()
        o3d.visualization.draw_geometries([pcd])
        return np.asarray(pcd.points,dtype=np.float32), np.asarray(pcd.normals,dtype=np.float32)
    else:
        return np.asarray(pcd.points,dtype=np.float32)
    
def read_scan(scannet_dir, id_scan, ds_vs=None, return_normals=False, with_color=False):
    if int(id_scan.split('_')[0][-4:]) > 706:
        scannet_dir = f'{scannet_dir}/scans_test'
    else:
        scannet_dir = f'{scannet_dir}/scans'

    plyfile = os.path.join(scannet_dir, id_scan, f'{id_scan}_vh_clean_2.labels.ply')#_vh_clean_2.ply
    if with_color:
        pcd = o3d.io.read_point_cloud(plyfile)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(read_ply_meshlab(plyfile))
    if ds_vs !=None:
        pcd = pcd.voxel_down_sample(ds_vs)
    
    # if return_labels:
    #     with open(plyfile, 'rb') as f:
    #         plydata = PlyData.read(f)
    #         #print(plydata.elements)
    #         labels = plydata['vertex'].data['label']
    if with_color:
        pc = np.concatenate((np.asarray(pcd.points,dtype=np.float32), np.asarray(pcd.colors,dtype=np.float32)), axis=1)
    else:
        pc = np.asarray(pcd.points,dtype=np.float32)
    if return_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.1))
        pcd.orient_normals_towards_camera_location()
        pcd.normalize_normals()
        o3d.visualization.draw_geometries([pcd])
        return pc, np.asarray(pcd.normals,dtype=np.float32)
    else:
        return pc
    
def make_M_from_tqs(t, q, s):
    import quaternion
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M 

def from_trs_to_M(trs, with_scale=True):
    if with_scale:
        return make_M_from_tqs(trs['translation'], trs['rotation'], trs['scale'])
    else:
        return make_M_from_tqs(trs['translation'], trs['rotation'], np.array([1,1,1]))

def ABfind_nndist(A,B):
    from sklearn.neighbors import KDTree
    # [n,3], [m,3]
    tree = KDTree(B)
    nndist, idx = tree.query(A, return_distance=True)
    return nndist, idx #[n,], [n,]

# model_gen_fun
def rigid_tranform_from_points_with_scale(data, choice, iter_num=3, device='cuda'):
    # data:[n,c]
    # choice:[min_sample]
    
    P_p = data[choice, 0:3]
    P = data[choice, 3:6]
    
    #print(P_p.shape, P.shape)
    P_p_mean = torch.mean(P_p, dim=0, keepdim=True) # 1,3
    P_mean = torch.mean(P, dim=0, keepdim=True) # 1,3
    Q_p = P_p - P_p_mean # n_matches,3
    Q = P - P_mean # n_matches,3

    
    s = torch.eye(3, device=device)
    for _ in range(iter_num):
        A = torch.matmul(Q.transpose(-1,-2), Q_p)# 3,3
        #B = A.cpu()
        u, _, v = torch.svd(s.matmul(A)) # 3,3
        R = torch.matmul(u, v.transpose(-1,-2)) # 3， 3
        s = (torch.sum(torch.diag_embed(torch.matmul(Q, R))*Q_p.unsqueeze(1), dim=0).diagonal(0)/torch.sum(torch.diag_embed(Q_p)*Q_p.unsqueeze(1), dim=0).diagonal(0)).diag()
        #print(s, R)
    t = (P_mean.transpose(-1,-2) - torch.matmul(R.matmul(s), P_p_mean.transpose(-1,-2))).squeeze(-1) # 3, 3 * 3, 1 -> 3, 1 -> 3
    #print(R, t)
    I = torch.eye(4, device=device).view(4, 4)
    I[:3, :3] = R.matmul(s)
    I[:3, 3] = t
    h = I.view(-1)# 16
    return h


def check_part_trans(part_trans):
    #torch [m, 4, 4]
    m = part_trans.shape[0]
    R = part_trans[:, :3, :3] #[m, 3, 3]
    t = part_trans[:, :3, 3] #[m, 3]

    re = torch.acos(torch.clamp((torch.einsum('bkii->bk', torch.matmul(R.permute(0, 2, 1).unsqueeze(1), R.unsqueeze(0))) - 1) / 2.0, min=-1, max=1)) # [m,m]
    te = torch.norm((t.unsqueeze(1) - t.unsqueeze(0)), dim=2) # [m,m]
    print(re, te)

def get_correspondence(src_pcd, tgt_pcd, part_trans, inlier_thresh, num_node=0):
    
     
    trans_src = ot.apply_transform_2dim_numpy(src_pcd, part_trans)
        
    nndist, idx = ABfind_nndist(trans_src[:, :3], tgt_pcd)
    
    #print(idx.shape)
    corr = np.concatenate((np.arange(idx.shape[0])[:, None], idx), axis=1)
    mask = np.squeeze(nndist) < inlier_thresh
    correspondences = corr[mask] # [m, 2]

    
    #print(correspondences)
    correspondences[:,0] = correspondences[:,0]%src_pcd.shape[0]
    
    # get correspondence at fine level
    #print(correspondences.shape)
    if num_node and len(correspondences)>num_node:
        idx = np.random.choice(len(correspondences), num_node, replace=False)
    else:
        idx = np.arange(correspondences.shape[0])
        np.random.shuffle(idx)
    #print(correspondences.shape, corr_labels.shape)
    correspondences = correspondences[idx]

    return correspondences

def splitscan(scan_pcd, grid=3):
    # [n,3]
    scan_xy = scan_pcd[:, :2]#np.stack((scan_pcd[:, 0], scan_pcd[:, 2]), axis=1) #[n,2]
    max_xy = np.max(scan_xy, axis=0) #[2,]
    min_xy = np.min(scan_xy, axis=0) #[2,]
    vox_sz = (max_xy-min_xy)/grid #[2,]
    vox = np.floor((scan_xy-min_xy)/vox_sz) #[n,2]
    vox_num = vox[:,0]*grid + vox[:,1] #[n,]
    unique = np.unique(vox_num)
    pcd_ls = []
    for i in unique:
        pcd_ls.append(scan_pcd[vox_num == i])
    return pcd_ls

def check_dulp(max_p, min_p, part_trans):
    # [m,3],[m,3], [m,4,4]
    
    m = max_p.shape[0]
    mask = np.ones((m, m), dtype=bool)
    for i in range(m):
        for j in range(i+1, m):
            max_norm = np.linalg.norm(max_p[i] - max_p[j])
            min_norm = np.linalg.norm(min_p[i] - min_p[j])
            #print(max_norm, min_norm)
            if max_norm<0.2 and min_norm <0.2:
                mask[i, j] = False
    #print(mask)
    mask = mask.min(axis=1)
    #print(mask)
    return max_p[mask], min_p[mask], part_trans[mask]

def max_rectange(part_rectangle, max_ratio):
    # [m, 8, 3], float
    max_corner = part_rectangle.max(axis=1)#[m,3]
    min_corner = part_rectangle.min(axis=1)#[m,3]
    lwh = (max_corner- min_corner)/2 #[m,3]
    center = min_corner + lwh #[m,3]
    lwh = max_ratio*lwh #[m,3]
    box = np.asarray([[-1, -1, -1],
                        [-1, 1, 1],
                        [1, -1, 1],
                        [1, 1, -1],
                        [-1, -1, 1],
                        [-1, 1, -1],
                        [1, -1, -1],
                        [1, 1, 1]]) # [8, 3]
    return center[:, None, :] + lwh[:, None, :]*box[None, :, :] # [m,1,3]*[1,8,3]->[m,8,3]

def scaleTo4(scale):
    T = np.eye(4)
    T[:3,:3] = np.diag(scale)
    return T

class Scan2CAD_oneTomore_box(Dataset):
    def __init__(self, config, split, shapenetroot, scannetroot, scan2cadroot, num_corr, ds_vs=None, with_scale=True, points_thresh=30000) -> None:
        super().__init__()
        self.shapenetroot = shapenetroot
        self.scannetroot = scannetroot
        self.scan2cadroot = scan2cadroot
        self.ds_vs = ds_vs
        self.with_scale = with_scale
        self.num_node = num_corr
        self.split = split
        self.config = config
        self.cache = {}
        self.inlier_thresh = ds_vs*1.25
        split_id_scan = JSONHelper.read(f"{scan2cadroot}/split.json")[split]
        self.full_annot = JSONHelper.read(f"{scan2cadroot}/full_annotations.json")
        
        self.valid_scan_file = f'id_scan_{self.ds_vs}_{points_thresh}_train.json'
        if os.path.exists(self.valid_scan_file):
            self.valid_scan = JSONHelper.read(self.valid_scan_file)
        else:
            self.valid_scan = []
        self.valid_scan_to_save = []
        self.files = []
        unvalid = 0
        for annot in self.full_annot:
            id_scan = annot['id_scan']
            if id_scan not in split_id_scan:
                continue
            
            if self.split=='train':
                if len(self.valid_scan):
                    if id_scan not in self.valid_scan:
                        continue
                else:
                    scan_pcd = read_scan(self.scannetroot, id_scan, ds_vs=self.ds_vs)
                    #print(scan_pcd.shape[0])
                    if scan_pcd.shape[0]<points_thresh:
                        self.valid_scan_to_save.append(id_scan)
                    else:
                        continue
            scan2world = from_trs_to_M(annot['trs'], self.with_scale)
            
            model_set = set()
            replicate_model = []
            for item in annot['aligned_models']:
                
                catid_cad = item['catid_cad']
                id_cad = item['id_cad']
                key = f'{catid_cad}_{id_cad}'

                if key in model_set:
                    if key not in replicate_model:
                        replicate_model.append(key)
                else:
                    model_set.add(key)
            
            gt_datas = {}
            for key in replicate_model:
                gt_datas[key] = {'gt_trans':[], 'scale':[], 'rectangle':[], 'bbox':[], 'lwh':[]}
            
            
            for item in annot['aligned_models']:
                catid_cad = item['catid_cad']
                id_cad = item['id_cad']
                
                
                key = f'{catid_cad}_{id_cad}'
                
                if key in gt_datas.keys():
                    cad2scan = (np.linalg.inv(scan2world)).dot(from_trs_to_M(item['trs'], self.with_scale)).astype(np.float32)
                    gt_datas[key]['scan2world'] = scan2world

                    scale = item['trs']['scale']
                    lwh = item['bbox']
                    center = item['center']
                    gt_datas[key]['gt_trans'].append(cad2scan)
                    gt_datas[key]['scale'].append(scale)
                    
                    l, w, h = np.asarray(lwh)
                    cx, cy, cz = center

                    rectangle = np.array([[cx - l, cy - w, cz - h],
                                [cx + l, cy + w, cz - h],
                                [cx + l, cy - w, cz - h],
                                [cx - l, cy + w, cz - h],
                                [cx - l, cy - w, cz + h],
                                [cx + l, cy + w, cz + h],
                                [cx + l, cy - w, cz + h],
                                [cx - l, cy + w, cz + h]])
                    if not self.with_scale: 
                        rectangle = rectangle.dot(np.diag(scale).astype(np.float32))   
                    rectangle = ot.apply_transform_2dim_numpy(rectangle, cad2scan)
                    
                    tt = np.asarray([[l, 0, 0, cx], [0, w, 0, cy], [0, 0, h, cz], [0, 0, 0, 1]])
                    bbox = tt
                    if not self.with_scale:
                        bbox = np.asarray([[scale[0], 0, 0, 0], [0, scale[1], 0, 0], [0, 0, scale[2], 0], [0, 0, 0, 1]]).dot(bbox)
                    bbox = cad2scan.dot(bbox)

                    
                    gt_datas[key]['bbox'].append(bbox)
                    gt_datas[key]['lwh'].append(np.asarray(lwh))

                    gt_datas[key]['rectangle'].append(rectangle)
                    
                    #gt_trans is cad2scan
                    
            for key, value in gt_datas.items():
                catid_cad, id_cad = key.split('_')
                part_trans = np.stack(value['gt_trans'], axis=0)
                part_scale = np.stack(value['scale'], axis=0)#.mean(axis=0)
                part_rectangle = np.stack(value['rectangle'], axis=0)# [m,8,3]
                lwh = np.stack(value['lwh'], axis=0).mean(axis=0)
                
                if part_scale.min()<0.1:
                    
                    unvalid += 1
                self.files.append((id_scan, catid_cad, id_cad, part_trans, part_scale, part_rectangle, value['bbox'], lwh, gt_datas[key]['scan2world']))
                
        if len(self.valid_scan_to_save):
            JSONHelper.write(self.valid_scan_file, self.valid_scan_to_save)

    def box(self, scan_pcd, part_rectangle, part_trans, check_dulplicate=True, max_ratio=1):
        #print(part_rectangle)
        if max_ratio!=1:
            part_rectangle = max_rectange(part_rectangle, max_ratio)
        
        min_p = part_rectangle.min(axis=1) # [m, 3]
        max_p = part_rectangle.max(axis=1) # [m, 3]
        #print(part_rectangle)
        if check_dulplicate:
            max_p, min_p, part_trans = check_dulp(max_p, min_p, part_trans)
        
        mask1 = (scan_pcd[:, None, :3] - min_p[None, :, :])>0 # [n, m, 3]
        mask2 = (max_p[None, :, :] - scan_pcd[:, None, :3])>0 # [n, m, 3]
        mask = (mask1 * mask2) # [n, m, 3]
        #print(mask)
        mask = (mask[..., 0]*mask[..., 1]*mask[..., 2])
        belong = mask.argmax(axis=1) # [n, m]->[n]
        mask = mask.max(axis=1) # [n, m]->[n]
        new_scan_pcd = scan_pcd[mask] # [n,3]->[n1,3]
        belong = belong[mask] # [n]->[n1]
        return new_scan_pcd, belong, part_trans, part_rectangle
        
    
    def __getitem__(self, index):
        #print(f'index = {index}')
        id_scan, catid_cad, id_cad, part_trans, part_scale, part_rectangle, bbox, lwh, scan2world = self.files[index]
        
        scan_pcd = read_scan(self.scannetroot, id_scan, ds_vs=self.ds_vs, return_normals=False, with_color=(self.config.in_feats_dim!=1))
        cad_pcd = read_cad(self.shapenetroot, catid_cad, id_cad, pts_num=10000, ds_vs=self.ds_vs, return_normals=False)


        if self.with_scale:
            iter_num = 3
            tgt_pcd_box = np.eye(4).astype(np.float32)
        else:
            cad_pcd = cad_pcd.dot(np.diag(part_scale.mean(axis=0)).astype(np.float32))

            sss = part_scale.mean(axis=0)
            #tgt_pcd_box = np.eye(4).astype(np.float32)
            tgt_pcd_box = np.asarray([[lwh[0]*sss[0], 0, 0, 0], [0, lwh[1]*sss[1], 0, 0], [0, 0, lwh[2]*sss[2], 0], [0, 0, 0, 1]])


        if len(part_trans.shape)==2 and len(part_rectangle.shape)==2:
            part_rectangle, part_trans = part_rectangle[None, :, :], part_trans[None, :, :]
        raw_part_rectangle = part_rectangle
        new_scan_pcd, belong, part_trans, part_rectangle = self.box(scan_pcd, part_rectangle, part_trans, check_dulplicate=True, max_ratio=1.25)

        idx = np.argsort(belong)
        new_scan_pcd = new_scan_pcd[idx] # [n,3]->[n1,3]
        belong = belong[idx] # [n]->[n1]
        uni, count = np.unique(belong, return_counts=True)
        #
        if part_trans.shape[0] != count.shape[0]:
            
            o3d.visualization.draw_geometries([vt.visualize_pcd(scan_pcd[:, :3], vt.SRC_COLOR, visual=False), vt.visualize_pcd(cad_pcd, vt.TGT_COLOR, visual=False), vt.visualize_pcd(part_rectangle.reshape(-1, 3), [1, 0, 0], visual=False)])
            for rectangle in part_rectangle:
                o3d.visualization.draw_geometries([vt.visualize_pcd(scan_pcd[:, :3], vt.SRC_COLOR, visual=False), vt.visualize_pcd(cad_pcd, vt.TGT_COLOR, visual=False), vt.visualize_pcd(rectangle, [1, 0, 0], visual=False)])
            o3d.visualization.draw_geometries([vt.visualize_pcd(new_scan_pcd, vt.SRC_COLOR, visual=False)])
        src_pcd = new_scan_pcd
        tgt_pcd = cad_pcd
        
        
        trans_src = []
        new_part_trans = []
        part_start = 0
        for p in range(part_trans.shape[0]):
            part_num = count[p]
            #print(trans)
            T = part_trans[p]
            trans_src_xyz = ot.apply_transform_2dim_numpy(src_pcd[part_start:part_start+part_num, :3], np.linalg.inv(T))
            new_part_trans.append(np.linalg.inv(T))
            
            if self.split != 'train':
                src_labels = np.ones((part_num,1))*(p + 1)
                trans_src.append(np.concatenate((trans_src_xyz, src_labels), axis=-1))
            else:
                trans_src.append(trans_src_xyz)
            part_start += part_num
        
        trans_src = np.concatenate(trans_src, axis=0)
        part_trans = np.stack(new_part_trans, axis=0)
        # get correspondence at fine level

        if self.config.in_feats_dim ==1:
            src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
            tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)
        else:
            src_feats = src_pcd[:, 3:6]
            tgt_feats = tgt_pcd[:, 3:6]
            src_pcd = src_pcd[:, 0:3]
            tgt_pcd = tgt_pcd[:, 0:3]

        
        
        nndist, idx = ABfind_nndist(trans_src[:, :3], tgt_pcd)

        #print(idx.shape)
        corr = np.concatenate((np.arange(idx.shape[0])[:, None], idx), axis=1)
        correspondences = corr[np.squeeze(nndist) < self.inlier_thresh] # [m, 2]

        corr_idx = np.arange(correspondences.shape[0])
        np.random.shuffle(corr_idx)
        
        correspondences = torch.from_numpy(correspondences[corr_idx])


        if self.split == 'test':
            rot = part_trans.astype(np.float32)
        else:
            rot = part_trans[:, :3, :3].astype(np.float32)
        trans = part_trans[:, :3, 3].astype(np.float32)


        if part_trans.shape[0] != count.shape[0]:
            print(part_trans.shape[0],count.shape[0])

        init_t = np.eye(4)
        init_t[:3,:3] = np.diag(part_scale.mean(axis=0)).astype(np.float32)
        
        return src_pcd,tgt_pcd,src_feats,tgt_feats,rot,trans, correspondences, trans_src, tgt_pcd, torch.ones(1), (bbox, tgt_pcd_box, catid_cad, id_cad, id_scan, init_t, scan2world)
    def __len__(self):
        return len(self.files)

def build_patch_input(pcd, keypts, vicinity=0.3, num_points_per_patch=2048):
    from sklearn.neighbors import KDTree
    refer_pts = keypts.astype(np.float32)#[m,3]
    pts = pcd.astype(np.float32)
    #print(refer_pts.shape)
    num_patches = refer_pts.shape[0]
    tree = KDTree(pts[:, 0:3])
    ind_local = tree.query_radius(refer_pts[:, 0:3], r=vicinity)
    #print(ind_local)
    
    if num_patches==1 and num_points_per_patch==None:
        local_neighbors = pts[ind_local[0], :]
        local_neighbors[-1, :] = refer_pts[0, :]
        local_patches = local_neighbors[None, :]
    else:
        local_patches = np.zeros([num_patches, num_points_per_patch, 3], dtype=float)
        for i in range(num_patches):
            local_neighbors = pts[ind_local[i], :]
            temp = np.random.choice(range(local_neighbors.shape[0]), num_points_per_patch, replace=(local_neighbors.shape[0] < num_points_per_patch))
            local_neighbors = local_neighbors[temp]
            local_neighbors[-1, :] = refer_pts[i, :]
            local_patches[i] = local_neighbors
    return local_patches # [n, m, 3]
def keypoint(pcd, ratio=0, num=0):
    #kpts = PCLKeypoint.keypointSift(pcd.cpu().numpy(), min_scale=0.1,n_octaves=4,n_scales_per_octave=8,min_contrast=1e-8)#, radius=0.1, nms_threshold=0.00001, is_nms=True, is_refine=False)
    if ratio:
        idx = np.random.choice(pcd.shape[0], int(pcd.shape[0]*ratio), replace=False)
    elif num:
        idx = np.random.choice(pcd.shape[0], num, replace=(pcd.shape[0]<num))
    kpts = pcd[idx]
    return kpts, idx

if __name__ == '__main__':
    shapenetroot = '/media/zebai/T7/Datasets/ShapeNetCore.v2'
    scannetroot = '/media/zebai/T7/Datasets/ScannetV2/ScanNet'
    scan2cadroot = '/home/zebai/scan2cadDataset'
    num_corr = 128
    dataset = Scan2CAD_oneTomore('train', shapenetroot, scannetroot, scan2cadroot, num_corr, ds_vs=0.006, with_scale=False, points_thresh=200000)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    re_ls = []
    te_ls = []
    for _, _, _, _ in dataloader:
        pass