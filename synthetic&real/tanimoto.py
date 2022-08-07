'''
Date: 2021-06-02 10:48:39
LastEditors: Gilgamesh666666 fengyujianchengcom@163.com
LastEditTime: 2022-08-06 16:56:38
FilePath: /new_OverlapPredator_kl/tanimoto.py
'''
import time
import open3d as o3d
from o3d_tools import operator_tools as ot
import torch
import torch.nn as nn
import numpy as np
import scipy
import numpy as np
import random
import copy
import sys
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_printoptions(
    precision=2,   
    threshold=1000,
    edgeitems=3,
    linewidth=150,  
    profile=None,
    sci_mode=False
)

def rigid_tranform_consistency_measure(h, data, device):
    # data: B x N x C
    # h: B x M x 16

    B = data.size(0)
    N = data.size(1)
    M = h.size(1)

    H = h.view(B, M, 1, 4, 4).to(device) # B, M,16 -> B, M, 1, 4, 4
    #print(torch.linalg.det(H))
    Hinv = torch.inverse(H)
    if torch.isnan(Hinv).sum():
        temp_H = H.reshape(-1, 4, 4)
        temp_Hinv = []
        for i in range(temp_H.shape[0]):
            temp_Hinv.append(torch.inverse(temp_H[i]))
        Hinv = torch.stack(temp_Hinv, dim=0).reshape(B, M, 1, 4, 4)
    x1 = data[:, :, 0:3] # b,n,3
    x2 = data[:, :, 3:6] # b,n,3
    X1 = torch.ones((B, N, 4, 1), device=device)
    X2 = torch.ones((B, N, 4, 1), device=device)
    X1[:, :, 0:3, 0] = x1 # B,1,n,4,1
    X2[:, :, 0:3, 0] = x2 # B,1,n,4,1

    HX1 = torch.matmul(H, X1) # b,m,n, 4, 1
    #print(HX1)
    HX1[:, :, :, 0, 0] /= (HX1[:, :, :, 3, 0] + 1e-8)
    HX1[:, :, :, 1, 0] /= (HX1[:, :, :, 3, 0] + 1e-8)
    HX1[:, :, :, 2, 0] /= (HX1[:, :, :, 3, 0] + 1e-8)
    HX1[:, :, :, 3, 0] /= (HX1[:, :, :, 3, 0] + 1e-8)
    HX2 = torch.matmul(Hinv, X2)# b,m,n, 4, 1
    #print(Hinv)
    #print(HX2)
    HX2[:, :, :, 0, 0] /= (HX2[:, :, :, 3, 0] + 1e-8)
    HX2[:, :, :, 1, 0] /= (HX2[:, :, :, 3, 0] + 1e-8)
    HX2[:, :, :, 2, 0] /= (HX2[:, :, :, 3, 0] + 1e-8)
    HX2[:, :, :, 3, 0] /= (HX2[:, :, :, 3, 0] + 1e-8)

    signed_distances_1 = HX1 - X2# b,m,n, 4, 1
    distances_1 = (signed_distances_1 * signed_distances_1).sum(dim=-2).squeeze(-2).squeeze(-1)
    signed_distances_2 = HX2 - X1# b,m,n, 4, 1
    distances_2 = (signed_distances_2 * signed_distances_2).sum(dim=-2).squeeze(-2).squeeze(-1)

    distances = distances_1 + distances_2
    # distances: B x N x M
    return torch.sqrt(distances.transpose(-2,-1)/2)# b,m,n->b,n,m

def rigid_tranform_from_points(data, choice, device):
    # data:[n,c]
    # choice:[min_sample]

    P_p = data[choice, 0:3]
    P = data[choice, 3:6]
    #print(P_p.shape, P.shape)
    P_p_mean = torch.mean(P_p, dim=0, keepdim=True) # 1,3
    P_mean = torch.mean(P, dim=0, keepdim=True) # 1,3
    Q_p = P_p - P_p_mean # n_matches,3
    Q = P - P_mean # n_matches,3

    A = torch.matmul(Q.transpose(-1,-2), Q_p)# 3,3
    #B = A.cpu()
    u, s, v = torch.svd(A) # 3,3
    reflect = torch.eye(3, device=device)
    R = torch.matmul(u, v.transpose(-1,-2)) # 3， 3
    reflect[2, 2] = torch.det(R)
    R = torch.matmul(u, reflect).matmul(v.transpose(-1,-2)) # 3， 3
    t = (P_mean.transpose(-1,-2) - torch.matmul(R, P_p_mean.transpose(-1,-2))).squeeze(-1) # 3, 3 * 3, 1 -> 3, 1 -> 3
    
    I = torch.eye(4, device=device).view(4, 4)
    I[:3, :3] = R
    I[:3, 3] = t
    h = I.view(-1)# 16
    if torch.isnan(h).sum():
        print(R, t)
        sys.exit(-1)
    return h
    
def rigid_tranform_from_corr_ransac(src_pcd, tgt_pcd, corr, distance_threshold, device):
    src_pcd, tgt_pcd, corr = src_pcd.cpu().numpy(), tgt_pcd.cpu().numpy(), corr.cpu().numpy()
    
    corrs = o3d.utility.Vector2iVector(corr)
    src_pcd = ot.make_o3d_PointCloud(src_pcd)
    tgt_pcd = ot.make_o3d_PointCloud(tgt_pcd)
    result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
                source=src_pcd, target=tgt_pcd,corres=corrs, 
                max_correspondence_distance=distance_threshold,
                estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                criteria=o3d.registration.RANSACConvergenceCriteria(50000, 1000))

    hs = copy.deepcopy(result_ransac.transformation)
    return torch.from_numpy(hs).reshape(-1).float().to(device)


class Tanimoto(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.with_dist_corr_compatibility = (not args.wo_dist_corr_compatibility)
        self.with_refine = (not args.wo_refine)
        self.with_preference = (not args.wo_preference)
        self.dist_for_preference = args.dist_for_preference
        self.refine_iter_num = args.refine_iter_num
        self.initial_inlier_mask_thresh = args.initial_inlier_mask_thresh #0, -1 = without initial outlier removal

        self.min_dist_thresh = args.min_dist_thresh #0.05
        self.cpat_thresh = args.cpat_thresh #0.8

        self.sigma_spat = args.sigma_spat#1.4
        self.dist_sigma = args.dist_sigma#3
        
        self.keypts_num = args.keypts_num#1000
        self.min_num4solve_model = args.min_num4solve_model#3
        self.inlier_dist_thresh = args.inlier_dist_thresh#0.0625
        self.iou_mask_thresh = args.iou_mask_thresh#0.8
        self.num_thresh_multipler = args.num_thresh_multipler
        self.N_multipler = args.N_multipler#100

        self.device = args.device

    def forward(self, corr_pos):
        """
        Input:
            - corr:   [bs, num_corr, 6]
            - src_keypts: [bs, n, 3]
            - tgt_keypts: [bs, n, 3]
            - testing:    flag for test phase, if False will not calculate M and post-refinement.
        """
        # corr_pos = data['corr_pos']
        corr_pos = corr_pos.to(self.device)

        src_keypts = corr_pos[:, :, 0:3] #[b,n,3]
        tgt_keypts = corr_pos[:, :, 3:6] #[b,n,3]
        if self.initial_inlier_mask_thresh>=0:
            with torch.no_grad():
                corr_compatibility = get_compatibility(src_keypts, tgt_keypts, self.with_dist_corr_compatibility, self.dist_sigma)

            inlier_mask = ((torch.triu(corr_compatibility, diagonal=1)>self.initial_inlier_mask_thresh).sum(dim=-1) >= 1)# [b,n]
        else:
            b,n, _ = src_keypts.shape
            inlier_mask = torch.ones((b,n), dtype=bool)
        
        bs = corr_pos.shape[0]
        num_corr = corr_pos.shape[1]

        all_idx = torch.arange(num_corr).to(self.device).long()
        pred_labels = torch.zeros((bs, num_corr)).to(self.device).long() # [b,n]
        new_initial_labels = torch.zeros(num_corr).to(self.device).long() # [n,]

        for i in range(bs):

            reverse_idx = all_idx[inlier_mask[i]] # inlier_idx

            inlier_data = corr_pos[i][inlier_mask[i]] #[n1,6]
            
            src_keypts = inlier_data[:, 0:3] #[n1,3]
            tgt_keypts = inlier_data[:, 3:6] #[n1,3]
            
        
            with torch.no_grad():
                
                #################################
                # Step3: down_sample and cluster --> initial_label
                #################################
                
                start = time.time()
                N = inlier_data.shape[0]
                
                if N > self.keypts_num:
                    idx = np.random.choice(N, self.keypts_num, replace=False)
                else:
                    idx = np.arange(N)
                    
                downsample_compatibility = get_compatibility(src_keypts[idx], tgt_keypts[idx], self.with_dist_corr_compatibility, self.dist_sigma)
                
                downsample_data = inlier_data[idx]
                
                downsample_initial_label = tanimoto_cluster(downsample_compatibility.T, self.min_dist_thresh, self.device)
                #print(f'initial_num = {downsample_initial_label.max()}')
                
                #################################
                # Step4: upsample_labels and relabel
                #################################
                succ, initial_label = upsample_labels(downsample_data, downsample_initial_label, inlier_data, iou_mask_thresh=self.iou_mask_thresh, min_num4solve_model=self.min_num4solve_model, max_distance=self.inlier_dist_thresh, device=self.device)
                
                if succ:
                    initial_label = new_initial_labels.scatter_(dim=0, index=reverse_idx, src=(initial_label+1).long()) #[n,]

                    #print(f'clustering time:{time.time()-start}')
                    #print(f'initial_num = {initial_label.max()}')
                    start = time.time()
                    #################################
                    # Step5: refine label
                    #################################
                    if self.with_refine:
                        new_label = refine_labels(corr_pos[i], initial_label, iou_mask_thresh=self.iou_mask_thresh, min_num4solve_model=self.min_num4solve_model, num_thresh_multipler=self.num_thresh_multipler,init_inlier_mask=(initial_label>0), with_preference=self.with_preference, max_distance=self.dist_for_preference, iter_num=self.refine_iter_num, device=self.device)
                    else:
                        new_label = initial_label
                    #print(f'new_label_num = {new_label.max()}')
                    #print(f'refine time:{time.time()-start}')
                    #print(succ)
                else:
                    corr_compatibility = get_compatibility(src_keypts, tgt_keypts, self.with_dist_corr_compatibility, self.dist_sigma)
                    #adjm = (corr_compatibility < self.cpat_thresh)
                    initial_label = tanimoto_cluster(corr_compatibility.T, self.min_dist_thresh, self.device)
                    initial_label = new_initial_labels.scatter_(dim=0, index=reverse_idx, src=(initial_label+1).long()) #[n,]
                    #print(f'clustering time:{time.time()-start}')
                    #print(f'initial_num = {initial_label.max()}')
                    
                    start = time.time()
                    if self.with_refine:
                        new_label = refine_labels(corr_pos[i], initial_label, iou_mask_thresh=self.iou_mask_thresh, min_num4solve_model=self.min_num4solve_model, num_thresh_multipler=self.num_thresh_multipler, init_inlier_mask=(initial_label>0), with_preference=self.with_preference, max_distance=self.dist_for_preference, iter_num=self.refine_iter_num,
                        N_multipler=self.N_multipler, device=self.device)
                    else:
                        new_label = initial_label
                    #print(f'refine time:{time.time()-start}')
                
                pred_labels[i] = new_label.long()#pred_labels[i].scatter_(dim=0, index=reverse_idx, src=new_label.long())
                #print(f'new_label_num = {new_label.max()}')

        return pred_labels
        
def get_compatibility(src_keypts, tgt_keypts, with_dist_corr_compatibility, dist_sigma=None):
    # [n,3], [n,3]
    #################################
    # Step1: corr_compatibily
    #################################
    src_dist = torch.norm((src_keypts.unsqueeze(-2) - src_keypts.unsqueeze(-3)), dim=-1) # [n1, n1]
    tgt_dist = torch.norm((tgt_keypts.unsqueeze(-2) - tgt_keypts.unsqueeze(-3)), dim=-1) # [n1, n1]
    
    corr_compatibility = torch.min(torch.stack((src_dist/(tgt_dist+1e-12),tgt_dist/(src_dist+1e-12)), dim=-1), dim=-1).values + torch.eye(src_dist.shape[-1]).to(src_dist)
    corr_compatibility = torch.clamp((corr_compatibility / 0.9)**2, min=0)
    
    #################################
    # Step2: dist_compatibily
    #################################
    if with_dist_corr_compatibility:
        dist_corr_compatibility = (src_dist + tgt_dist)/2
        dist_corr_compatibility = torch.clamp(1.0 - dist_corr_compatibility ** 2 / dist_sigma ** 2, min=0) # [n1, n1]
        corr_compatibility *= dist_corr_compatibility
        corr_compatibility /= corr_compatibility.max(dim=-1, keepdim=True).values.max(dim=-1, keepdim=True).values
    return corr_compatibility

def tanimoto_cluster(adjm, min_dist_thresh, device='cuda'):
    # adjm [m,n]
    adjm = adjm.to(device)
    ncluster = adjm.shape[1]
    cluster_info = {}
    for ss in range(ncluster):
        cluster_info[ss] = set([ss])

    Tanimoto_dist = computeAllPairTanimotoDist(adjm, adjm)# [n1,n2]
    Tanimoto_dist += torch.eye(ncluster, device=device)

    min_dist = 0
    k = 0
    initial_label = torch.zeros(adjm.shape[1]).to(device).long() # [n,]
    while min_dist < min_dist_thresh:
        #find the smallest Tanimoto distance
        cluster_i, cluster_j, min_dist = findTwoClustersToMerge(Tanimoto_dist)
        #merge the two clusters and update the Tanimoto distances between clusters
        adjm, Tanimoto_dist, cluster_info = mergeTwoCluster(adjm, Tanimoto_dist, cluster_info, cluster_i, cluster_j)
        k = k+1
    for ss, values in enumerate(cluster_info.values()):
        idx = np.array(list(values))
        initial_label[idx] = ss + 1
    return initial_label

def computeAllPairTanimotoDist(A,B):
    # [m,n1], [m,n2]
    inner_product = torch.matmul(A.float().T, B.float()) # [n1,n2]
    
    Tanimoto_dist = 1 - inner_product/((A*A).sum(dim=0, keepdim=True).T + (B*B).sum(dim=0, keepdim=True)-inner_product + 1e-12) #[n1, 1],[1,n2]->[n1,n2]
    return Tanimoto_dist# [n1,n2]

def findTwoClustersToMerge(Tanimoto_dist):
    # [n,n]
    values, row_idx = torch.min(Tanimoto_dist, dim=0)
    col_idx = torch.argmin(values)
    i, j = row_idx[col_idx].item(), col_idx.item()
    min_dist = values[col_idx]
    sorted_ind = [i,j]
    sorted_ind.sort()
    cluster_i = sorted_ind[0]
    cluster_j = sorted_ind[1]
    return cluster_i, cluster_j, min_dist

def mergeTwoCluster(adjm, Tanimoto_dist, cluster_info, cluster_i, cluster_j):
    #merge the two clusters
    #adjm [m,n]
    adjm[:,cluster_i] = torch.min(adjm[:,[cluster_i, cluster_j]], dim=1).values
    adjm[:,cluster_j] = 0
    Tanimoto_dist[cluster_j, :] = 1
    Tanimoto_dist[:, cluster_j] = 1
    
    cluster_info[cluster_i] = cluster_info[cluster_i].union(cluster_info[cluster_j])
    cluster_info.pop(cluster_j)
    
    #update the Tanimoto distances
    Tanimoto_dist[:, cluster_i] = Tanimoto_dist[cluster_i, :] = computeAllPairTanimotoDist(adjm[:,cluster_i].unsqueeze(1), adjm).squeeze(0) #[1，n]->[n,]
    Tanimoto_dist[cluster_i,cluster_i] = 1

    return adjm, Tanimoto_dist, cluster_info



def upsample_labels(downdata_ori, downlabels_ori, data_ori, iou_mask_thresh=0.8, min_num4solve_model=3, max_distance=0.0625, device='cuda'):
    '''
    input torch [nl,c]
          torch [nl]
          torch [n,c]
    return torch [n]
    '''
    # default 0 is outlier
    downdata = downdata_ori.to(device)
    downlabels = downlabels_ori.to(device) + 1
    data = data_ori.to(device)
    n = data.shape[0]
    new_label = torch.zeros(n, device=device)

    #inliers = torch.ones_like(labels, device=device)
    nl = downdata.shape[0]
    full_downidx = torch.arange(nl).to(device)
    
    output, counts = torch.unique(downlabels, return_counts=True)
    model_labels = output[counts >= min_num4solve_model]
    h_list = []
    for i in model_labels:
        choice_idx = full_downidx[downlabels==i]
        if choice_idx.shape[0]:
            h_list.append(rigid_tranform_from_points(downdata, choice_idx, device)) # 16
    if len(h_list):
        hs = torch.stack(h_list,dim=0) # [m,16]
        distance = rigid_tranform_consistency_measure(hs.unsqueeze(0), data.unsqueeze(0), device).squeeze(0) # [b,n,m]->[n,m]
        #--------- merge similar model ------------
        preference = (distance < max_distance) # [n,m]
        intersection = torch.matmul(preference.float().T, preference.float()) # [m,m]
        union = torch.logical_or(preference.T.unsqueeze(1), preference.T.unsqueeze(0)).float().sum(-1) + 1e-12 # [m,m]
        iou_mask = (intersection/union)>iou_mask_thresh
        unique_mask = torch.logical_not((torch.triu(iou_mask, diagonal=1).sum(-1))>0)
        hs = hs[unique_mask]
        distance = distance[:, unique_mask]
        #------------------------------------------
        values, assign_label = torch.min(distance, dim=1) #[n,]

        inlier_mask = values < max_distance
        
        new_label[inlier_mask] = (assign_label + 1)[inlier_mask].to(new_label)
        new_label[torch.logical_not(inlier_mask)] = 0
    else:
        return False, 0

    return True, new_label.to(device)


def refine_labels(data, labels, iou_mask_thresh=0.8, min_num4solve_model=3, with_preference=True, max_distance=0.0625, iter_num=3, num_thresh_multipler=5, N_multipler=100, init_inlier_mask=None, device='cuda'):
    '''
    input torch [n,c]
          torch [n]
    return torch [n]
    '''
    # default 0 is outlier
    if init_inlier_mask != None:
        iter_data = data.to(device)[init_inlier_mask]
        iter_labels = labels.to(device)[init_inlier_mask] + 1
    else:
        iter_data = data.to(device)
        iter_labels = labels.to(device) + 1
    data_num = data.shape[0]
    new_label = labels.to(device) + 1
    for iter_i in range(iter_num):

        n = iter_data.shape[0]
        full_idx = torch.arange(n).to(device)
        
        output, counts = torch.unique(iter_labels, return_counts=True)
        num_thresh = min(min_num4solve_model*(num_thresh_multipler**iter_i), int(data_num/N_multipler))
        model_labels = output[counts >= num_thresh]

        h_list = []
        for i in model_labels:
            choice_idx = full_idx[iter_labels==i]
            if choice_idx.shape[0]:
                h_list.append(rigid_tranform_from_points(iter_data, choice_idx, device))
        if len(h_list):
            hs = torch.stack(h_list,dim=0) # [m,16]
            distance = rigid_tranform_consistency_measure(hs.unsqueeze(0), data.unsqueeze(0), device).squeeze(0) # [b,n,m]->[n,m]
            #--------- merge similar model ------------
            if with_preference:
                preference = (distance < max_distance) # [n,m]

                intersection = torch.matmul(preference.float().T, preference.float()) # [m,m]
                union = torch.logical_or(preference.T.unsqueeze(1), preference.T.unsqueeze(0)).float().sum(-1) + 1e-12 # [m,m]
                iou_mask = (intersection/union)>iou_mask_thresh

                unique_mask = torch.logical_not((torch.triu(iou_mask, diagonal=1).sum(-1))>0)
                hs = hs[unique_mask]
                distance = distance[:, unique_mask]
            #------------------------------------------
            values, assign_label = torch.min(distance, dim=1) #[n,]

            inlier_mask = values < max_distance#*max((0.5**iter_i), 0.1)
            
            if inlier_mask.sum():
                iter_data = data[inlier_mask]
                iter_labels = labels[inlier_mask]
                new_label[inlier_mask] = (assign_label + 1)[inlier_mask].to(new_label)
                new_label[torch.logical_not(inlier_mask)] = 0
            else:
                iter_data = data
                iter_labels = assign_label + 1


    return new_label
    

def regular_score(score):
    score = np.where(np.isnan(score), np.zeros_like(score), score)
    score = np.where(np.isinf(score), np.zeros_like(score), score)
    return score        
def me(output_labels: torch.Tensor, gt_labels: torch.Tensor):
    # numpy [b,n], [b,n]
    true_labels = gt_labels.astype(np.int)
    estm_labels = output_labels.astype(np.int)
    assert true_labels.shape == estm_labels.shape
    
    b = true_labels.shape[0]
    n = true_labels.shape[1]

    num_true_labels = np.max(true_labels) + 1
    num_estm_labels = np.max(estm_labels) + 1
    
    true_clusters = np.zeros((b, num_true_labels, n)).astype(np.int) # [b, k1, n]
    estm_clusters = np.zeros((b, num_estm_labels, n)).astype(np.int) # [b, k2, n]

    for li in range(num_true_labels):
        mask = (true_labels == li) # [b,n]
        true_clusters[:, li, :] = mask
        
    for li in range(num_estm_labels):
        mask = (estm_labels == li) # [b,n]
        estm_clusters[:, li, :] = mask

    intersection = np.einsum("ijk,ilk->ijl", true_clusters, estm_clusters) # [b, k1, k2]
    union = np.sum(np.logical_or(true_clusters[:, :, None, :], estm_clusters[:, None, :, :]), axis=-1) # [b, k1, k2, n]->[b,k1,k2]
    
    batch_iou_matrix = 1. * intersection/(union+1e-12) # [b, k1, k2]
    assigned_labels = -1 * np.ones_like(estm_labels) #[b,n]

    batch_iou_matrix = regular_score(batch_iou_matrix)

    for i, iou_matrix in enumerate(batch_iou_matrix):

        true_ind, pred_ind = scipy.optimize.linear_sum_assignment(1 - iou_matrix)
        for idx, ei in enumerate(pred_ind):
            ai = true_ind[idx]
            mask = (estm_labels[i] == ei) #[b,n]
            assigned_labels[i, mask] = ai
    
    correct = (assigned_labels == true_labels) #[b,n]
    false = np.logical_not(correct) #[b,n]
    
    batched_num_correct = np.sum(correct, axis=-1) #[b,]
    batched_num_false = np.sum(false, axis=-1) #[b,]
    
    batched_miss_rate = batched_num_false * 1. / (batched_num_false + batched_num_correct) #[b,]
    
    batched_mean_miss_rate = np.mean(batched_miss_rate)

    mean_iou = 0
    
    for bi in range(b):
        iou = 0
        label_sum = 0
        for i in range(max(true_labels[bi])+1):
            true_mask = (true_labels[bi] == i) #[n]
            pred_mask = (assigned_labels[bi] == i) #[n]
            if true_mask.sum():
                iou += (true_mask & pred_mask).sum()/(true_mask | pred_mask).sum() # intersection/union
                label_sum += 1
        mean_iou += (iou/label_sum)/b

    return batched_mean_miss_rate, assigned_labels, mean_iou

def pred_model(data, pred_labels, min_num4solve_model=3, device='cuda', lambda_thresh=0.5):
    '''
    torch [n,c]
            [n]
    retrun: bool, [m,16]
    '''
    data = data.to(device)
    pred_labels = pred_labels.to(device)
    pred_labels_wo_outlier = pred_labels[pred_labels>0].to(device)
    #print(f'pred_labels_min:{pred_labels.min()}, pred_labels_max:{pred_labels.max()}')
    n = data.shape[0]
    full_idx = torch.arange(n).to(device)
    output, counts = torch.unique(pred_labels_wo_outlier, return_counts=True)
    model_labels = output[counts >= min_num4solve_model]
    model_labels_counts = counts[counts >= min_num4solve_model]
    if not len(model_labels_counts):
        return False, 0, 0

    h_list = []
    label_list = []
    for i in model_labels:
        choice_idx = full_idx[pred_labels==i]
        if choice_idx.shape[0]:
            h_list.append(rigid_tranform_from_points(data, choice_idx, device)) # 16
            label_list.append(i)
    idx = torch.argsort(model_labels_counts, descending=True)

    sort_model_number = model_labels_counts[idx]

    diff_over_the_biggest = sort_model_number/sort_model_number[0]
    #print(f'model_point_num = {sort_model_number}')
    #print(f'diff over the biggest = {diff_over_the_biggest}')
    hs_mask = torch.arange(idx.shape[0]).to(idx)[diff_over_the_biggest<lambda_thresh] #0.25
    
    if len(h_list):
        hs = torch.stack(h_list,dim=0) # [m,16]
        label_list = torch.stack(label_list)
        if hs_mask.shape[0]:
            label_list = label_list[idx][:(hs_mask[0])]
            hs = hs[idx][:(hs_mask[0])]
            return True, hs, label_list # [m,16]
        else:
            label_list = label_list[idx]
            hs = hs[idx]
            return True, hs, label_list # [m,16]
    else:
        return False, 0, 0

def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T
        
def get_re_te(pred_model, gt_model, device='cuda'):
    #[m,16], [obj_num, 4, 4]
    pred_model, gt_model = pred_model.to(device), gt_model.to(device)
    pred_model44 = pred_model.view(-1, 4, 4) #[m,4,4]
    gt_model16 = gt_model.view(-1, 16) # [obj_num, 16]

    sim_matrix = (pred_model.unsqueeze(1) - gt_model16.unsqueeze(0)).norm(dim=-1)# [m, obj_num]
    pred_ind, true_ind = scipy.optimize.linear_sum_assignment(sim_matrix.cpu().numpy())
    #print(f'get_re_te:{true_ind}, {pred_ind}')
    pred_model44 = pred_model44[pred_ind] # [m, 4, 4]
    gt_model = gt_model[true_ind] # [obj_num, 4, 4]
    R = pred_model44[:, :3, :3].float() # [m, 3, 3]
    t = pred_model44[:, :3, 3].float() # [m, 3]
    gt_R = gt_model[:, :3, :3].float() # [m, 3, 3]
    gt_t = gt_model[:, :3, 3].float() # [m, 3]
    re_list, te_list = [], []
    for i in range(gt_model.shape[0]):
        re = torch.acos(torch.clamp((torch.trace(R[i].T @ gt_R[i]) - 1) / 2.0, min=-1, max=1))
        te = torch.sqrt(torch.sum((t[i] - gt_t[i]) ** 2))
        re = re * 180 / np.pi
        re_list.append(re.item())
        te_list.append(te.item())
    mean_re, mean_te = np.mean(re_list), np.mean(te_list)

    return true_ind, pred_ind, re_list, te_list, mean_re, mean_te
#-------------------------------------------------------------
def relabel(ori_labels, hs, data, device='cuda'):
    #[n,], [m, 4, 4], [n,6]
    hs, data = hs.to(device), data.to(device)
    inlier_mask = ori_labels > 0
    new_label = ori_labels
    distance = rigid_tranform_consistency_measure(hs.unsqueeze(0), data.unsqueeze(0), device).squeeze(0) # [b,n,m]->[n,m]
    values, assign_label = torch.min(distance, dim=1) #[n,]
    
    new_label[inlier_mask] = (assign_label + 1)[inlier_mask].to(new_label)
    new_label[torch.logical_not(inlier_mask)] = 0
    return new_label
         