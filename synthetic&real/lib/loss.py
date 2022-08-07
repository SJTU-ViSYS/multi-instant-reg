"""
Loss functions

Author: Shengyu Huang
Last modified: 30.11.2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from lib.utils import square_distance
from sklearn.metrics import precision_recall_fscore_support
import o3d_tools.visualize_tools as vt
import open3d as o3d
class MetricLoss(nn.Module):
    """
    We evaluate both contrastive loss and circle loss
    """
    def __init__(self,configs,log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(MetricLoss,self).__init__()
        self.config = configs
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = configs.pos_margin
        self.neg_margin = configs.neg_margin
        self.max_points = configs.max_points

        self.safe_radius = configs.safe_radius 
        self.matchability_radius = configs.matchability_radius
        self.pos_radius = configs.pos_radius # just to take care of the numeric precision
    
    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """
        
        pos_mask = coords_dist < self.pos_radius 
        neg_mask = coords_dist > self.safe_radius 

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()
        
        
        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive 
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() 

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2
        #print(loss_row[row_sel].shape)
        #print(loss_col[col_sel].shape)
        
        if loss_row[row_sel].shape[0]==0 or loss_col[col_sel].shape[0]==0:# or coords_dist.shape[0] < self.max_points:
            pass

        return circle_loss

    def get_recall(self,coords_dist,feats_dist):
        """
        Get feature match recall, divided by number of true inliers
        """
        pos_mask = coords_dist < self.pos_radius
        n_gt_pos = (pos_mask.sum(-1)>0).float().sum()+1e-12
        _, sel_idx = torch.min(feats_dist, -1)
        sel_dist = torch.gather(coords_dist,dim=-1,index=sel_idx[:,None])[pos_mask.sum(-1)>0]
        n_pred_pos = (sel_dist < self.pos_radius).float().sum()
        recall = n_pred_pos / n_gt_pos
        return recall

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt) 

        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0) 
        w_positive = 1 - w_negative  
        
        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(),predicted_labels, average='binary')

        return w_class_loss, cls_precision, cls_recall
            
    def contrastive_loss(self, F0, F1, pos_pairs, neg_pairs, neg_weight):
        # pairs consist of (xyz1 index, xyz0 index)
        neg0 = F0.index_select(0, neg_pairs[:, 0])
        neg1 = F1.index_select(0, neg_pairs[:, 1])
        pos0 = F0.index_select(0, pos_pairs[:, 0])
        pos1 = F1.index_select(0, pos_pairs[:, 1])

        # Positive loss
        pos_loss = (pos0 - pos1).pow(2).sum(1)

        # Negative loss
        neg_loss = F.relu(self.neg_margin -
                            ((neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)

        pos_loss_mean = pos_loss.mean()
        neg_loss_mean = neg_loss.mean()

        # Weighted loss
        loss = pos_loss_mean + neg_weight * neg_loss_mean
        return loss

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, rot, trans,scores_overlap,scores_saliency):
        """
        Circle loss for metric learning, here we feed the positive pairs only
        Input:
            src_pcd:        [N, 3]  
            tgt_pcd:        [M, 3]
            rot:            [3, 3]
            trans:          [3, 1]
            src_feats:      [N, C]
            tgt_feats:      [M, C]
        """
        #print(correspondence)
        if not self.config.dataset in ['multi', 'box', 'bigger_box', 'semantic', 'ipa', 'multi_multi']:
            #print(rot.shape, src_pcd.transpose(0,1).shape, trans.unsqueeze(1).shape)
            src_pcd = (torch.matmul(rot,src_pcd.transpose(0,1))+trans.unsqueeze(1)).transpose(0,1)
        #o3d.visualization.draw_geometries([vt.visualize_pcd(src_pcd.cpu().numpy(), color=vt.SRC_COLOR, visual=False), vt.visualize_pcd(tgt_pcd.cpu().numpy(), color=vt.TGT_COLOR, visual=False)])
        stats=dict()

        src_idx = np.asarray(list(set(correspondence[:,0].int().tolist())))
        tgt_idx = np.asarray(list(set(correspondence[:,1].int().tolist())))

        #######################
        # get BCE loss for overlap, here the ground truth label is obtained from correspondence information
        src_gt = torch.zeros(src_feats.size(0))
        src_gt[src_idx%src_feats.shape[0]]=1.
        tgt_gt = torch.zeros(tgt_feats.size(0))
        tgt_gt[tgt_idx]=1.
        gt_labels = torch.cat((src_gt, tgt_gt)).to(torch.device('cuda'))
        #print(scores_overlap.shape, gt_labels.shape)
        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_overlap, gt_labels)
        stats['overlap_loss'] = class_loss
        stats['overlap_recall'] = cls_recall
        stats['overlap_precision'] = cls_precision

        #######################
        # get BCE loss for saliency part, here we only supervise points in the overlap region
        src_feats_sel, src_pcd_sel = src_feats[src_idx%src_feats.shape[0]], src_pcd[src_idx]
        tgt_feats_sel, tgt_pcd_sel = tgt_feats[tgt_idx], tgt_pcd[tgt_idx]
        
        scores = torch.matmul(src_feats_sel, tgt_feats_sel.transpose(0,1))
        _, idx = scores.max(1)
        distance_1 = torch.norm(src_pcd_sel - tgt_pcd_sel[idx], p=2, dim=1)
        _, idx = scores.max(0)
        distance_2 = torch.norm(tgt_pcd_sel - src_pcd_sel[idx], p=2, dim=1)

        gt_labels = torch.cat(((distance_1<self.matchability_radius).float(), (distance_2<self.matchability_radius).float()))

        src_saliency_scores = scores_saliency[:src_feats.size(0)][src_idx%src_feats.shape[0]]
        tgt_saliency_scores = scores_saliency[src_feats.size(0):][tgt_idx]
        scores_saliency = torch.cat((src_saliency_scores, tgt_saliency_scores))

        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_saliency, gt_labels)
        stats['saliency_loss'] = class_loss
        stats['saliency_recall'] = cls_recall
        stats['saliency_precision'] = cls_precision

        #######################################
        # filter some of correspondence as we are using different radius for "overlap" and "correspondence"
        c_dist = torch.norm(src_pcd[correspondence[:,0]] - tgt_pcd[correspondence[:,1]], dim = 1)
        c_select = c_dist < self.pos_radius - 0.001
        #print(f'c_dist = {c_dist}, {self.pos_radius - 0.001}')
        #print(correspondence.shape)
        correspondence = correspondence[c_select]
        #print(correspondence.shape)
        if(correspondence.size(0) > self.max_points):
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]
        
        src_idx = correspondence[:,0]
        tgt_idx = correspondence[:,1]
        src_pcd_raw, tgt_pcd_raw = src_pcd, tgt_pcd
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[src_idx%src_feats.shape[0]], tgt_feats[tgt_idx]
        #print(src_feats.shape)
        #######################
        # get L2 distance between source / target point cloud
        coords_dist = torch.sqrt(square_distance(src_pcd[None,:,:], tgt_pcd[None,:,:]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(src_feats[None,:,:], tgt_feats[None,:,:],normalised=True)).squeeze(0)

        ##############################
        # get FMR and circle loss
        ##############################
        #print(feats_dist.shape)
        recall = self.get_recall(coords_dist, feats_dist)
        
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)

        print(recall, circle_loss)
        stats['circle_loss']= circle_loss
        stats['recall']=recall

        return stats

class MetricLoss_contrastive(nn.Module):
    """
    We evaluate both contrastive loss and circle loss
    """
    def __init__(self,configs,log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(MetricLoss_contrastive,self).__init__()
        self.config = configs
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = configs.pos_margin
        self.neg_margin = configs.neg_margin
        self.max_points = configs.max_points

        self.safe_radius = configs.safe_radius 
        self.matchability_radius = configs.matchability_radius
        self.pos_radius = configs.pos_radius # just to take care of the numeric precision
    
    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """
        
        pos_mask = coords_dist < self.pos_radius 
        neg_mask = coords_dist > self.safe_radius 

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
        col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()
        
        
        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive 
        pos_weight = (pos_weight - self.pos_optimal) # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() 

        neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight) # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight,dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight,dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row)/self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col)/self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2
        #print(loss_row[row_sel].shape)
        #print(loss_col[col_sel].shape)
        
        if loss_row[row_sel].shape[0]==0 or loss_col[col_sel].shape[0]==0:# or coords_dist.shape[0] < self.max_points:
            pass
            
        return circle_loss

    def get_recall(self,coords_dist,feats_dist):
        """
        Get feature match recall, divided by number of true inliers
        """
        pos_mask = coords_dist < self.pos_radius
        n_gt_pos = (pos_mask.sum(-1)>0).float().sum()+1e-12
        _, sel_idx = torch.min(feats_dist, -1)
        sel_dist = torch.gather(coords_dist,dim=-1,index=sel_idx[:,None])[pos_mask.sum(-1)>0]
        n_pred_pos = (sel_dist < self.pos_radius).float().sum()
        recall = n_pred_pos / n_gt_pos
        return recall

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt) 

        weights = torch.ones_like(gt)
        w_negative = gt.sum()/gt.size(0) 
        w_positive = 1 - w_negative  
        
        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(),predicted_labels, average='binary')

        return w_class_loss, cls_precision, cls_recall
            
    def contrastive_loss(self, F0, F1, pos_pairs, neg_pairs, neg_weight):
        # pairs consist of (xyz1 index, xyz0 index)
        neg0 = F0.index_select(0, neg_pairs[:, 0])
        neg1 = F1.index_select(0, neg_pairs[:, 1])
        pos0 = F0.index_select(0, pos_pairs[:, 0])
        pos1 = F1.index_select(0, pos_pairs[:, 1])

        # Positive loss
        pos_loss = (pos0 - pos1).pow(2).sum(1)

        # Negative loss
        neg_loss = F.relu(self.neg_margin -
                            ((neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)

        pos_loss_mean = pos_loss.mean()
        neg_loss_mean = neg_loss.mean()

        # Weighted loss
        loss = pos_loss_mean + neg_weight * neg_loss_mean
        return loss

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, rot, trans,scores_overlap,scores_saliency):
        """
        Circle loss for metric learning, here we feed the positive pairs only
        Input:
            src_pcd:        [N, 3]  
            tgt_pcd:        [M, 3]
            rot:            [3, 3]
            trans:          [3, 1]
            src_feats:      [N, C]
            tgt_feats:      [M, C]
        """
        #print(correspondence)
        if not self.config.dataset in ['multi', 'box', 'bigger_box']:
            #print(rot.shape, src_pcd.transpose(0,1).shape, trans.unsqueeze(1).shape)
            src_pcd = (torch.matmul(rot,src_pcd.transpose(0,1))+trans.unsqueeze(1)).transpose(0,1)
        
        stats=dict()
        
        src_idx = list(set(correspondence[:,0].int().tolist()))
        tgt_idx = list(set(correspondence[:,1].int().tolist()))

        #######################
        # get BCE loss for overlap, here the ground truth label is obtained from correspondence information
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx]=1.
        tgt_gt = torch.zeros(tgt_pcd.size(0))
        tgt_gt[tgt_idx]=1.
        gt_labels = torch.cat((src_gt, tgt_gt)).to(torch.device('cuda'))
        #print(scores_overlap.shape, gt_labels.shape)
        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_overlap, gt_labels)
        stats['overlap_loss'] = class_loss
        stats['overlap_recall'] = cls_recall
        stats['overlap_precision'] = cls_precision

        #print(correspondence)
        #######################
        # get BCE loss for saliency part, here we only supervise points in the overlap region
        src_feats_sel, src_pcd_sel = src_feats[src_idx], src_pcd[src_idx]
        tgt_feats_sel, tgt_pcd_sel = tgt_feats[tgt_idx], tgt_pcd[tgt_idx]
        #print(correspondence)

        scores = torch.matmul(src_feats_sel, tgt_feats_sel.transpose(0,1))
        _, idx = scores.max(1)
        distance_1 = torch.norm(src_pcd_sel - tgt_pcd_sel[idx], p=2, dim=1)
        _, idx = scores.max(0)
        distance_2 = torch.norm(tgt_pcd_sel - src_pcd_sel[idx], p=2, dim=1)

        gt_labels = torch.cat(((distance_1<self.matchability_radius).float(), (distance_2<self.matchability_radius).float()))

        src_saliency_scores = scores_saliency[:src_pcd.size(0)][src_idx]
        tgt_saliency_scores = scores_saliency[src_pcd.size(0):][tgt_idx]
        scores_saliency = torch.cat((src_saliency_scores, tgt_saliency_scores))

        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_saliency, gt_labels)
        stats['saliency_loss'] = class_loss
        stats['saliency_recall'] = cls_recall
        stats['saliency_precision'] = cls_precision

        #######################################
        # filter some of correspondence as we are using different radius for "overlap" and "correspondence"
        c_dist = torch.norm(src_pcd[correspondence[:,0]] - tgt_pcd[correspondence[:,1]], dim = 1)
        c_select = c_dist < self.pos_radius - 0.001
        #print(f'c_dist = {c_dist}, {self.pos_radius - 0.001}')
        #print(correspondence.shape)
        correspondence = correspondence[c_select]
        #print(correspondence.shape)
        if(correspondence.size(0) > self.max_points):
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]
        
        src_idx = correspondence[:,0]
        tgt_idx = correspondence[:,1]
        src_pcd_raw, tgt_pcd_raw = src_pcd, tgt_pcd
        src_feats_raw, tgt_feats_raw = src_feats, tgt_feats
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[src_idx], tgt_feats[tgt_idx]
        #print(src_feats.shape)
        #######################
        # get L2 distance between source / target point cloud
        coords_dist = torch.sqrt(square_distance(src_pcd[None,:,:], tgt_pcd[None,:,:]).squeeze(0))
        feats_dist = torch.sqrt(square_distance(src_feats[None,:,:], tgt_feats[None,:,:],normalised=True)).squeeze(0)
        ##########################################################################################
        ##############################
        # get FMR and circle loss
        ##############################
        #print(feats_dist.shape)
        recall = self.get_recall(coords_dist, feats_dist)
        #print(recall)
        circle_loss = self.contrastive_loss(src_feats_raw, tgt_feats_raw, correspondence[:, 0:2], correspondence[:, 2:4], 1)

        print(recall, circle_loss)
        stats['circle_loss']= circle_loss
        stats['recall']=recall

        return stats
