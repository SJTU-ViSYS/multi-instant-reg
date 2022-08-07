
from lib.trainer import Trainer
import os, torch
from tqdm import tqdm
import numpy as np
from lib.benchmark_utils import to_array, batch_to_tsfm
import open3d as o3d
import o3d_tools.visualize_tools as vt
# Modelnet part
from common.math_torch import se3
from common.math.so3 import dcm2euler

from tanimoto import Tanimoto, me, pred_model, get_re_te, relabel

import time
from lib.logger import Logger
import o3d_tools.visualize_tools as vt

class seg_model_configs:
    def __init__(self, config=None):
        self.wo_dist_corr_compatibility = False#True
        self.wo_refine = False
        self.wo_preference = False
        
        #------------------------------
        if config:
            self.lambda_thresh = config.lambda_thresh
            self.min_dist_thresh = config.min_dist_thresh
            self.inlier_dist_thresh = config.inlier_dist_thresh
            self.iou_mask_thresh = config.iou_mask_thresh
            self.N_multipler = config.N_multipler
        else:
            self.lambda_thresh = 0.5
            self.min_dist_thresh = 0.2#0.1
            self.inlier_dist_thresh = 0.3
            self.iou_mask_thresh = 0.9
            self.N_multipler = 100
        #------------------------------

        self.dist_for_preference = self.inlier_dist_thresh
        self.refine_iter_num = 3
        self.initial_inlier_mask_thresh = -1#-1 #0, -1 = without initial outlier removal

        self.cpat_thresh = 0.8

        self.n_clusters = 10

        self.sigma_spat = 0.4
        self.dist_sigma = 2

        self.keypts_num = 1024
        self.min_num4solve_model = 3
        
        self.num_thresh_multipler = 3
        self.device = 'cuda'

def ABfind_nndist(A,B):
    from sklearn.neighbors import KDTree
    # [n,3], [m,3]
    tree = KDTree(B)
    nndist, idx = tree.query(A, return_distance=True)
    return nndist.squeeze(-1), idx.squeeze(-1) #[n,], [n,]


class IndoorTester(Trainer):
    """
    3DMatch tester
    """
    def __init__(self,args):
        Trainer.__init__(self,args)
    def ir(self, trans_src, tgt, corr, inlier_thresh):
        # [b,n,3], [b,n,3], [b,corr_num, 2]
        corr_trans_src = trans_src.gather(dim=1, index=corr[:, :, 0].unsqueeze(dim=-1).expand(-1, -1, 3))#[b,corr_num,3]
        corr_tgt = tgt.gather(dim=1, index=corr[:, :, 1].unsqueeze(dim=-1).expand(-1, -1, 3))#[b,corr_num,3]
        
        inliermask = ((corr_trans_src - corr_tgt).norm(dim=-1) < inlier_thresh)#[b,corr_num,3]->[b,corr_num]
        ir = inliermask.float().mean()
        return ir, inliermask
    def test(self):
        logging = Logger(f'snapshot/{self.config.exp_dir}/{self.config.logfile}')
        print('Start to evaluate on test datasets...')
        #self.loader['test'].dataset[137]
        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        self.model.eval()
        smconfigs = seg_model_configs(self.config)
        seg_model = Tanimoto(smconfigs).cuda().eval()
        with torch.no_grad():
            mean_pred_num = []
            mean_src_ir, mean_re, mean_te = [], [], []
            mean_hr = []
            mean_prec = []
            mean_f1 = []
            mean_time = []
            skip = []
            
            
            for idx in tqdm(range(num_iter)): # num_iter loop through this epoch
                
                inputs = c_loader_iter.next()
                ##################################
                # load inputs to device.
                for k, v in inputs.items():  
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    elif type(v) == tuple:
                        continue
                    else:
                        inputs[k] = v.to(self.device)
                ###############################################
                # forward pass
                len_src_c = inputs['stack_lengths'][-1][0]
                pcd_c = inputs['points'][-1]
                src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]
                if src_pcd_c.shape[0] <=1 or tgt_pcd_c.shape[0] <=1:
                    skip.append(idx)
                    continue
                feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
                pcd = inputs['points'][0]
                len_src = inputs['stack_lengths'][0][0]

                
                

                src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]


                if self.config.dataset in ['box']:
                    trans_src = inputs['src_pcd_raw'].unsqueeze(0)#[N1, 3]
                    raw_tgt = inputs['tgt_pcd_raw'].unsqueeze(0)#[N2, 3]
                    feat1, feat2 = src_feats.unsqueeze(0), tgt_feats.unsqueeze(0)

                    
                    b, n1, c = feat1.shape
                    _, n2, _ = feat2.shape

                    raw_tgt, trans_src = raw_tgt.to(feat1.device), trans_src.to(feat1.device)
                    trans_src_xyz = trans_src[..., :3]
                    tgt = raw_tgt[..., :3]

                    
                    _, src_tgt = ABfind_nndist(feat1.squeeze(0).cpu().numpy(),feat2.squeeze(0).cpu().numpy())#[n1,], [n1,]
                    src_tgt = torch.from_numpy(src_tgt).unsqueeze(0).to(feat1.device)#[b,n1]
                    inlier_thresh = self.config.overlap_radius
                    
                    # wo_mutual, src
                    corr = torch.stack((torch.arange(n1, device=src_tgt.device).unsqueeze(0).expand(b, -1), src_tgt), dim=-1)#
                    src_ir, inliermask = self.ir(trans_src_xyz, tgt, corr, inlier_thresh)
                    mean_src_ir.append(src_ir.item())
                   
                    
                    if self.config.benchmark_set_split!='train':
                        src_labels = trans_src[..., 3]
                        src_labels[torch.logical_not(inliermask)] = 0
                    
                    # # Analyze the proportion of each objects in the input correspondences
                    # valid_num = 0
                    # for i in range(src_labels[0].int().max().item()+1):
                    #     mask = (src_labels[0] == i)
                    #     print(f'gt label {i} num: {mask.float().sum()} ratio: {mask.float().mean()}')
                    #     if mask.float().mean() > 0.1 and i:
                    #         valid_num += 1
                    
                    corr_pos = torch.cat((src_pcd[corr[..., 0]], tgt_pcd[corr[..., 1]]), dim=-1)

                    if self.config.benchmark_set_split=='test':
                        gt_models = inputs['rot']
                    else:
                        # just for train and valid:
                        gt_models = torch.from_numpy(batch_to_tsfm(inputs['rot'].cpu().numpy(),inputs['trans'].cpu().numpy())).to(corr_pos.device)

                    if torch.max(torch.abs(gt_models))>1000:
                        skip.append(idx)
                        continue
                    
                    # clustering and prediction
                    corr_pos = corr_pos.to(seg_model.device)
                    start = time.time()
                    pred_labels = seg_model(corr_pos)
                    pred_labels = pred_labels.to(seg_model.device)
                    succ, hs, label_list = pred_model(corr_pos[0], pred_labels[0], min_num4solve_model=10, lambda_thresh=smconfigs.lambda_thresh, device=seg_model.device)
                    pred_time = time.time()- start


                    mean_time.append(pred_time)
                    #logging.info(f'mean_time = {np.mean(mean_time)}')
                    
                    
                    if succ:
                        
                        gt_num = gt_models.shape[0]
                        pred_num = hs.shape[0]

                        mean_pred_num.append(pred_num)

                        
                        true_ind, pred_ind, re_list, te_list, m_re, m_te = get_re_te(hs, gt_models)#[:measure_num]
                        re = np.asarray(re_list)
                        te = np.asarray(te_list)
                        

                        mask = ((re < 20) & (te < 0.5))

                        if mask.sum():
                            mean_re.append(np.mean(re[mask]))
                            mean_te.append(np.mean(te[mask]))
                            recall = np.sum(mask)/gt_num
                            precision = np.sum(mask)/pred_num
                            f1 = 2*precision*recall/(precision+recall+1e-12)
                            mean_hr.append(recall)
                            mean_prec.append(precision)
                            mean_f1.append(f1)
                        else:
                            mean_hr.append(0)
                            mean_prec.append(0)
                            mean_f1.append(0)
                    else:
                        mean_hr.append(0)
                        mean_prec.append(0)
                        mean_f1.append(0)
            logging.info(f'mean_re = {np.mean(mean_re)}, mean_te={np.mean(mean_te)}, mean_hit_ratio={np.mean(mean_hr):.4f}, mean_prec={np.mean(mean_prec):.4f}, mean_f1={np.mean(mean_f1):.4f}, mean_time = {np.mean(mean_time)}')
            print(f'mean_re = {np.mean(mean_re)}, mean_te={np.mean(mean_te)}, mean_hit_ratio={np.mean(mean_hr):.4f}, mean_prec={np.mean(mean_prec):.4f}, mean_f1={np.mean(mean_f1):.4f}, mean_time = {np.mean(mean_time)}')
class BiggerTester(Trainer):
    """
    3DMatch tester
    """
    def __init__(self,args):
        Trainer.__init__(self,args)
    def biggerir(self, part_num, src_num, trans_src, tgt, corr, inlier_thresh):
        # [b,n,3], [b,n,3], [b,corr_num, 2]
        corr_trans_src = []
        for i in range(part_num):
            corr_trans_src.append(trans_src.gather(dim=1, index=(corr[:, :, 0]+i*src_num).unsqueeze(dim=-1).expand(-1, -1, 3)))#[b,corr_num,3]
        corr_trans_src = torch.stack(corr_trans_src, dim=1)#[b, part_num, corr_num,3]
        corr_tgt = tgt.gather(dim=1, index=corr[:, :, 1].unsqueeze(dim=-1).expand(-1, -1, 3))#[b,corr_num,3]
        
        inliermask = ((corr_trans_src - corr_tgt.unsqueeze(1)).norm(dim=-1).min(dim=1).values < inlier_thresh)#[b, part_num,corr_num,3]->[b, part_num,corr_num]->[b,corr_num]
        ir = inliermask.float().mean()
        return ir, inliermask
    

    def test(self):
        print('Start to evaluate on test datasets...')
        logging = Logger(f'snapshot/{self.config.exp_dir}/{self.config.logfile}')
        os.makedirs(f'{self.snapshot_dir}/{self.config.benchmark}',exist_ok=True)
        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        self.model.eval()
        smconfigs = seg_model_configs(self.config)
        seg_model = Tanimoto(smconfigs).eval()
        with torch.no_grad():
            mean_pred_num = []
            mean_tgt_ir, mean_re, mean_te = [], [], []
            mean_hr = []
            mean_prec = []
            mean_f1 = []
            mean_time = []
            skip = []
            for idx in tqdm(range(num_iter)): # num_iter loop through this epoch
                inputs = c_loader_iter.next()

                ##################################
                # load inputs to device.
                for k, v in inputs.items():  
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    elif type(v) == tuple:
                        continue
                    else:
                        inputs[k] = v.to(self.device)
                ###############################################
                # forward pass
                len_src_c = inputs['stack_lengths'][-1][0]
                pcd_c = inputs['points'][-1]
                src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]
                if src_pcd_c.shape[0] <=1 or tgt_pcd_c.shape[0] <=1:
                    skip.append(idx)
                    continue
                feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
                pcd = inputs['points'][0]
                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'][:, :3, :3], inputs['trans']
            


                src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]


                if self.config.dataset in ['multi_multi']: 
                    trans_src = inputs['src_pcd_raw'].unsqueeze(0)#[N1, 3]
                    raw_tgt = inputs['tgt_pcd_raw'].unsqueeze(0)#[N2, 3]
                    feat1, feat2 = src_feats.unsqueeze(0), tgt_feats.unsqueeze(0)

                    
                    
                    b, n1, c = feat1.shape
                    _, n2, _ = feat2.shape

                    tgt, trans_src_xyz = raw_tgt[..., :3].to(feat1.device), trans_src.to(feat1.device)
                    if self.config.benchmark_set_split!='train':
                        tgt_labels = raw_tgt[..., 3]
                    
                    
                    _, tgt_src = ABfind_nndist(feat2.squeeze(0).cpu().numpy(),feat1.squeeze(0).cpu().numpy())#[n2,], [n2,]

                    tgt_src = torch.from_numpy(tgt_src).unsqueeze(0).to(feat2.device)#[b,n2]

                    inlier_thresh = self.config.overlap_radius

                    # wo_mutual, tgt
                    corr = torch.stack((tgt_src, torch.arange(n2, device=tgt_src.device).unsqueeze(0).expand(b, -1)), dim=-1)#[b,n2,2]
                    
                    
                    tgt_ir, inliermask = self.biggerir(c_rot.shape[0],n1, trans_src_xyz, tgt, corr, inlier_thresh)

                    if self.config.benchmark_set_split!='train':
                        tgt_labels[torch.logical_not(inliermask)] = 0
                    
                    mean_tgt_ir.append(tgt_ir.item())

                    # # Analyze the proportion of each objects in the input correspondences
                    # if self.config.benchmark_set_split!='train':
                    #     valid_num = 0
                    #     for i in range(tgt_labels[0].int().max().item()+1):
                    #         mask = (tgt_labels[0] == i)
                    #         print(f'gt label {i} num: {mask.float().sum()} ratio: {mask.float().mean()}')
                    #         if mask.float().mean() > 0.1 and i:
                    #             valid_num += 1
                    #print(src_labels)
                    
                    corr_pos = torch.cat((src_pcd[corr[..., 0]], tgt_pcd[corr[..., 1]]), dim=-1)

                    gt_models = inputs['rot'].cpu().numpy()
                    

                    # just for train and valid:
                    if self.config.benchmark_set_split!='test':
                        gt_models = batch_to_tsfm(inputs['rot'].cpu().numpy(),inputs['trans'].cpu().numpy())

                    if np.max(np.abs(gt_models))>1000:
                        skip.append(idx)
                        continue

                    
                    if self.config.benchmark_set_split=='test':
                        #gt_models = tsfm
                        start = time.time()
                        pred_labels = seg_model(corr_pos)#data)#(corr_pos.to(self.device))
                        succ, hs, label_list = pred_model(corr_pos[0], pred_labels[0], min_num4solve_model=10, lambda_thresh=smconfigs.lambda_thresh, device=smconfigs.device)
                        pred_time = time.time()- start
                        mean_time.append(pred_time)
                        #print(f'mean_time = {np.mean(mean_time)}')
                        #logging.info(f'mean_time = {np.mean(mean_time)}')
                        
                        
                        
                        if succ:
                            
                            gt_num = inputs['rot'].shape[0]
                            pred_num = hs.shape[0]
                            

                            mean_pred_num.append(pred_num)
                            true_ind, pred_ind, re_list, te_list, m_re, m_te = get_re_te(hs, inputs['rot'])#[:measure]
                            re = np.asarray(re_list)
                            te = np.asarray(te_list)

                            mask = ((re < 20) & (te < 0.5))
                            if mask.sum():
                                mean_re.append(np.mean(re[mask]))
                                mean_te.append(np.mean(te[mask]))
                                recall = np.sum(mask)/gt_num
                                precision = np.sum(mask)/pred_num
                                f1 = 2*precision*recall/(precision+recall+1e-12)
                                mean_hr.append(recall)
                                mean_prec.append(precision)
                                mean_f1.append(f1)
                            else:
                                mean_hr.append(0)
                                mean_prec.append(0)
                                mean_f1.append(0)
                           
                            
                        else:
                            mean_hr.append(0)
                            mean_prec.append(0)
                            mean_f1.append(0)
                            
            logging.info(f'mean_re = {np.mean(mean_re)}, mean_te={np.mean(mean_te)}, mean_hit_ratio={np.mean(mean_hr):.4f}, mean_prec={np.mean(mean_prec):.4f}, mean_f1={np.mean(mean_f1):.4f}, mean_time = {np.mean(mean_time)}')
            print(f'mean_re = {np.mean(mean_re)}, mean_te={np.mean(mean_te)}, mean_hit_ratio={np.mean(mean_hr):.4f}, mean_prec={np.mean(mean_prec):.4f}, mean_f1={np.mean(mean_f1):.4f}, mean_time = {np.mean(mean_time)}')
                       


def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized
        

def get_trainer(config):
    if(config.dataset in ['box']):
        return IndoorTester(config)
    elif(config.dataset in ['multi_multi']):
        return BiggerTester(config)
    else:
        raise NotImplementedError
