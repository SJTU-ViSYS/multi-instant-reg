import time, os, torch,copy
import numpy as np
import open3d as o3d
import torch.nn as nn
from tensorboardX import SummaryWriter
from lib.timer import Timer, AverageMeter
from lib.utils import Logger,validate_gradient

from tqdm import tqdm
import torch.nn.functional as F
import gc
from lib.utils import square_distance

import o3d_tools.visualize_tools as vt
class Trainer(object):
    def __init__(self, args):
        self.config = args
        # parameters
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose
        self.max_points = args.max_points

        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_freq = args.snapshot_freq
        self.snapshot_dir = args.snapshot_dir 
        self.benchmark = args.benchmark
        self.iter_size = args.iter_size
        self.verbose_freq= args.verbose_freq

        self.w_circle_loss = args.w_circle_loss
        self.w_overlap_loss = args.w_overlap_loss
        self.w_saliency_loss = args.w_saliency_loss 
        self.desc_loss = args.desc_loss

        self.best_loss = 1e5
        self.best_recall = -1e5
        self.best_overlap_loss = -1e5
        self.best_chamfer_dist_loss = -1e5
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        self.logger = Logger(args.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M\n')
        

        if (args.pretrain !=''):
            self._load_pretrain(args.pretrain)
        
        self.loader =dict()
        self.loader['train']=args.train_loader
        self.loader['val']=args.val_loader
        self.loader['test'] = args.test_loader

        with open(f'{args.snapshot_dir}/model','w') as f:
            f.write(str(self.model))
        f.close()
 
    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_recall': self.best_recall
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        if os.path.isfile(resume):
            state = torch.load(resume)
            self.model.load_state_dict(state['state_dict'])
            self.start_epoch = state['epoch']
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']
            self.best_recall = state['best_recall']
            
            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f'Current best recall {self.best_recall}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        stats=dict()
        stats['circle_loss']=0.
        stats['recall']=0.  # feature match recall, divided by number of ground truth pairs
        stats['saliency_loss'] = 0.
        stats['saliency_recall'] = 0.
        stats['saliency_precision'] = 0.
        stats['overlap_loss'] = 0.
        stats['overlap_recall']=0.
        stats['overlap_precision']=0.
        stats['chamfer_dist_loss']=0.
        return stats

    def stats_meter(self):
        meters=dict()
        stats=self.stats_dict()
        for key,_ in stats.items():
            meters[key]=AverageMeter()
        return meters


    def inference_one_batch(self, inputs, phase):
        assert phase in ['train','val','test']
        ##################################
        # training
        if(phase == 'train'):
            self.model.train()
            #print(self.model)
            ###############################################
            pcd = inputs['points'][0]
            #print(pcd.shape)
            len_src = inputs['stack_lengths'][0][0]
            c_rot, c_trans = inputs['rot'], inputs['trans']
            correspondence = inputs['correspondences']
            # forward pass
            #print(pcd.shape)
            feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
            #print('here')
            #print(feats)
            src_pcd, tgt_pcd = inputs['src_pcd_raw'][:, :3], inputs['tgt_pcd_raw'][:, :3]
            src_feats, tgt_feats = feats[:len_src], feats[len_src:]

            ##########################################################################################
            if correspondence.shape[0] < 10:
                o3d.visualization.draw_geometries([vt.visualize_pcd(src_pcd.cpu().numpy(), color=vt.SRC_COLOR, visual=False), vt.visualize_pcd(tgt_pcd.cpu().numpy(), color=vt.TGT_COLOR, visual=False)])
                
            # if not self.config.dataset in ['multi', 'box', 'bigger_box']:
            # #print(rot.shape, src_pcd.transpose(0,1).shape, trans.unsqueeze(1).shape)
            #     temp_src_pcd = (torch.matmul(c_rot,src_pcd.transpose(0,1))+c_trans.unsqueeze(1)).transpose(0,1)
            # else:
            #     temp_src_pcd = src_pcd
            # coords_dist = torch.sqrt(square_distance(temp_src_pcd[None,:,:], tgt_pcd[None,:,:]).squeeze(0))
            # pos_mask = coords_dist < self.config.pos_radius 
            # neg_mask = coords_dist > self.config.safe_radius 

            # ## get anchors that have both positive and negative pairs
            # row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
            # col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()
            # if torch.all(torch.logical_not(row_sel)) or torch.all(torch.logical_not(col_sel)):
            #     print(tgt_pcd)
            #     print(coords_dist.shape)
            #     print(coords_dist)
            #     print(correspondence)
            #     print(pos_mask.sum(-1))
            #     print(neg_mask.sum(-1))
            #     print(pos_mask.sum(-2))
            #     print(neg_mask.sum(-2))
                
            #     o3d.visualization.draw_geometries([vt.visualize_pcd(temp_src_pcd.cpu().numpy(), color=vt.SRC_COLOR, visual=False), vt.visualize_pcd(tgt_pcd.cpu().numpy(), color=vt.TGT_COLOR, visual=False)])
            ##########################################################################################
            # get loss
            #print(correspondence.shape)
            #print('here')
            stats= self.desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats,correspondence, c_rot, c_trans, scores_overlap, scores_saliency)
            #print('here')
            c_loss = stats['circle_loss'] * self.w_circle_loss # + stats['overlap_loss'] * self.w_overlap_loss + stats['saliency_loss'] * self.w_saliency_loss# + stats['chamfer_dist_loss']

            c_loss.backward()

        else:
            self.model.eval()
            with torch.no_grad():
                ###############################################
                # forward pass
                feats, scores_overlap, scores_saliency = self.model(inputs)  #[N1, C1], [N2, C2]
                pcd =  inputs['points'][0]
                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                correspondence = inputs['correspondences']

                src_pcd, tgt_pcd = inputs['src_pcd_raw'][:, :3], inputs['tgt_pcd_raw'][:, :3]
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]

                
                ###################################################
                # get loss
                stats= self.desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats,correspondence, c_rot, c_trans, scores_overlap, scores_saliency)
                
                if self.config.dataset in ['multi', 'box']:
                    trans_src = inputs['src_pcd_raw'].unsqueeze(0)#[N1, 3]
                    raw_tgt = inputs['tgt_pcd_raw'].unsqueeze(0)#[N2, 3]
                    feat1, feat2 = src_feats.unsqueeze(0), tgt_feats.unsqueeze(0)
                    b, n1, c = feat1.shape
                    _, n2, _ = feat2.shape
                    
                    raw_tgt, trans_src = raw_tgt.to(feat1.device), trans_src.to(feat1.device)
                    trans_src_xyz, src_labels = trans_src[..., :3], trans_src[..., 3]
                    tgt = raw_tgt[..., :3]
                    
                    feats_dist = (feat1[:, :, None, :] - feat2[:, None, :, :]).norm(dim=-1)#[b,n1,n2]
                    #print(feats_dist.shape)
                    #print(feats_dist)
                    src_tgt = feats_dist.argmin(-1).long()#[b,n1]
                    tgt_src = feats_dist.argmin(-2).long()#[b,n2]
                    
                    inlier_thresh = self.config.overlap_radius
                    
                    # wo_mutual, src
                    corr = torch.stack((torch.arange(n1, device=src_tgt.device).unsqueeze(0).expand(b, -1), src_tgt), dim=-1)#[b,n1,2]
                    
                    src_ir, inliermask = self.ir(trans_src_xyz, tgt, corr, inlier_thresh)
                    src_labels[torch.logical_not(inliermask)] = 0
                    

                    # vis_ls = []
                    # for src_labels_i in range(1,src_labels[0].max().int().item()+1):
                    #     vis_ls.append(vt.make_o3d_PointCloud(trans_src_xyz[src_labels==src_labels_i].cpu().numpy(), color=vt.COLOR_MAP_NROM[src_labels_i*5]))
                    # vis_ls.append(vt.make_o3d_PointCloud(raw_tgt[0].cpu().numpy(), color=vt.TGT_COLOR))
                    # vis_ls.append(vt.make_o3d_PointCloud(pcd[:len_src].cpu().numpy(), color=vt.SRC_COLOR))
                    # o3d.visualization.draw_geometries(vis_ls)
                    

                    message = ''
                    for i in range(src_labels[0].int().max().item()+1):
                        mask = (src_labels[0] == i)
                        message += f'{i} ratio:{mask.float().mean():.3f}\t'
                    for i in range(src_labels[0].int().max().item()+1):
                        mask = (src_labels[0] == i)
                        message += f'{i} num:{mask.sum()}\t'
                    self.logger.write(message + '\n')

                if self.config.dataset in ['bigger_box', 'ipa', 'multi_multi']:
                    trans_src = inputs['src_pcd_raw'].unsqueeze(0)#[N1, 3]
                    raw_tgt = inputs['tgt_pcd_raw'].unsqueeze(0)#[N2, 3]
                    feat1, feat2 = src_feats.unsqueeze(0), tgt_feats.unsqueeze(0)
                    b, n1, c = feat1.shape
                    _, n2, _ = feat2.shape
                    
                    tgt, trans_src_xyz = raw_tgt[..., :3].to(feat1.device), trans_src.to(feat1.device)
                    tgt_labels = raw_tgt[..., 3]
                    
                    feats_dist = (feat1[:, :, None, :] - feat2[:, None, :, :]).norm(dim=-1)#[b,n1,n2]
                    #print(feats_dist.shape)
                    #print(feats_dist)
                    src_tgt = feats_dist.argmin(-1).long()#[b,n1]
                    tgt_src = feats_dist.argmin(-2).long()#[b,n2]
                    
                    inlier_thresh = self.config.overlap_radius
                    
                    # wo_mutual, tgt
                    corr = torch.stack((tgt_src, torch.arange(n2, device=tgt_src.device).unsqueeze(0).expand(b, -1)), dim=-1)#[b,n2,2]
                    
                    tgt_ir, inliermask = self.biggerir(c_rot.shape[0],n1, trans_src_xyz, tgt, corr, inlier_thresh)
                    tgt_labels[torch.logical_not(inliermask)] = 0
                    

                    # vis_ls = []
                    # for src_labels_i in range(1,src_labels[0].max().int().item()+1):
                    #     vis_ls.append(vt.make_o3d_PointCloud(trans_src_xyz[src_labels==src_labels_i].cpu().numpy(), color=vt.COLOR_MAP_NROM[src_labels_i*5]))
                    # vis_ls.append(vt.make_o3d_PointCloud(raw_tgt[0].cpu().numpy(), color=vt.TGT_COLOR))
                    # vis_ls.append(vt.make_o3d_PointCloud(pcd[:len_src].cpu().numpy(), color=vt.SRC_COLOR))
                    # o3d.visualization.draw_geometries(vis_ls)
                    

                    message = ''
                    for i in range(tgt_labels[0].int().max().item()+1):
                        mask = (tgt_labels[0] == i)
                        message += f'{i} ratio:{mask.float().mean():.3f}\t'
                    for i in range(tgt_labels[0].int().max().item()+1):
                        mask = (tgt_labels[0] == i)
                        message += f'{i} num:{mask.sum()}\t'
                    self.logger.write(message + '\n')
        ##################################        
        # detach the gradients for loss terms
        stats['circle_loss'] = float(stats['circle_loss'].detach())
        stats['overlap_loss'] = float(stats['overlap_loss'].detach())
        stats['saliency_loss'] = float(stats['saliency_loss'].detach())
        
        return stats


    def inference_one_epoch(self,epoch, phase):
        gc.collect()
        assert phase in ['train','val','test']

        # init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()
        
        self.optimizer.zero_grad()
        skip = []
        for c_iter in tqdm(range(num_iter)): # loop through this epoch   
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            for k, v in inputs.items():  
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v.to(self.device)
            try:
                ##################################
                # forward pass
                # with torch.autograd.detect_anomaly():
                
                pcd =  inputs['points'][0]
                len_src = inputs['stack_lengths'][0][0]
                if (phase=='train' and pcd.shape[0] > 15000) or (phase=='val' and pcd.shape[0] > 10000) or (inputs['correspondences'].shape[0] < 10):
                    skip.append(c_iter)
                    continue
                stats = self.inference_one_batch(inputs, phase)
                
                ###################################################
                # run optimisation
                if((c_iter+1) % self.iter_size == 0 and phase == 'train'):
                    gradient_valid = validate_gradient(self.model)
                    if(gradient_valid):
                        self.optimizer.step()
                    else:
                        self.logger.write('gradient not valid\n')
                    self.optimizer.zero_grad()
                
                ################################
                # update to stats_meter
                for key,value in stats.items():
                    stats_meter[key].update(value)
            except Exception as inst:
                print(inst)
            
            torch.cuda.empty_cache()
            
            if (c_iter + 1) % self.verbose_freq == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)
                
                message = f'{phase} Epoch: {epoch} [{c_iter+1:4d}/{num_iter}]'
                for key,value in stats_meter.items():
                    message += f'{key}: {value.avg:.2f}\t'

                self.logger.write(message + '\n')
        print(f'skip = {skip}')
        message = f'{phase} Epoch: {epoch}'
        for key,value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        self.logger.write(message+'\n')

        return stats_meter


    def train(self):
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            #stats_meter = self.inference_one_epoch(epoch,'val')
            
            self.inference_one_epoch(epoch,'train')
            self.scheduler.step()
            
            stats_meter = self.inference_one_epoch(epoch,'val')
            
            if stats_meter['circle_loss'].avg < self.best_loss:
                self.best_loss = stats_meter['circle_loss'].avg
                self._snapshot(epoch,'best_loss')
            if stats_meter['recall'].avg > self.best_recall:
                self.best_recall = stats_meter['recall'].avg
                self._snapshot(epoch,'best_recall')
            if stats_meter['overlap_loss'].avg > self.best_overlap_loss:
                self.best_overlap_loss = stats_meter['overlap_loss'].avg
                self._snapshot(epoch,'best_overlap_loss')
            if 'chamfer_dist_loss' in stats_meter.keys() and (stats_meter['chamfer_dist_loss'].avg > self.best_chamfer_dist_loss):
                self.best_chamfer_dist_loss = stats_meter['chamfer_dist_loss'].avg
                self._snapshot(epoch,'best_chamfer_dist_loss')
            if (epoch + 1) % self.snapshot_freq:
                self._snapshot(epoch,'common')
            # we only add saliency loss when we get descent point-wise features
            if(stats_meter['recall'].avg>0.3):
                self.w_saliency_loss = 1.
            else:
                self.w_saliency_loss = 0.
                    
        # finish all epoch
        print("Training finish!")


    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0,'val')
        
        for key, value in stats_meter.items():
            print(key, value.avg)