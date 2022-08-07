import os, torch, time, shutil, json,glob, argparse, shutil
import numpy as np
from easydict import EasyDict as edict
from tqdm.std import tqdm

from datasets.dataloader import get_dataloader, get_datasets
from models.architectures import KPFCNN
from lib.utils import setup_seed, load_config
from lib.tester import get_trainer
from lib.loss import MetricLoss, MetricLoss_contrastive
from configs.models import architectures

from torch import optim
from torch import nn
setup_seed(0)


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    parser.add_argument('--benchmark_set_split', type=str, default='test', help= 'benchmark_set_split')

    parser.add_argument('--lambda_thresh', type=float, default=0.5, help= 'lambda_thresh')
    parser.add_argument('--min_dist_thresh', type=float, default=0.2, help= 'min_dist_thresh')
    parser.add_argument('--inlier_dist_thresh', type=float, default=0.3, help= 'inlier_dist_thresh')
    parser.add_argument('--iou_mask_thresh', type=float, default=0.8, help= 'iou_mask_thresh')
    parser.add_argument('--N_multipler', type=float, default=100, help= 'iou_mask_thresh')
    
    
    args = parser.parse_args()
    config = load_config(args.config)
    config['snapshot_dir'] = 'snapshot/%s' % config['exp_dir']
    config['tboard_dir'] = 'snapshot/%s/tensorboard' % config['exp_dir']
    config['save_dir'] = 'snapshot/%s/checkpoints' % config['exp_dir']

    config['logfile'] = config['logfile'][:-4] + f'_{args.lambda_thresh}_{args.min_dist_thresh}_{args.inlier_dist_thresh}_{args.iou_mask_thresh}_{args.N_multipler}.log'

    config = edict(config)

    config.benchmark_set_split = args.benchmark_set_split
    config.lambda_thresh = args.lambda_thresh
    config.min_dist_thresh = args.min_dist_thresh
    config.inlier_dist_thresh = args.inlier_dist_thresh
    config.iou_mask_thresh = args.iou_mask_thresh
    config.N_multipler = args.N_multipler

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    
    # backup the files
    os.system(f'cp -r models {config.snapshot_dir}')
    os.system(f'cp -r datasets {config.snapshot_dir}')
    os.system(f'cp -r lib {config.snapshot_dir}')
    shutil.copy2('main.py',config.snapshot_dir)
    
    
    # model initialization
    config.architecture = architectures[config.dataset]
    config.model = KPFCNN(config)   

    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    
    # create learning rate scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )
    
    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_datasets(config)
    config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers,
                                        )
    config.val_loader, _ = get_dataloader(dataset=val_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=neighborhood_limits
                                        )
    config.test_loader, _ = get_dataloader(dataset=benchmark_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=neighborhood_limits)
    
    # create evaluation metrics
    if config.dataset == 'split':
        config.desc_loss = MetricLoss_contrastive(config)
    else:
        config.desc_loss = MetricLoss(config) #OverlapKLCDLoss(config)#OverlapKLLoss(config) #MetricLoss(config)
    trainer = get_trainer(config)
    if(config.mode=='train'):
        trainer.train()
    elif(config.mode =='val'):
        trainer.eval()
    else:
        trainer.test()

    # config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
    #                                     batch_size=config.batch_size,
    #                                     shuffle=True,
    #                                     num_workers=1,
    #                                     )
    # for item in tqdm(config.test_loader):
    #     pass
    # for i in range(len(benchmark_set)):
    #     train_set[i]