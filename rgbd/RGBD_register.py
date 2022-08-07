'''
Date: 2021-08-11 11:09:44
LastEditors: Gilgamesh666666 fengyujianchengcom@163.com
LastEditTime: 2022-08-06 14:30:46
FilePath: /rgbdone2more/RGBD_register.py
'''
import open3d as o3d
import o3d_tools.operator_tools as ot
import o3d_tools.visualize_tools as vt
import numpy as np
import cv2
from tanimoto import Tanimoto, pred_model
import torch
import copy
from tqdm import tqdm
import time

class seg_model_configs:
    def __init__(self):
        self.wo_dist_corr_compatibility = False
        self.wo_refine = False
        self.wo_preference = False
        self.inlier_dist_thresh = 0.015
        self.dist_for_preference = self.inlier_dist_thresh
        self.refine_iter_num = 3
        self.initial_inlier_mask_thresh = -1#-1 #0, -1 = without initial outlier removal

        self.min_dist_thresh = 0.2#0.1
        self.cpat_thresh = 0.8

        self.n_clusters = 10

        self.sigma_spat = 0.4
        self.dist_sigma = 2

        self.keypts_num = 1024
        self.min_num4solve_model = 3
        self.iou_mask_thresh = 0.8
        self.device = 'cuda'
        self.num_thresh_multipler = 3

smconfigs = seg_model_configs()
seg_model = Tanimoto(smconfigs).cuda().eval()

INTRINSIC = (640, 480, 379.05242919921875, 378.6828918457031, 321.88623046875, 238.96923828125)
CORR_RADIUS = 0.0005

# for better visualiztion
for_t = {'watson':np.asarray([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        'kitkat':np.asarray([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
        'snickers':np.asarray([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
        'green_choclate':np.asarray([[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
        'pupper_instant_noodles':np.asarray([[0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        'crisp':np.asarray([[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
        'MACADAMIA':np.asarray([[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
        'MM':np.asarray([[-1, 0, 0, 0],[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
        'coo':np.asarray([[1, 0, 0, 0],[0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
        'mm_kit':np.asarray([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
        'costa':np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        'iocioc':np.asarray([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])}

# # For snickers
# move_T = np.asarray([[0,-1,0,0.5],
#                     [1,0,0,0],
#                     [0,0,1,0],#z back
#                     [0,0,0,1]])
move_T = np.asarray([[1,0,0,0.5],#x left
                    [0,1,0,0],#y up
                    [0,0,1,0],#z back
                    [0,0,0,1]])
ALL_VIS_RATIO = 1
INLIER_VIS_RATIO = 1
VISUALIZE = 1

# (object, scene)
sample_list = [
#('pupper_instant_noodles','videoscene3'),
#('MACADAMIA','videoscene3'),
#('watson','videoscene2'),('watson','videoscene3'),
#('crisp','videoscene3'),
#('milk','multiple_milk'),
('snickers','videoscene2'),
#('costa','videoscene3'),
#('iocioc','videoscene2'),
# ('kitkat','videoscene2')
]

def savetxt(filename, ls):
    with open(filename, 'w') as f:
        for item in ls:
            f.write(item + '\n')
def parser_rgbd_ls(data_dir):
    depth_files_list = np.loadtxt(f'{data_dir}/depth.txt', dtype=str)[:, 1]
    rgb_files_list = np.loadtxt(f'{data_dir}/rgb.txt', dtype=str)[:, 1]
    return rgb_files_list, depth_files_list

def rgbd2pcd(rgb, depth, intrinsic):
    # numpy [h, w, 3] [h, w]
    # K = [w, h, fx, fy, cx, cy]
    w, h, fx, fy, cx, cy = intrinsic
    K = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) # [3, 3]
    x = np.tile(np.arange(w), h) # [1, 2, 3, ..., 1, 2, 3, ..., 1, 2, 3, ...]
    y = np.repeat(np.arange(h), w) # [1, 1, 1, ..., 2, 2, 2, ..., 3, 3, 3, ...]
    
    img_xy = np.stack((x, y, np.ones_like(x)), axis=1) #[w*h,3]
    flatten_depth = depth.flatten()#[w*h]
    valid_mask = flatten_depth>0
    pcd = np.dot((img_xy*flatten_depth[:, None]), np.linalg.inv(K.T))[valid_mask]
    pcd_rgb = rgb.reshape(-1, 3)
    pcd_img_xy = img_xy[valid_mask, :2]
    return pcd, pcd_rgb[valid_mask], np.stack((pcd_img_xy[:, 1], pcd_img_xy[:, 0]), axis=1)
    
def build_rgbd(rgb_file, depth_file, intrinsic, mode='tgt'):
    # K = [width, height, fx, fy, cx, cy]
    if mode=='src':
        depth_thresh = 0.5
    else:
        depth_thresh = 0.6
    color_raw = o3d.io.read_image(rgb_file)
    depth_raw = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
    rgb = np.asarray(rgbd_image.color)
    depth = np.asarray(rgbd_image.depth)
    depth[(depth>depth_thresh)] = 0
    pcd, pcd_rgb, pcd_img_xy = rgbd2pcd(rgb, depth, intrinsic)
    pcd = vt.make_o3d_PointCloud(pcd)
    pcd.colors = o3d.utility.Vector3dVector(pcd_rgb/255)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return rgbd_image, pcd, pcd_img_xy

def match(src_kp, src_des, tgt_kp, tgt_des):
    matches = flann.knnMatch(src_des, tgt_des, k=2)
    goodMatch = []
    correspondence = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            goodMatch.append(m)
            pt1 = src_kp[m.queryIdx].pt
            pt2 = tgt_kp[m.trainIdx].pt
            correspondence.append([pt1[1], pt1[0], pt2[1], pt2[0]])
    goodMatch = np.expand_dims(goodMatch, 1)
    return goodMatch, np.stack(correspondence, axis=0).astype(np.int32)

def sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    return kp, des
def orb(img):
    orb = cv2.ORB_create()
    kp1 = orb.detect(img)
    kp, des = orb.compute(img, kp1)
    return kp, des
def surf(img):
    surf = cv2.SURF_create(400)
    kp, des = surf.detectAndCompute(img,None)
    return kp, des

def twoDcorr2threeDcorr(src_pcd_img_xy, tgt_pcd_img_xy, twoDcorrs, w, h):
    # numpy [h, w]
    src_pcd_img_idx = src_pcd_img_xy[:, 0]*w + src_pcd_img_xy[:, 1]
    tgt_pcd_img_idx = tgt_pcd_img_xy[:, 0]*w + tgt_pcd_img_xy[:, 1]
    
    src_keypts_img_idx = twoDcorrs[:, 0]*w + twoDcorrs[:, 1]
    tgt_keypts_img_idx = twoDcorrs[:, 2]*w + twoDcorrs[:, 3]

    valid_2dcorrs_mask = np.isin(src_keypts_img_idx, src_pcd_img_idx) & np.isin(tgt_keypts_img_idx, tgt_pcd_img_idx)
    
    src_keypts_img_idx = src_keypts_img_idx[valid_2dcorrs_mask]
    tgt_keypts_img_idx = tgt_keypts_img_idx[valid_2dcorrs_mask]
    
    src_keypts_pcd_idx = get_subset_idx_in_big_subset(src_keypts_img_idx, src_pcd_img_idx)
    tgt_keypts_pcd_idx = get_subset_idx_in_big_subset(tgt_keypts_img_idx, tgt_pcd_img_idx)
    return np.stack((src_keypts_pcd_idx, tgt_keypts_pcd_idx), axis=1)

def get_subset_idx_in_big_subset(subset_idx,bigsubset_idx):
    # [m,], [n,], return [m,]
    from operator import itemgetter
    subset_idx_ls = list(subset_idx)
    all_idx_in_bigsubset = np.arange(bigsubset_idx.shape[0], dtype=int)
    bigsubset_idx_dict = dict(zip(bigsubset_idx,all_idx_in_bigsubset))
   
    if len(subset_idx_ls)==1:
        return np.asarray([bigsubset_idx_dict[subset_idx_ls[0]], 0])
    elif len(subset_idx_ls)==0:
        return np.asarray([0, 0])
    else:
        return np.asarray(list(itemgetter(*subset_idx_ls)(bigsubset_idx_dict)))

def remove_plane(pcd):
    flag=0
    if isinstance(pcd, np.ndarray):
        flag=1
        pcd = ot.make_o3d_PointCloud(pcd)
    _, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=5,
                                             num_iterations=500)
    # inliers here means planes, not the object!
    pcd = pcd.select_by_index(inliers, invert=True)
    if flag:
        return np.asarray(pcd.points).astype(np.float32), inliers
    else:
        return pcd, inliers


def visual_bbox(R, center, extent_ori, color, FOR_T):
    bbox = o3d.io.read_triangle_mesh('/home/zebai/new_OverlapPredator_kl/o3d_tools/bbox.ply')
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).transform(FOR_T)
    rr = np.eye(4)
    rr[:3, :3] = R
    extent = [i/2 for i in extent_ori]
    idx = np.argmin(extent)
    extent[idx] = 0.03
    bbox.transform(np.asarray([[extent[0], 0, 0, 0], [0, extent[1], 0, 0], [0, 0, extent[2], 0], [0, 0, 0, 1]])).transform(rr).translate(center)
    FOR1.transform(rr).translate(center)

    bbox = bbox.paint_uniform_color(color)
    return bbox + FOR1


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
feat_func = sift


def main(src_data_name, tgt_data_name, FOR_T=np.eye(4)):
    src_data_dir = 'objects/' + src_data_name
    tgt_data_dir = 'scenes/' + tgt_data_name
    rgb_files_list, depth_files_list = parser_rgbd_ls(src_data_dir)
    src_rgbd_image, src_pcd, src_pcd_img_xy = build_rgbd(src_data_dir + '/' + rgb_files_list[0] , src_data_dir + '/' + depth_files_list[0], INTRINSIC, mode='src')
    
    src_image = np.asarray(src_rgbd_image.color, dtype=np.uint8)
    w, h = src_image.shape[:2]
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    src_kp, src_des = feat_func(src_image)
    src_des = src_des.astype(np.float32)
    draw_src_pcd, outliers = remove_plane(src_pcd)
    _, ind = draw_src_pcd.remove_statistical_outlier(nb_neighbors=500,std_ratio=0.1)
    draw_src_pcd = draw_src_pcd.select_by_index(ind)

    index = np.arange(len(src_pcd.points), dtype=np.int32)
    index[outliers] = -1
    mask = index>=0
    index = index[mask]
    index = index[ind]
    idxmap = np.ones(len(src_pcd.points), dtype=np.int32)*(-1)
    idxmap[index] = np.arange(len(ind), dtype=np.int32)
    src_bbox = draw_src_pcd.get_oriented_bounding_box()
    src_bbox_ = visual_bbox(src_bbox.R, src_bbox.center, src_bbox.extent, [1, 0, 0], FOR_T)
    
    draw_src_pcd_corr = copy.deepcopy(draw_src_pcd)

    draw_src_pcd_corr = draw_src_pcd_corr.transform(move_T)
    src_bbox_corr = copy.deepcopy(src_bbox_)
    src_bbox_corr = src_bbox_corr.transform(move_T)

    rgb_files_list, depth_files_list = parser_rgbd_ls(tgt_data_dir)

    hs_num = []
    for depth_file, rgb_file in tqdm(zip(depth_files_list, rgb_files_list), total=min(len(depth_files_list), len(rgb_files_list))):

        start = time.time()
        tgt_rgbd_image, tgt_pcd, tgt_pcd_img_xy = build_rgbd(tgt_data_dir + '/' + rgb_file , tgt_data_dir + '/' + depth_file, INTRINSIC)
        
        tgt_image = np.asarray(tgt_rgbd_image.color, dtype=np.uint8)
        tgt_image = cv2.cvtColor(tgt_image, cv2.COLOR_BGR2RGB)
        tgt_kp, tgt_des = feat_func(tgt_image)
        tgt_des = tgt_des.astype(np.float32)
        
        # exchange src and tgt--> src is scene and tgt is object
        src_pcd_i, src_kp_i, src_des_i, src_pcd_img_xy_i, src_pcd_draw_i, tgt_pcd_i, tgt_kp_i, tgt_des_i, tgt_pcd_img_xy_i = \
        np.asarray(tgt_pcd.points), tgt_kp, tgt_des, tgt_pcd_img_xy, tgt_pcd, np.asarray(draw_src_pcd.points), src_kp, src_des, src_pcd_img_xy
        
        # build 2d correspondence
        _, corrs = match(src_kp_i, src_des_i, tgt_kp_i, tgt_des_i)
        if corrs.shape[0]<3:
            continue
        else:
            threeDcorrs = twoDcorr2threeDcorr(src_pcd_img_xy_i, tgt_pcd_img_xy_i, corrs, w, h)
        if threeDcorrs.shape[0]<3:
            print(threeDcorrs)
            continue
        
        # 3d corrs
        threeDcorrs = twoDcorr2threeDcorr(src_pcd_img_xy_i, tgt_pcd_img_xy_i, corrs, w, h)
        threeDcorrs[:,1] = idxmap[threeDcorrs[:,1]]
        threeDcorrs = threeDcorrs[threeDcorrs[:,1]>0]
        
        #[n,6]
        corr_pos = np.concatenate((src_pcd_i[threeDcorrs[..., 0]], tgt_pcd_i[threeDcorrs[..., 1]]), axis=-1)
        
        src_pcd_i = torch.from_numpy(src_pcd_i).float().unsqueeze(0).to(smconfigs.device)
        tgt_pcd_i = torch.from_numpy(tgt_pcd_i).float().unsqueeze(0).to(smconfigs.device)
        threeDcorrs = torch.from_numpy(threeDcorrs).long().unsqueeze(0).to(smconfigs.device)
        corr_pos = torch.from_numpy(corr_pos).float().unsqueeze(0).to(smconfigs.device)

        # clustering
        pred_labels = seg_model(corr_pos)
        # calculate poses
        succ, pred_poses, label_list = pred_model(corr_pos[0], pred_labels[0], min_num4solve_model=4, device=smconfigs.device)
        
        runtime = time.time() - start
        print(f'runtime = {runtime}')
        if not succ :
            print('-----------------fail------------------------')
            print('-----------------fail------------------------')
        elif VISUALIZE:
            hs_num.append(pred_poses.shape[0])
            print(pred_poses.shape)
            
            src_pcd_i = src_pcd_i[0].detach().cpu().numpy()
            tgt_pcd_i = tgt_pcd_i[0].detach().cpu().numpy()
            label_list = label_list.cpu().numpy()
            pred_labels = pred_labels[0].cpu().numpy()
            threeDcorrs = threeDcorrs[0].cpu().numpy()
            pred_poses = pred_poses.cpu().numpy()
            #---------------------------------------------------------------------------------------------
            # RANSAC Results for comparasion
            source = ot.make_o3d_PointCloud(src_pcd_i)
            target = ot.make_o3d_PointCloud(tgt_pcd_i)
            ransac_corr = o3d.utility.Vector2iVector(threeDcorrs)
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(source, target, ransac_corr, max_correspondence_distance=0.15, ransac_n=4)
            ransac_pred_poses = copy.deepcopy(np.asarray(result.transformation))[None, :, :]
            #----------------------------------------------------------------------------------------
            print(f'{src_data_name}_{tgt_data_name}')
            for hsname, hs in zip(['pred_poses', 'ransac_pred_poses'], [pred_poses, ransac_pred_poses]):
                bboies = o3d.geometry.TriangleMesh()
                for h in hs.reshape(-1, 4, 4):
                    draw_src_pcd_i = copy.deepcopy(draw_src_pcd)
                    draw_src_pcd_i = draw_src_pcd_i.transform(np.linalg.inv(h))
                    bbox_i = copy.deepcopy(src_bbox_).transform(np.linalg.inv(h))
                    bboies += bbox_i

                # Visualize the correspondences
                corr_idx_vis = np.random.choice(threeDcorrs.shape[0], int(threeDcorrs.shape[0]*ALL_VIS_RATIO), replace=False)
                corr_mesh = vt.visualize_correspondences_official(src_pcd_draw_i, draw_src_pcd_corr, threeDcorrs, color=[1,0,0])
                pred_labels[np.logical_not(np.isin(pred_labels, label_list))] = 0

                corr_idx_vis = np.random.choice(pred_labels.shape[0], int(pred_labels.shape[0]*INLIER_VIS_RATIO), replace=False)
                pred_labels = pred_labels[corr_idx_vis]
                
                if hsname == 'pred_poses':
                    o3d.visualization.draw_geometries([draw_src_pcd_corr, src_bbox_corr, bboies, corr_mesh, src_pcd_draw_i], window_name=hsname)
                else:
                    o3d.visualization.draw_geometries([bboies, src_pcd_draw_i], window_name=hsname)
                
for src_data_name, tgt_data_name in sample_list:
    main(src_data_name, tgt_data_name, FOR_T=for_t.get(src_data_name, np.eye(4)))