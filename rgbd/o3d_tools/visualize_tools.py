'''
Date: 2021-04-07 11:18:37
LastEditors: Please set LastEditors
LastEditTime: 2021-11-14 15:48:48
FilePath: /D3Feat.pytorch/o3d_tools/visualize_tools.py
'''
import open3d as o3d
import numpy as np
from torch import take
import yaml
import os

# with open('o3d_tools/color.yaml', 'r') as f:
#     color = yaml.safe_load(f)

# COLOR_MAP = color['color_map']
# COLOR_MAP_NROM = color['color_map_norm']
color_num = 40#'', 40, 137
with open(f'o3d_tools/color{color_num}.yaml', 'r') as f:
    color = yaml.safe_load(f)

COLOR_MAP = list(color['color_map'].values())
COLOR_MAP_NROM = list(color['color_map_norm'].values())

def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

def get_green():
    """
    Get color green for rendering
    """
    return [0.651, 0.929, 0]
def get_orange():
    """
    Get color orange for rendering
    """
    return [1, 0.3, 0.05]
SRC_COLOR = [0, 0.651, 0.929] # blue
TGT_COLOR = [0.651, 0.929, 0] # green
GT_COLOR = [1,1,0]
SPHERE_COLOR = [0,1,0.1]
SPHERE_COLOR_2 = [0.5,1,0.1]

INLIER_COLOR = [0, 0.9, 0.1]
OUTLIER_COLOR = [1, 0.1, 0.1]
def make_o3d_PointCloud(input_nparr:np.array, color:np.array=None):
    # [n, 3]
    pcd = o3d.geometry.PointCloud()
    assert len(input_nparr.shape) == 2
    assert input_nparr.shape[1] == 3
    pcd.points = o3d.utility.Vector3dVector(input_nparr)
    if color is not None:
        #assert color.shape == (3, 1)
        pcd.paint_uniform_color(color)
    return pcd

#color = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
# color = [[250,235,215],[0,114,189],[217, 83, 25],[237, 177, 32],[126,47,142],[119,172,48],[77,190,238],[162,20,47]]
# colormap = np.asarray(COLOR_MAP_NROM)
# colormap[:8, :] = np.asarray(color)/255
def visual_multi_correspondences(source_point, target_point, corrs, labels, visual=False):
    # [n,3], [n,3], [m,2], [m]
    #corr_list = []
    corrs_tri = o3d.geometry.LineSet()
    #labels += 1
    count = 0
    for i in range(labels.max()+1):
        mask = (labels == i)
        if mask.sum():
            count += 1
            print(f'label {i} ratio: {mask.mean()}')
            if i == 0:
                color = [1,0,0]
            else:
                color = COLOR_MAP_NROM[count]#COLOR_MAP_NROM[(i*3)%26] #colormap[i]
            
            if mask.sum():
                corr = visualize_correspondences_official(source_point, target_point, corrs[mask], color)
                #corr_list.append(corr)
                corrs_tri += corr
            
    if visual:
        source_pcd = make_o3d_PointCloud(source_point)
        target_pcd = make_o3d_PointCloud(target_point)
        source_pcd.paint_uniform_color(get_blue())
        target_pcd.paint_uniform_color(get_green())
        o3d.visualization.draw_geometries([source_pcd, corrs_tri, target_pcd])
    else:
        return corrs_tri

def visual_multi_correspondences_radius(source_point, target_point, corrs, labels, radius=0.01, visual=False):
    # [n,3], [n,3], [m,2], [m]
    #corr_list = []
    corrs_tri = o3d.geometry.TriangleMesh()
    #labels += 1
    count = 0
    for i in range(labels.max()+1):
        mask = (labels == i)
        if mask.sum():
            count += 1
            print(f'label {i} ratio: {mask.mean()}')
            if i == 0:
                color = [1,0,0]
            else:
                #color = COLOR_MAP_NROM[(count*3)%40]#COLOR_MAP_NROM[(i*3)%26] #colormap[i]
                color = COLOR_MAP_NROM[count]
            if mask.sum():
                corr = visualize_correspondences(source_point, target_point, corrs[mask], color, radius=radius)
                #corr_list.append(corr)
                corrs_tri += corr
            
    if visual:
        source_pcd = make_o3d_PointCloud(source_point)
        target_pcd = make_o3d_PointCloud(target_point)
        source_pcd.paint_uniform_color(get_blue())
        target_pcd.paint_uniform_color(get_green())
        o3d.visualization.draw_geometries([source_pcd, corrs_tri, target_pcd])
    else:
        return corrs_tri

def visual_multi_correspondences_onebyone(source_point, target_point, corrs, labels):
    # [n,3], [n,3], [m,2], [m]
    source_pcd = make_o3d_PointCloud(source_point)
    target_pcd = make_o3d_PointCloud(target_point)
    source_pcd.paint_uniform_color(get_blue())
    target_pcd.paint_uniform_color(get_green())
    labels += 1
    for i in range(labels.max()+1):
        mask = (labels == i)
        print(f'label {i} ratio: {mask.mean()}')
        if i == 0:
            color = [1,0,0]
        else:
            color = COLOR_MAP_NROM[(i*3)%26]
        corr = visualize_correspondences_official(source_point, target_point, corrs[mask], color)  
        o3d.visualization.draw_geometries([source_pcd, corr, target_pcd])

def visualize_correspondences_official(source_pcd, target_pcd, inliers, color, visual=False):
    inliers_temp = []
    if isinstance(inliers, np.ndarray):
        for item in inliers:
            inliers_temp.append((item[0], item[1]))
    else:
        inliers_temp = inliers
    if isinstance(source_pcd, np.ndarray):
        source_pcd = make_o3d_PointCloud(source_pcd)
    if isinstance(target_pcd, np.ndarray):
        target_pcd = make_o3d_PointCloud(target_pcd)
    if isinstance(color, np.ndarray):
        color = list(color)
    corr = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_pcd, target_pcd, inliers_temp)
    corr.paint_uniform_color(color)
    if visual:
        o3d.visualization.draw_geometries([source_pcd, corr, target_pcd])
    else:
        return corr

def visualize_pcd(pc, color=None, visual=True):
    # [n, 3] or str
    if isinstance(pc, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        if color != None:
            pcd.paint_uniform_color(color)
    else:
        print('Input must be numpy!')
        raise ValueError
    if visual:
        o3d.visualization.draw_geometries([pcd])
    else:
        return pcd

# Visualize the detected keypts on src_pcd and tgt_pcd
def visualize_keypoint(keypts, color=[0, 0, 1], size=0.03):
    # input: numpy[n, 3]
    # output: List of open3d object (which can directly add to open3d.visualization.draw_geometries())
    box_list0 = []
    for i in range(keypts.shape[0]):
        # Which request open3d 0.9
        # For open3d 0.7: open3d.geometry.create_mesh_sphere(radius=size)
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_box.translate(keypts[i].reshape([3, 1]))
        mesh_box.paint_uniform_color(color)
        box_list0.append(mesh_box)
    return box_list0


def visualize_bbox(rectangle, color, width=0.02, offical=False):
    #[8,3]
    # rectangle = np.array([[cx - l, cy - w, cz - h],
    #                     [cx + l, cy + w, cz - h],
    #                     [cx + l, cy - w, cz - h],
    #                     [cx - l, cy + w, cz - h],
    #                     [cx - l, cy - w, cz + h],
    #                     [cx + l, cy + w, cz + h],
    #                     [cx + l, cy - w, cz + h],
    #                     [cx - l, cy + w, cz + h]])
    # [0,2],[0,3],[1,2],[1,3],[4,7],[4,6],[5,7],[5,6],[0,4],[3,7],[1,5],[2,6]
    #rectangle_new = np.stack((rectangle[:, :, 1], rectangle[:, :, 0], rectangle[:, :, 2]), axis=-1)
    #print(rectangle_new.shape)
    

    if len(rectangle.shape)==2:
        rectangle = rectangle[None, :, :]
    elif len(rectangle.shape)>3:
        rectangle = rectangle.reshape(-1, 8, 3)
    inlier_line_mesh_geoms = []
    lines = np.asarray([[0,2],[0,3],[1,2],[1,3],[4,7],[4,6],[5,7],[5,6],[0,4],[3,7],[1,5],[2,6]])
    #lines = np.asarray([[0,6],[0,5],[3,6],[3,5],[1,4],[1,7],[2,4],[2,7],[0,4],[3,7],[1,5],[2,6]])
    colors = [color for _ in range(len(lines))] 
    for rect in rectangle:
        if offical:
            line_pcd = o3d.geometry.LineSet()
            line_pcd.lines = o3d.utility.Vector2iVector(lines)
            line_pcd.colors = o3d.utility.Vector3dVector(colors)
            line_pcd.points = o3d.utility.Vector3dVector(rect)
            inlier_line_mesh_geoms.append(line_pcd)
        else:
            all_line_mesh = LineMesh(rect, lines, colors=color, radius=width)
            inlier_line_mesh_geoms.extend(all_line_mesh.cylinder_segments)
        #o3d.visualization.draw_geometries(all_line_mesh.cylinder_segments)
    return inlier_line_mesh_geoms


def visualize_correspondences(source_pcd, target_pcd, inliers, color, radius=0.001):
    """
    Helper function for visualizing the correspondences

    Just plot segments and two vertex of segments

    [n,3], [n,3], [m,]

    gt_inliers is the indices in "source_corrs_points" which mark the inliers in "source_corrs_points"
    """
    if not isinstance(source_pcd, np.ndarray):
        source = np.asarray(source_pcd.points)
    else:
        source = source_pcd
    if not isinstance(target_pcd, np.ndarray):
        target = np.asarray(target_pcd.points)
    else:
        target = target_pcd
    if isinstance(inliers, (list, tuple)):
        inliers_temp = np.asarray(inliers)
    else:
        inliers_temp = inliers
    line_mesh = LineMesh(source, target, inliers_temp, color, radius=radius)
    line_mesh_geoms = line_mesh.cylinder_segments
    
    # estimate normals
    #vis_list = [*source_all_spheres, *target_all_spheres, *source_inlier_spheres, *target_inlier_spheres]
    #vis_list.extend([all_line_mesh_geoms, inlier_line_mesh_geoms])
    return line_mesh_geoms

# def visualize_correspondences(
#     source_corrs_points, target_corrs_points, gt_inliers, translate=[-1.3,-1.5,0]
# ):
#     """
#     Helper function for visualizing the correspondences

#     Just plot segments and two vertex of segments

#     [n,3], [n,3], [m,]

#     gt_inliers is the indices in "source_corrs_points" which mark the inliers in "source_corrs_points"
#     """

#     if isinstance(gt_inliers, (list, set, tuple)):
#         gt_inliers = np.asarray(list(gt_inliers))

#     source = o3d.geometry.PointCloud()
#     source.points = o3d.utility.Vector3dVector(source_corrs_points)
#     target = o3d.geometry.PointCloud()
#     target.points = o3d.utility.Vector3dVector(target_corrs_points)
    
#     target.translate(translate)
    
#     # get inliers
#     source_inlier_points = source_corrs_points[gt_inliers, :]
#     target_inlier_points = target_corrs_points[gt_inliers, :]
    
   
#     source_inlier_spheres = visualize_keypoint(source_inlier_points, color=INLIER_COLOR, size=0.01)
#     target_inlier_spheres = visualize_keypoint(target_inlier_points, color=INLIER_COLOR, size=0.01)
    
#     source_all_spheres = visualize_keypoint(source_corrs_points, color=OUTLIER_COLOR, size=0.01)
#     target_all_spheres = visualize_keypoint(target_corrs_points, color=OUTLIER_COLOR, size=0.01)

#     inlier_line_mesh = LineMesh(source_inlier_points, target_inlier_points, None, INLIER_COLOR, radius=0.012)
#     inlier_line_mesh_geoms = inlier_line_mesh.cylinder_segments

#     all_line_mesh = LineMesh(source_corrs_points, target_corrs_points, None, OUTLIER_COLOR, radius=0.001)
#     all_line_mesh_geoms = all_line_mesh.cylinder_segments
    
#     # estimate normals
#     #vis_list = [*source_all_spheres, *target_all_spheres, *source_inlier_spheres, *target_inlier_spheres]
#     #vis_list.extend([all_line_mesh_geoms, inlier_line_mesh_geoms])
#     return all_line_mesh_geoms + inlier_line_mesh_geoms


# Credit to JeremyBYU in this Open3D issue: https://github.com/intel-isl/Open3D/pull/738
# Modified to fit the latest version of Open3D

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    if np.linalg.norm(axis_):
        axis_ = axis_ / np.linalg.norm(axis_)
    else:
        axis_ = None
    angle = np.arccos(np.dot(a, b))
    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, srcpcd, tgtpcd, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.srcpcd = srcpcd
        self.tgtpcd = tgtpcd
        self.lines = lines

        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments =  o3d.geometry.TriangleMesh()

        self.create_line_mesh()

    def create_line_mesh(self):
        #print(self.lines, self.srcpcd)
        first_points = self.srcpcd[self.lines[:, 0], :]
        second_points = self.tgtpcd[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                        R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                        center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)
            #o3d.visualization.draw_geometries([cylinder_segment])
            self.cylinder_segments += cylinder_segment
            #o3d.visualization.draw_geometries(self.cylinder_segments)

    # def add_line(self, vis):
    #     """Adds this line to the visualizer"""
    #     for cylinder in self.cylinder_segments:
    #         vis.add_geometry(cylinder)

    # def remove_line(self, vis):
    #     """Removes this line from the visualizer"""
    #     for cylinder in self.cylinder_segments:
    #         vis.remove_geometry(cylinder)
