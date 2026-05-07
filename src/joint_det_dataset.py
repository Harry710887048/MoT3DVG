# ==================== Imports ====================
from cgi import print_arguments
from collections import defaultdict
import json
import multiprocessing as mp
import os
import shutil
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
# import open3d as o3d
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenes

from utils.transform_waymo import transform_to_front_view
# from utils.pcds_in_bbox import get_points_in_bbox
from ops.teed_pointnet.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from utils.box_util import extract_points_in_bbox_3d, project_points_to_2d, draw_points_on_image, conver_box2d, draw_projected_box3d
from PIL import Image
import pickle
import ipdb
# ==================== Constants ====================
MAX_NUM_OBJ = 132

CLASS_MAPPINGS = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "trailer": 3,
    "construction_vehicle": 4,
    "pedestrian": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic_cone": 8,
    "barrier": 9,
}

# Waymo dataset synonyms
MOT3DVG_SYNONYMS = {
    "car": ["car", "vehicle", "sedan", "van", "coupe", "automobile", "convertible", "hatchback", "SUV", 
            "pickup truck", "pickup", "minivan", "taxi", "cab", "utility truck", "delivery truck",
            "CR-V", "Corolla", "Tundra", "Camry", "Civic", "Accord"],
    "truck": ["truck", "lorry", "freight", "cargo truck", "semi-truck", "flatbed truck", 
              "concrete mixer truck", "mixer truck", "cement truck"],
    "bus": ["bus", "coach", "minibus", "shuttle", "school bus", "public transport"],
    "othervehicle": ["vehicle", "jeep", "tractor", "trailer", "machinery"],
    "pedestrian": ["pedestrian", "person", "man", "woman", "people", "child", "boy", "girl", "adult", 
                   "passerby", "walker", "worker", "individual", "guy", "lady"],
    "cyclist": ["cyclist", "biker", "bike rider", "rider", "bicycle", "bike", "person riding"],
    "trailer": ["trailer", "cargo trailer", "vehicle trailer"],
    "construction_vehicle": ["construction vehicle", "bulldozer", "excavator", "dump truck"],
    "motorcycle": ["motorcycle", "bike", "motorbike", "scooter"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "traffic_cone": ["traffic cone", "cone", "road cone"],
    "barrier": ["barrier", "road barrier", "divider", "bollard"],
}

def read_file(file_path):
    if file_path.endswith('.pkl'):
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.endswith('.json'):
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.endswith('.npy'):
        import numpy as np
        try:
            data = np.load(file_path, allow_pickle=True).item()
        except:
            data = np.load(file_path, allow_pickle=True)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            data = f.readlines()
    elif file_path.endswith('.yaml'):
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported file format")
    return data

def filter_by_dist(points, max_dist=50.0):
    dists = np.linalg.norm(points[:, :3], axis=1)
    mask = dists < max_dist
    return points[mask]

def get_curr_scene(scene_splits, index):        
    mask = scene_splits > index
    mask_ = ~mask
    # mask_ = self.scene_splits <= index
    curr_scene_start_id = np.where(mask_)[0][-1]

    return curr_scene_start_id

def points_rigid_transform(cloud, pose):
    cloud = np.array(cloud)
    if cloud.shape[0] == 0:
        return cloud
    mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
    pose_mat = np.mat(pose)
    mat[:, 0:3] = cloud[:, 0:3]
    mat = np.mat(mat)
    transformed_mat = pose_mat * mat.T
    T = np.array(transformed_mat.T, dtype=np.float32)
    return T[:, 0:3]

def cat_multiframe( 
                   index,
                   nusc_root,
                   nusc_infos,
                   window_size=6,
                   window_step=1,
                   mode="train",
                   ):
    if mode == "train":
        scene_splits = read_file('group_same_scene2_train.npy')
        scene_infos = read_file('scene_info2_train.npy')
    elif mode == "val":
        scene_splits = read_file('group_same_scene2_val.npy')
        scene_infos = read_file('scene_info2_val.npy')
    
    scene_id = get_curr_scene(scene_splits, index)
    scene_begin = scene_splits[scene_id]
    scene_end = scene_splits[scene_id+1]
    
    this_frame_id = index - scene_begin
    all_points = []
    
    curr_pcd_path = str(nusc_infos[index]['lidar_path']).replace("/LIDAR_TOP", "/LIDAR_TOP_SEG").replace(".pcd.bin", "_seg.npy")
    lidar_path = os.path.join(nusc_root, curr_pcd_path)
    current_frame_points = np.load(lidar_path, allow_pickle=True).reshape(-1, 7)
    
    if mode == "train":
        i_pose = scene_infos[scene_id]['scenes'][this_frame_id]['transform_matrix']
    elif mode == "val":
        i_pose = scene_infos[scene_id][this_frame_id]
        
    if i_pose is None:
        i_pose = np.eye(4)
    i_pose_inv = np.linalg.inv(i_pose)
    
    for j in range(index-window_size, index, window_step):
        if j < scene_begin or j >= scene_end:
            continue
            
        lidar_path_prex = str(nusc_infos[j]['lidar_path']).replace("/LIDAR_TOP", "/LIDAR_TOP_SEG").replace(".pcd.bin", "_seg.npy")
        lidar_path = os.path.join(nusc_root, lidar_path_prex)
        lidar_points = np.load(lidar_path, allow_pickle=True).reshape(-1, 7)
        
        xyz = lidar_points[:, :3] 
        seg_map = lidar_points[:, 3:7]
        
        time_offset = j - index
        time_dim = np.full((xyz.shape[0], 1), time_offset, dtype=np.float32)

        try:
            pos_i_id = j - scene_begin
            if mode == "train":            
                curr_transmat = scene_infos[scene_id]['scenes'][pos_i_id]['transform_matrix']
            elif mode == "val":
                curr_transmat = scene_infos[scene_id][pos_i_id]
            
            if curr_transmat is not None:
                xyz = points_rigid_transform(xyz, curr_transmat)
                xyz = points_rigid_transform(xyz, i_pose_inv)

            lidar_points_with_time = np.concatenate([xyz, seg_map, time_dim], axis=1)
            all_points.append(lidar_points_with_time)
            
        except Exception as e:
            print(f"Warning: frame {j} error: {e}")
            continue

    if len(all_points) > 0:
        historical_points = np.concatenate(all_points)
    else:
        historical_points = np.empty((0, 8)) 

    return current_frame_points, historical_points


# ==================== Dataset Class ====================
class Joint3DDataset(Dataset):
    """Dataset utilities for ReferIt3D."""

    def __init__(
        self,
        dataset_dict={"nuscenes": 1},
        test_dataset={"nuscenes": 1},
        split="train",
        overfit=False,
        data_path="./",
        split_dir="data/splits",
        use_color=False,
        use_height=False,
        use_multiview=False,
        detect_intermediate=False,
        butd=False,
        butd_gt=False,
        butd_cls=False,
        augment_det=False,
        debug=False,
    ):
        """Initialize dataset (here for ReferIt3D utterances)."""
        # Basic configuration
        self.debug = debug
        self.dataset_dict = dataset_dict
        self.test_dataset = test_dataset
        self.split = split
        self.use_color = use_color
        self.use_height = use_height
        self.overfit = overfit
        self.detect_intermediate = detect_intermediate
        self.augment = self.split == "train"
        self.use_multiview = use_multiview
        self.data_path = data_path
        self.split_dir = split_dir
        self.visualize = False
        # self.visualize = True
        if self.visualize:
            self.vis_save_dir = "./visualization/"
            os.makedirs(self.vis_save_dir, exist_ok=True)

        self.butd = butd
        self.butd_gt = butd_gt
        self.butd_cls = butd_cls
        self.joint_det = "scannet" in dataset_dict and len(dataset_dict.keys()) > 1 and self.split == "train"  # false
        self.augment_det = augment_det

        # Initialize tokenizer and other utilities
        self.mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
        self.tokenizer = RobertaTokenizerFast.from_pretrained("./data/roberta_base/")
        
        self.nusc = NuScenes(version='v1.0-trainval', 
                             dataroot='data/nuscenes/v1.0-trainval', verbose=True)

        # Load classification results if available
        # if os.path.exists("data/cls_results.json"):
        #     with open("data/cls_results.json") as fid:
        #         self.cls_results = json.load(fid) 

        # Load annotations
        self.annos = []

        if self.split == "train":
            for dset in dataset_dict.keys():
                _annos = self.load_annos(dset)
                self.annos += _annos
        else:
            for dset in test_dataset.keys():
                _annos = self.load_annos(dset)
                self.annos += _annos

    def _format_caption(self, utterance):
        """Format caption by adding spaces and handling commas."""
        return " " + " ".join(utterance.replace(",", " ,").split()) + " "

    def _get_frame_paths(self, frame_path, dataset):
        """Get all relevant file paths for a frame."""
        return {
            "image": os.path.join(frame_path, "image.jpg"),
            "lidar": os.path.join(frame_path, "lidar.npy" if dataset == "waymo" else "lidar.bin"),
            "meta": os.path.join(frame_path, "meta_info.json"),
        }
        
    def _process_utterance(self, utterance, dataset):
        """Process utterance based on dataset type."""
        if dataset == "waymo":
            return utterance
        elif dataset in ["drone", "quad"]:
            return utterance.split("Summary:")[-1].strip()
        return utterance

    def _get_synonyms_dict(self, dataset):
        """Get appropriate synonyms dictionary based on dataset."""
        return MOT3DVG_SYNONYMS

    # ==================== Data Loading Methods ====================
    def load_annos(self, dset):
        """Load annotations of given dataset."""
        # ipdb.set_trace()
        loaders = {
            "nuscenes": lambda: self.load_mot3dvg_annos(dataset="nuscenes"),
        }
        annos = loaders[dset]()
        if self.overfit:
            annos = annos[:128]
        return annos

    def load_mot3dvg_annos(self, dataset="nuscenes"):
        """Load mot3dvg annotations for nuScenes."""
        import pickle as pkl
        from nuscenes.nuscenes import NuScenes
        
        split = "train" if self.split == "train" else "val"
        
        data_path = os.path.join(self.data_path, dataset)
        assert os.path.exists(data_path), f"data path not found: {data_path}"
        
        synonyms_dict = MOT3DVG_SYNONYMS
        frames_names = []
        annos = []
        class_set = set()
        
        split_file = os.path.join(data_path, "v1.0-trainval", f"mot3dvg_{split}.pkl")
        assert os.path.exists(split_file), f"split file not found: {split_file}"
        
        with open(split_file, "rb") as f:
            sequence_list = pkl.load(f)

        self.nusc_infos = sequence_list
        
        for idx, sequence in enumerate(sequence_list):
            lidar_path = sequence["lidar_path"]
            image_path_front = sequence["cam_front_path"]
            frame_id = idx
            
            sweeps = sequence["sweeps"]
            sample = self.nusc.get('sample', sequence['token'])['data']
            
            try:
                class_set.add(sequence["referent_gt_names"].lower())
            except:
                continue
            
            utterance = sequence["referent_prompt"]
            cat_names = sequence["referent_gt_names"]

            caption = self._format_caption(utterance)
            candidate_words = synonyms_dict.get(cat_names, [cat_names])
            
            positions = []
            for word in candidate_words:
                matches = list(re.finditer(rf"\b{re.escape(word)}\b", caption, flags=re.IGNORECASE))
                positions.extend([(match.start(), match.end(), word) for match in matches])
            
            if len(positions) > 0:
                tokens_positive = torch.tensor([positions[0][0], positions[0][1]], dtype=torch.long)
                matched_cls = positions[0][2]  # e.g. van
            else:
                continue
            
            tokenized = self.tokenizer.batch_encode_plus([self._format_caption(utterance)], padding="longest", return_tensors="pt")
            gt_map = get_positive_map(tokenized, [tokens_positive])
            
            bbox_3d = sequence["referent_box"].tolist()
            annos.append(
                {
                    "scan_id": lidar_path,
                    "target_id": CLASS_MAPPINGS[sequence["referent_gt_name"].lower()],
                    "frame_id": frame_id,
                    "sweeps": sweeps,
                    "sample": sample,
                    "caption": caption,
                    "utterance": utterance,
                    "pred_pos_map": gt_map,
                    "meta_path": lidar_path,  # grounding_evaluator.py
                    "dataset": dataset,
                    "pcd_path": os.path.join('./data/nuscenes/v1.0-trainval', lidar_path),
                    "tokens_positive": tokens_positive,
                    "gt_map": gt_map,
                    "bbox_3d": bbox_3d,
                    "image_path": image_path_front,
                    "gt_bbox": bbox_3d,
            }
        )

        print(f"Loaded {len(annos)} annotations from {split}.")
        return annos
    
    def _get_nusc_pcd(self, anno, max_sweeps=7, dist_filter=True):
        """Process point cloud data."""
        pcd_path = anno["pcd_path"]

        curr_pcd, pcd_sweep = cat_multiframe(
            index=anno["frame_id"], 
            nusc_root='data/nuscenes/v1.0-trainval', 
            nusc_infos=self.nusc_infos, mode=self.split)        
        
        if dist_filter:
            curr_pcd = filter_by_dist(curr_pcd, max_dist=50.0)
            pcd_sweep = filter_by_dist(pcd_sweep, max_dist=50.0)
        
        num_points = 16384
        curr_pcd = self.sample(curr_pcd, num_points)
        
        xyz = curr_pcd[:, 0:3] 
        cls_prob = curr_pcd[:, 3:7]
        point_cloud_ = np.concatenate([xyz, cls_prob], axis=1)

        point_cloud_wo_sample = pcd_sweep 

        return point_cloud_, point_cloud_wo_sample

    def _get_token_positive_map(self, anno):
        """Return correspondence of boxes to tokens."""
        # Token start-end span in characters
        caption = self._format_caption(anno["utterance"])
        tokens_positive = np.zeros((MAX_NUM_OBJ, 2))
        if isinstance(anno["target"], list):
            cat_names = anno["target"]
        else:
            cat_names = [anno["target"]]
        if self.detect_intermediate:
            cat_names += anno["anchors"]
        for c, cat_name in enumerate(cat_names):
            start_span = caption.find(" " + cat_name + " ")
            len_ = len(cat_name)
            if start_span < 0:
                start_span = caption.find(" " + cat_name)
                len_ = len(caption[start_span + 1 :].split()[0])
            if start_span < 0:
                start_span = caption.find(cat_name)
                orig_start_span = start_span
                while caption[start_span - 1] != " ":
                    start_span -= 1
                len_ = len(cat_name) + orig_start_span - start_span
                while caption[len_ + start_span] != " ":
                    len_ += 1
            end_span = start_span + len_
            assert start_span > -1, caption
            assert end_span > 0, caption
            tokens_positive[c][0] = start_span
            tokens_positive[c][1] = end_span

        # Positive map (for soft token prediction)
        tokenized = self.tokenizer.batch_encode_plus([self._format_caption(anno["utterance"])], padding="longest", return_tensors="pt")
        positive_map = np.zeros((MAX_NUM_OBJ, 256))
        gt_map = get_positive_map(tokenized, tokens_positive[: len(cat_names)])
        positive_map[: len(cat_names)] = gt_map
        return tokens_positive, positive_map

    def _get_3eed_target_boxes(self, anno, xyz):
        """Return gt boxes to detect."""

        tids = [anno["target_id"]]  

        # Generate instance label, default -1 (unmarked), if 3D point belongs to a target object, fill in target object ID
        xyz = xyz[:, :3]
        point_instance_label = -np.ones(len(xyz))
        # Find points inside bbox and mark as 0
        
        # Generate axis_align_bbox for 3D object
        bbox = np.array(anno["gt_bbox"])#.reshape(-1)
        
        assert len(bbox) == 9

        # Use adjusted bbox to extract points (now xyz and bbox height match)
        points_in_bbox, point_mask = extract_points_in_bbox_3d(
            xyz, bbox[:7]
        )  
        point_instance_label[point_mask] = 0
        
        # # NOTE Visual sanity check
        if self.visualize:
            print(f"    Found {len(points_in_bbox)} points inside 3D bbox")
            img_path = anno["image_path"]
            image = Image.open(img_path)
            
            # Project point cloud onto 2D image
            points_2d, depth, valid_mask = project_points_to_2d(
                points_in_bbox[:, :3], image.size, anno
            )
            proj_pcd_image, mask = draw_points_on_image(
                image, points_2d, color=(0, 255, 0), radius=3, create_mask=True
            )
            save_path = os.path.join(
                './vis_debug', anno['scan_id'], f"proj_pcd.jpg"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, proj_pcd_image)
            print(f"    Projected point cloud image saved: {save_path}") 
            
            ipdb.set_trace()
            
            # Project 3D bbox onto 2D image
            bbox_3d_corners, _ = conver_box2d(bbox, image.size, anno)
            contour_color = (0, 0, 255) 
            img_with_3d_bbox = draw_projected_box3d(
                image,
                bbox_3d_corners[0],
                color=contour_color,
                thickness=2)
            # save image
            save_path = os.path.join(
                './vis_debug', anno['scan_id'], f"proj_bbox.jpg"
            )
            cv2.imwrite(save_path, img_with_3d_bbox)
            print(f"    3D bbox image saved: {save_path}")

        bbox = bbox.reshape(-1)

        # Generate axis_align_bbox for 3D object
        bboxes = np.zeros((MAX_NUM_OBJ, 7))
        bboxes[: len(tids)] = bbox[:7]  # shape: (N, 6) # The first N are real bboxes

        box_label_mask = np.zeros(MAX_NUM_OBJ)
        box_label_mask[: len(tids)] = 1  # Mark first N as valid

        return bboxes, box_label_mask, point_instance_label

    def _get_waymo_multi_target_boxes(self, anno, xyz):
        """Return gt boxes to detect."""
        boxes_info = anno["boxes_info"]
        tids = boxes_info["class_id"]
        gt_bbox = np.stack(boxes_info["bbox3d"], axis=0).astype(np.float32)  # shape: (N, 7)
        xyz = xyz[:, :3]
        point_instance_label = -np.ones(len(xyz))  # Points inside bbox are marked 0

        point_indices = points_in_boxes_cpu(torch.from_numpy(xyz), torch.from_numpy(gt_bbox)).numpy()
        for i in range(gt_bbox.shape[0]):
            fg_mask = point_indices[i] > 0
            # point_instance_label[fg_mask] = i
            point_instance_label[fg_mask] = 0

        bboxes = np.zeros((MAX_NUM_OBJ, 7))
        bboxes[: len(tids)] = gt_bbox[:, :7]  # shape: (N, 6)

        bboxes[len(tids) :, :3] = 1000  # Pad non-target bboxes; first len(tids) are real targets
        box_label_mask = np.zeros(MAX_NUM_OBJ)
        box_label_mask[: len(tids)] = 1  # Mark which bboxes are valid

        return bboxes, box_label_mask, point_instance_label

    # ==================== Data Augmentation Methods ====================
    def _augment(self, pc, color, rotate):
        """Apply data augmentation to point cloud."""
        augmentations = {}

        # Rotate/flip only if we don't have a view_dep sentence
        if rotate:
            theta_z = 90 * np.random.randint(0, 4) + 10 * np.random.rand() - 5
            # Flipping along the YZ plane
            augmentations["yz_flip"] = np.random.random() > 0.5
            if augmentations["yz_flip"]:
                pc[:, 0] = -pc[:, 0]
            # Flipping along the XZ plane
            augmentations["xz_flip"] = np.random.random() > 0.5
            if augmentations["xz_flip"]:
                pc[:, 1] = -pc[:, 1]
        else:
            theta_z = (2 * np.random.rand() - 1) * 5
        augmentations["theta_z"] = theta_z
        pc[:, :3] = rot_z(pc[:, :3], theta_z)
        # Rotate around x
        theta_x = (2 * np.random.rand() - 1) * 2.5
        augmentations["theta_x"] = theta_x
        pc[:, :3] = rot_x(pc[:, :3], theta_x)
        # Rotate around y
        theta_y = (2 * np.random.rand() - 1) * 2.5
        augmentations["theta_y"] = theta_y
        pc[:, :3] = rot_y(pc[:, :3], theta_y)

        # Add noise
        noise = np.random.rand(len(pc), 3) * 5e-3
        augmentations["noise"] = noise
        pc[:, :3] = pc[:, :3] + noise

        # Translate/shift
        augmentations["shift"] = np.random.random((3,))[None, :] - 0.5
        pc[:, :3] += augmentations["shift"]

        # Scale
        augmentations["scale"] = 0.98 + 0.04 * np.random.random()
        pc[:, :3] *= augmentations["scale"]

        # Color
        if color is not None:
            color += self.mean_rgb
            color *= 0.98 + 0.04 * np.random.random((len(color), 3))
            color -= self.mean_rgb
        return pc, color, augmentations

    def aug_points(
        self,
        xyz: np.array,
        if_flip: bool = False,
        if_scale: bool = False,
        scale_axis: str = "xyz",
        scale_range: list = [0.9, 1.1],
        if_jitter: bool = False,
        if_rotate: bool = False,
        if_tta: bool = False,
        num_vote: int = 0,
    ):
        """Apply various augmentations to points."""
        # aug (random rotate)
        if if_rotate:
            if if_tta:
                angle_vec = [0, 1, -1, 2, -2, 6, -6, 7, -7, 8]
                assert len(angle_vec) == 10
                angle_vec_new = [cnt * np.pi / 8.0 for cnt in angle_vec]
                theta = angle_vec_new[num_vote]
            else:
                theta = np.random.uniform(0, 2 * np.pi)
            rot_mat = np.array(
                [
                    [np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            xyz = np.dot(xyz, rot_mat)

        # aug (random scale)
        if if_scale:
            # scale_range = [0.95, 1.05]
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            xyz = xyz * scale_factor

        # aug (random flip)
        if if_flip:
            if if_tta:
                flip_type = num_vote
            else:
                flip_type = np.random.choice(4, 1)

            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        # aug (random jitter)
        if if_jitter:
            noise_translate = np.array(
                [
                    np.random.normal(0, 0.1, 1),
                    np.random.normal(0, 0.1, 1),
                    np.random.normal(0, 0.1, 1),
                ]
            ).T
            xyz += noise_translate

        return xyz

    # ==================== Dataset Interface Methods ====================
    def sample(self, pc, num_points):
        N = pc.shape[0]
        # if N == 0:
        #     return np.zeros((num_points, pc.shape[1]), dtype=pc.dtype)
        
        if N >= num_points:
            idx = np.random.choice(N, num_points, replace=False)
        else:
            idx = np.random.choice(N, num_points, replace=True)
        return pc[idx]
    
    def __getitem__(self, index):
        """Get current batch for input index."""

        # Read annotation
        anno = self.annos[index]

        if anno["dataset"] == "waymo-multi":
            return self.getitem_waymo_multi(index)

        if self.debug:
            index = 0

        self.random_utt = False

        # Point cloud representation
        # max_sweep = 5 if self.split == "train" else 1
        max_sweep = 7
        point_cloud, point_cloud_sweep_wo_sample = self._get_nusc_pcd(anno, max_sweeps=max_sweep)
        
        gt_bboxes, box_label_mask, point_instance_label = self._get_3eed_target_boxes(anno, point_cloud)
        
        neighbor_ids = point_cloud_sweep_wo_sample[:, -1]
        clear_ids = np.unique(neighbor_ids)
        num_neighbors = len(np.unique(neighbor_ids))
        
        pcds_list = []
        max_neighbors = 4
        for nbor in range(min(num_neighbors, max_neighbors)):
            if nbor < len(clear_ids):
                curr_mask = point_cloud_sweep_wo_sample[:, -1] == clear_ids[nbor]
                pcd = point_cloud_sweep_wo_sample[curr_mask][:, :7]
                
                if pcd.shape[0] > 0:
                    pcd_sampled = self.sample(pcd, num_points=16384)
                else:
                    pcd_sampled = point_cloud.copy()
            else:
                pcd_sampled = point_cloud.copy()
            
            pcds_list.append(pcd_sampled)

        while len(pcds_list) < max_neighbors:
            pcds_list.append(point_cloud.copy())
        
        pcds_list = pcds_list[:max_neighbors]
        pcds_list = np.asarray(pcds_list)  # (max_neighbors, 16384, 3)

        if anno["dataset"] == "waymo":
            lidar_id = int(anno["scan_id"].split("_")[-1])
            xyz = point_cloud[:, :3] # (n_p, 3)
            WAYMO_VIEWS = ["F", "FL", "FR", "SL", "SR"]
            xyz, target_box = transform_to_front_view(xyz, gt_bboxes[0][None, :], WAYMO_VIEWS[lidar_id])
            point_cloud[:, :3] = xyz
            gt_bboxes[0] = target_box[0]

        positive_map = np.zeros((MAX_NUM_OBJ, 256))  #  1, 256
        positive_map_ = np.array(anno["pred_pos_map"]).reshape(-1, 256)
        positive_map[: len(positive_map_)] = positive_map_

        # Return
        _labels = np.zeros(MAX_NUM_OBJ)  # 132
        
        ret_dict = {
            "box_label_mask": box_label_mask.astype(np.float32),  # NOTE Used in loss calculation
            "center_label": gt_bboxes[:, :3].astype(np.float32),  # xyz
            "sem_cls_label": _labels.astype(np.int64),  # NOTE Used in loss calculation
            "size_gts": gt_bboxes[:, 3:6].astype(np.float32),  # NOTE w h d
            "gt_bboxes": gt_bboxes.astype(np.float32), # NOTE shape: (N, 9) 
            "meta_path": anno['meta_path'], 
            "point_clouds": point_cloud.astype(np.float32),
            "pcds_list": pcds_list.astype(np.float32),  # (n_neabors, 16384, 3)
            "utterances": (" ".join(anno["utterance"].replace(",", " ,").split()) + " . not mentioned"),
            "positive_map": positive_map.astype(np.float32),
            "point_instance_label": point_instance_label.astype(np.int64),  # NOTE Used in loss calculation
            "is_view_dep": self._is_view_dep(anno["utterance"]),
            "is_hard": False,
            "is_unique": False,
        }
        

        return ret_dict

    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)

    @staticmethod
    def _is_view_dep(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = ["front", "behind", "back", "left", "right", "facing", "leftmost", "rightmost", "looking", "across"]
        words = set(utterance.split())
        return any(rel in words for rel in rels)


# ==================== Utility Functions ====================
def get_positive_map(tokenized, tokens_positive):
    """Construct a map of box-token associations."""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos : end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()


def rot_x(pc, theta):
    """Rotate along x-axis."""
    theta = theta * np.pi / 180
    return np.matmul(np.array([[1.0, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]), pc.T).T


def rot_y(pc, theta):
    """Rotate along y-axis."""
    theta = theta * np.pi / 180
    return np.matmul(np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1.0, 0], [-np.sin(theta), 0, np.cos(theta)]]), pc.T).T


def rot_z(pc, theta):
    """Rotate along z-axis."""
    theta = theta * np.pi / 180
    return np.matmul(np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]]), pc.T).T


def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack(
        (
            np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
            np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
            np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
            np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
            np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
            np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
            np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
            np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1),
        ),
        axis=1,
    )
