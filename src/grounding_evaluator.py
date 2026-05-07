import torch

from utils.eval_det import iou3d_rotated_vs_aligned
import utils.misc as misc
from collections import defaultdict
import ipdb

st = ipdb.set_trace


class GroundingEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): Dist. thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(self, only_root=False, thresholds=[1.0, 0.5], topks=[1, 5, 10], prefixes=[]):
        """Initialize accumulators."""
        self.only_root = only_root
        self.thresholds = [1.0, 0.5]
        self.topks = topks
        self.prefixes = prefixes

        self.reset()

    def reset(self):
        """Reset accumulators to empty."""
        self.dets = defaultdict(int)
        self.gts = defaultdict(int)

        self.dets.update({"vd": 0, "vid": 0})
        self.dets.update({"hard": 0, "easy": 0})
        self.dets.update({"multi": 0, "unique": 0})
        self.gts.update({"vd": 1e-14, "vid": 1e-14})
        self.gts.update({"hard": 1e-14, "easy": 1e-14})
        self.gts.update({"multi": 1e-14, "unique": 1e-14})

        self.dets.update({("total_acc", t, "bbf"): 0 for t in self.thresholds})
        self.gts.update({("total_acc", t, "bbf"): 1e-14 for t in self.thresholds})

        self.prediction_records = []

    def print_stats(self):
        """Print accumulated accuracies."""
        return_str = "\n"
        mode_str = {"bbs": "Box given span (soft-token)", "bbf": "Box given span (contrastive)"}
        for prefix in ["last_", "proposal_"]:  
            for mode in ["bbs", "bbf"]:
                for t in self.thresholds:
                    line = f"{prefix} {mode_str[mode]} Dist{t:.2f}: " + ", ".join(
                        [f"Top-{k}: {self.dets[(prefix, t, k, mode)] / max(self.gts[(prefix, t, k, mode)], 1):.3f}" for k in self.topks]
                    )
                    # print(line)
                    return_str += line + "\n"

        return_str += "\n==Analysis==\n"

        for t in self.thresholds:
            acc = self.dets[("total_acc", t, "bbf")] / self.gts[("total_acc", t, "bbf")]
            return_str += f"Acc@{t} = {acc:.4f}  "

        return_str += "\n\n"

        return return_str

    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)

        if misc.is_main_process():
            merged_predictions = {}
            for key in all_dets[0].keys():
                merged_predictions[key] = 0
                for p in all_dets:
                    if isinstance(p[key], torch.Tensor):
                        p[key] = p[key].cpu()
                    merged_predictions[key] += p[key]
            self.dets = merged_predictions

            merged_predictions = {}
            for key in all_gts[0].keys():
                merged_predictions[key] = 0
                for p in all_gts:
                    if isinstance(p[key], torch.Tensor):
                        p[key] = p[key].cpu()
                    merged_predictions[key] += p[key]
            self.gts = merged_predictions

    def evaluate(self, batch_data, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            batch_data (dict): contains original data (utterances, meta_path, etc.)
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        self.evaluate_bbox_by_span(batch_data, end_points, prefix)
        self.evaluate_bbox_by_contrast(batch_data, end_points, prefix)

    def evaluate_bbox_by_span(self, batch_data, end_points, prefix):
        """
        Evaluate bounding box IoU for top gt span detections.

        Args:
            batch_data (dict): contains original data (utterances, meta_path, etc.)
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        positive_map = torch.clone(end_points["positive_map"])
        positive_map[positive_map > 0] = 1
        gt_bboxes_rotated = batch_data["gt_bboxes"]
        
        sem_scores = end_points[f"{prefix}sem_cls_scores"].softmax(-1)  # B, num_query=256, len_token=256

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(sem_scores.shape[0], sem_scores.shape[1], positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, : sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        pred_center = end_points[f"{prefix}center"]  # B, Q, 3
        pred_size = end_points[f"{prefix}pred_size"]  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)  # B, Q=256, 6, each query corresponds to a box

        for bid in range(len(positive_map)):
            num_obj = int(end_points["box_label_mask"][bid].sum())  # 1
            pmap = positive_map[bid, :num_obj]
            scores = (sem_scores[bid].unsqueeze(0) * pmap.unsqueeze(1)).sum(-1)  # (1, Q, 256)  # (obj, 1, 256)  # (obj, Q) # Score of each query for target token

            top = scores.argsort(1, True)[:, :10]  # (obj, 10) # Sort each GT (only 1 here) and get top 10 queries
            pbox = pred_bbox[bid, top.reshape(-1)]  #  # Query indices, sorted by score from high to low

            gt_boxes = gt_bboxes_rotated[bid][:num_obj]
            _, _, c_dist = iou3d_rotated_vs_aligned(
                gt_boxes, 
                pbox
            )

            topks = self.topks  # [1, 5, 10]
            for t in self.thresholds:  # 0.5, 1.0
                thresholded = c_dist < t
                for k in topks:
                    found = thresholded[:, :k].any(1)
                    self.dets[(prefix, t, k, "bbs")] += found.sum().item()  # Number of hit GT boxes
                    self.gts[(prefix, t, k, "bbs")] += len(thresholded)  # Total number of GT boxes

    def evaluate_bbox_by_contrast(self, batch_data, end_points, prefix):
        """
        Evaluate bounding box IoU using contrastive learning (via similarity between query and token features)

        Core idea:
        1. DETR model predicts 256 candidate boxes (set prediction)
        2. Compute contrastive matching score between each candidate and language tokens
        3. Select top-k candidates with highest scores
        4. Compute IoU between these candidates and GT to evaluate accuracy

        Args:
            batch_data (dict): contains original data (utterances, meta_path, etc.)
            end_points (dict): contains model predictions and ground truth
            prefix (str): layer name, e.g., "last_" or "proposal_"
        """
        positive_map = torch.clone(end_points["positive_map"])
        positive_map[positive_map > 0] = 1
        gt_bboxes_rotated = batch_data["gt_bboxes"]
        pred_center = end_points[f"{prefix}center"]
        pred_size = end_points[f"{prefix}pred_size"]
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)

        proj_tokens = end_points["proj_tokens"]
        proj_queries = end_points[f"{prefix}proj_queries"]
        sem_scores = torch.matmul(
            proj_queries, proj_tokens.transpose(-1, -2)
        )
        sem_scores_ = (sem_scores / 0.07).softmax(-1)
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, : sem_scores_.size(1), : sem_scores_.size(2)] = sem_scores_

        for bid in range(len(positive_map)):
            num_obj = int(end_points["box_label_mask"][bid].sum())
            assert num_obj == 1, f"num_obj: {num_obj}. only support obj number is 1."
            pmap = positive_map[bid, :num_obj]
            scores = (sem_scores[bid].unsqueeze(0) * pmap.unsqueeze(1)).sum(-1)
            top = scores.argsort(1, True)[:, :10]
            pbox = pred_bbox[bid, top.reshape(-1)]

            gt_boxes = gt_bboxes_rotated[bid][:num_obj]
            ious, _, c_dist = iou3d_rotated_vs_aligned(
                gt_boxes, 
                pbox
            )

            meta_path = batch_data["meta_path"][bid]

            record = {
                "id": meta_path.split("/")[-1].replace(".pcd.bin", ""),
                "utterance": batch_data["utterances"][bid],
                "gt_box": batch_data["gt_bboxes"][bid][:num_obj].cpu().numpy().tolist(),
                "pred_box": pbox[0].cpu().numpy().tolist(),
                "ious": ious[:, 0].cpu().numpy().tolist(),
                "c_dist": c_dist[0].cpu().numpy().tolist(),
                "top_k": self.topks,
            }
            self.prediction_records.append(record)

            self.dets["iou"] += ious[:, 0].cpu().numpy().sum()
            self.dets["num_iou"] += num_obj

            for t in self.thresholds:
                thresholded = c_dist < t
                for k in self.topks:
                    found = thresholded[:, :k].any(1)
                    all_found = found.all().item()

                    self.dets[(prefix, t, k, "bbf")] += all_found
                    self.gts[(prefix, t, k, "bbf")] += 1

                    if prefix == "last_" and k == 1:
                        self.dets[("total_acc", t, "bbf")] += all_found
                        self.gts[("total_acc", t, "bbf")] += 1

    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points["positive_map"])
        positive_map[positive_map > 0] = 1
        gt_center = end_points["center_label"][:, :, 0:3]
        gt_size = end_points["size_gts"]
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)
        if self.only_root:
            positive_map = positive_map[:, :1]
            gt_bboxes = gt_bboxes[:, :1]
        return positive_map, gt_bboxes
