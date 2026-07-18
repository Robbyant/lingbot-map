"""AUC (Area Under Curve) evaluation for camera poses."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from benchmark.geometry.transform import invert_transform
from benchmark.geometry.quaternion import rotation_matrix_to_quaternion
from benchmark.evaluation.trajectory import _filter_valid_pose_pairs


# AUC evaluation thresholds (degrees)
DEFAULT_AUC_THRESHOLDS = [3, 5, 15, 30]


class AUCEvaluator:
    """Evaluates camera pose accuracy using AUC metrics."""

    def __init__(self, thresholds: List[int] = None):
        """Initialize AUC evaluator.

        Args:
            thresholds: List of angle thresholds in degrees (default: [3, 5, 15, 30])
        """
        self.thresholds = thresholds if thresholds else DEFAULT_AUC_THRESHOLDS

    def evaluate(self, gt_loader, pred_loader,
                 logger: Optional[logging.Logger] = None) -> Dict[str, float]:
        """Compute pose AUC metrics for a scene.

        Args:
            gt_loader:   BSSLoader for ground truth directory
            pred_loader: BSSLoader for predicted directory
            logger:      Optional logger

        Returns:
            Dictionary containing AUC metrics:
            - 'AUC_03', 'AUC_05', 'AUC_15', 'AUC_30': AUC at different thresholds
            - 'Racc_03', 'Racc_05', 'Racc_15', 'Racc_30': Rotation accuracy
            - 'Tacc_03', 'Tacc_05', 'Tacc_15', 'Tacc_30': Translation accuracy

        Raises:
            ValueError: If no valid pose pairs found
        """
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            logger = logging.getLogger(__name__)

        gt_traj = gt_loader.load_trajectory()
        pred_traj = pred_loader.load_trajectory()

        if gt_traj is None:
            raise FileNotFoundError(f"GT trajectory not found: {gt_loader.artifact.traj_file}")
        if pred_traj is None:
            raise FileNotFoundError(f"Predicted trajectory not found: {pred_loader.artifact.traj_file}")

        # Use frame_index_map to select matching GT poses for sparse SLAM outputs.
        # For dense methods the map is identity [0, 1, ..., N-1].
        pred_frame_indices = pred_loader.get_frame_indices()
        gt_poses   = gt_traj[pred_frame_indices]  # (K, 4, 4)
        pred_poses = pred_traj                    # (K, 4, 4), always dense/valid
        num_frames = len(pred_frame_indices)

        if num_frames == 0:
            raise ValueError("Empty trajectory — no frames to evaluate")

        timestamps_float = np.array(pred_frame_indices, dtype=float)
        gt_poses, pred_poses, timestamps_float = _filter_valid_pose_pairs(
            gt_poses, pred_poses, timestamps_float, logger)
        num_frames = len(gt_poses)

        # C2W → W2C: use np.linalg.inv for proper inverse of non-orthogonal R
        gt_extrs = np.linalg.inv(gt_poses).astype(np.float32)
        pred_extrs = np.linalg.inv(pred_poses).astype(np.float32)

        # Align to first camera
        gt_extrs = self._align_to_first_camera(gt_extrs)
        pred_extrs = self._align_to_first_camera(pred_extrs)

        # Compute relative pose errors
        rel_rangle, rel_tangle = self._se3_to_relative_pose_error(
            pred_se3=pred_extrs,
            gt_se3=gt_extrs,
            num_frames=num_frames,
        )

        # Compute metrics for each threshold
        metrics = dict()
        for threshold in self.thresholds:
            metrics[f"Racc_{threshold:02d}"] = float(np.mean(rel_rangle < threshold) * 100)
            metrics[f"Tacc_{threshold:02d}"] = float(np.mean(rel_tangle < threshold) * 100)
            auc, _ = self._calculate_auc_np(rel_rangle, rel_tangle, max_threshold=threshold)
            metrics[f"AUC_{threshold:02d}"] = float(auc * 100)

        metrics['num_pairs'] = len(rel_rangle)
        metrics['rError'] = rel_rangle.tolist()
        metrics['tError'] = rel_tangle.tolist()

        return metrics

    @staticmethod
    def _build_pair_index(N: int, B: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate all pairwise indices for N frames."""
        i1_, i2_ = np.triu_indices(N, k=1)
        if B == 1:
            return i1_, i2_
        else:
            i1 = (i1_[None, :] + np.arange(B)[:, None] * N).reshape(-1)
            i2 = (i2_[None, :] + np.arange(B)[:, None] * N).reshape(-1)
            return i1, i2

    @staticmethod
    def _align_to_first_camera(poses: np.ndarray) -> np.ndarray:
        """Align all camera poses to the first camera's coordinate frame."""
        first_pose_inv = invert_transform(poses[0:1])[0]
        return poses @ first_pose_inv

    @staticmethod
    def _rotation_angle(rot_gt: np.ndarray, rot_pred: np.ndarray,
                        batch_size: Optional[int] = None, eps: float = 1e-15) -> np.ndarray:
        """Compute rotation angle error between ground truth and predicted rotations."""
        q_pred = rotation_matrix_to_quaternion(rot_pred)
        q_gt = rotation_matrix_to_quaternion(rot_gt)
        loss_q = np.maximum(1 - (q_pred * q_gt).sum(axis=1) ** 2, eps)
        err_q = np.arccos(1 - 2 * loss_q)
        rel_rangle_deg = err_q * 180 / np.pi
        if batch_size is not None:
            rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)
        return rel_rangle_deg

    @staticmethod
    def _translation_angle(tvec_gt: np.ndarray, tvec_pred: np.ndarray,
                           batch_size: Optional[int] = None, ambiguity: bool = True,
                           eps: float = 1e-15, default_err: float = 1e6) -> np.ndarray:
        """Compute translation angle error between ground truth and predicted translations."""
        t_norm = np.linalg.norm(tvec_pred, axis=1, keepdims=True)
        t = tvec_pred / (t_norm + eps)
        t_gt_norm = np.linalg.norm(tvec_gt, axis=1, keepdims=True)
        t_gt = tvec_gt / (t_gt_norm + eps)
        loss_t = np.maximum(1.0 - np.sum(t * t_gt, axis=1) ** 2, eps)
        err_t = np.arccos(np.sqrt(1 - loss_t))
        err_t[np.isnan(err_t) | np.isinf(err_t)] = default_err
        rel_tangle_deg = err_t * 180.0 / np.pi
        if ambiguity:
            rel_tangle_deg = np.minimum(rel_tangle_deg, np.abs(180 - rel_tangle_deg))
        if batch_size is not None:
            rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)
        return rel_tangle_deg

    def _se3_to_relative_pose_error(self, pred_se3: np.ndarray, gt_se3: np.ndarray,
                                    num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute relative pose errors between frame pairs (W2C format)."""
        pair_idx_i1, pair_idx_i2 = self._build_pair_index(num_frames)
        relative_pose_gt = invert_transform(gt_se3[pair_idx_i1]) @ gt_se3[pair_idx_i2]
        relative_pose_pred = invert_transform(pred_se3[pair_idx_i1]) @ pred_se3[pair_idx_i2]
        rel_rangle_deg = self._rotation_angle(
            relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
        )
        rel_tangle_deg = self._translation_angle(
            relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
        )
        return rel_rangle_deg, rel_tangle_deg

    @staticmethod
    def _calculate_auc_np(r_error: np.ndarray, t_error: np.ndarray,
                          max_threshold: int = 30) -> Tuple[float, np.ndarray]:
        """Calculate the Area Under the Curve (AUC) for the given error arrays."""
        error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
        max_errors = np.max(error_matrix, axis=1)
        bins = np.arange(max_threshold + 1)
        histogram, _ = np.histogram(max_errors, bins=bins)
        num_pairs = float(len(max_errors))
        normalized_histogram = histogram.astype(float) / num_pairs
        return np.mean(np.cumsum(normalized_histogram)), normalized_histogram

    def compute_auc_vs_frames(self, gt_loader, pred_loader,
                              step: int = 10, threshold: int = 30,
                              logger: Optional[logging.Logger] = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute how AUC changes as more frames are added.

        Args:
            gt_loader:   BSSLoader for ground truth directory
            pred_loader: BSSLoader for predicted directory
            step:        Step size for frame increments
            threshold:   Threshold to use for AUC computation
            logger:      Optional logger

        Returns:
            Tuple of (frame_counts, auc_values, racc_values, tacc_values)
        """
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            logger = logging.getLogger(__name__)

        gt_traj = gt_loader.load_trajectory()
        pred_traj = pred_loader.load_trajectory()

        if gt_traj is None:
            raise FileNotFoundError(f"GT trajectory not found: {gt_loader.artifact.traj_file}")
        if pred_traj is None:
            raise FileNotFoundError(f"Predicted trajectory not found: {pred_loader.artifact.traj_file}")

        pred_frame_indices = pred_loader.get_frame_indices()
        gt_poses   = gt_traj[pred_frame_indices]
        pred_poses = pred_traj
        num_frames = len(pred_frame_indices)

        if num_frames == 0:
            raise ValueError("Empty trajectory — no frames to evaluate")

        timestamps_float = np.array(pred_frame_indices, dtype=float)
        gt_poses, pred_poses, timestamps_float = _filter_valid_pose_pairs(
            gt_poses, pred_poses, timestamps_float, logger)
        num_frames = len(gt_poses)

        gt_extrs = np.linalg.inv(gt_poses).astype(np.float32)
        pred_extrs = np.linalg.inv(pred_poses).astype(np.float32)
        gt_extrs = self._align_to_first_camera(gt_extrs)
        pred_extrs = self._align_to_first_camera(pred_extrs)

        frame_counts = list(range(step, num_frames + 1, step))
        if frame_counts[-1] != num_frames:
            frame_counts.append(num_frames)

        auc_values, racc_values, tacc_values = [], [], []

        for n in frame_counts:
            rel_rangle, rel_tangle = self._se3_to_relative_pose_error(
                pred_se3=pred_extrs[:n],
                gt_se3=gt_extrs[:n],
                num_frames=n
            )
            auc, _ = self._calculate_auc_np(rel_rangle, rel_tangle, max_threshold=threshold)
            auc_values.append(auc * 100)
            racc_values.append(np.mean(rel_rangle < threshold) * 100)
            tacc_values.append(np.mean(rel_tangle < threshold) * 100)

        return (
            np.array(frame_counts),
            np.array(auc_values),
            np.array(racc_values),
            np.array(tacc_values)
        )

    def save_visualization(self, frame_counts: np.ndarray, auc_values: np.ndarray,
                           racc_values: np.ndarray, tacc_values: np.ndarray,
                           output_dir: Path, threshold: int = 30,
                           logger: Optional[logging.Logger] = None) -> None:
        """Plot AUC and accuracy metrics vs number of frames and save to output_dir.

        Args:
            frame_counts: Array of frame counts
            auc_values:   AUC values for each frame count
            racc_values:  Rotation accuracy values
            tacc_values:  Translation accuracy values
            output_dir:   Directory to save the plot (file: auc_vs_frames.png)
            threshold:    Threshold used for AUC computation
            logger:       Optional logger
        """
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            logger = logging.getLogger(__name__)

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(frame_counts, auc_values, 'b-o', label=f'AUC@{threshold}', linewidth=2, markersize=4)
        ax.plot(frame_counts, racc_values, 'g--s', label=f'Racc@{threshold}', linewidth=1.5, markersize=3, alpha=0.7)
        ax.plot(frame_counts, tacc_values, 'r--^', label=f'Tacc@{threshold}', linewidth=1.5, markersize=3, alpha=0.7)

        ax.set_xlabel('Number of Frames', fontsize=12)
        ax.set_ylabel('Metric Value (%)', fontsize=12)
        ax.set_title(f'AUC vs Frame Count', fontsize=14)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        # Mark significant drop points (>5% drop from previous)
        for i in range(1, len(auc_values)):
            drop = auc_values[i - 1] - auc_values[i]
            if drop > 5:
                ax.annotate(
                    f'-{drop:.1f}%',
                    xy=(frame_counts[i], auc_values[i]),
                    xytext=(frame_counts[i] + 0.5, auc_values[i] + 5),
                    fontsize=8,
                    color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
                )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'auc_vs_frames.png'
        plt.savefig(output_path, dpi=120)
        plt.close(fig)

        logger.info(f"AUC vs frames plot saved to {output_path}")

    @classmethod
    def aggregate_micro(cls, all_scene_metrics: List[Dict]) -> Dict[str, float]:
        """Pool rError/tError from all scenes, compute AUC once."""
        all_r = np.concatenate([m['rError'] for m in all_scene_metrics])
        all_t = np.concatenate([m['tError'] for m in all_scene_metrics])
        result = {'num_pairs': int(len(all_r))}
        for threshold in DEFAULT_AUC_THRESHOLDS:
            result[f'Racc_{threshold:02d}'] = float(np.mean(all_r < threshold) * 100)
            result[f'Tacc_{threshold:02d}'] = float(np.mean(all_t < threshold) * 100)
            auc, _ = cls._calculate_auc_np(all_r, all_t, max_threshold=threshold)
            result[f'AUC_{threshold:02d}'] = float(auc * 100)
        return result

    @classmethod
    def aggregate_macro(cls, all_scene_metrics: List[Dict]) -> Dict[str, float]:
        """Average per-scene AUC/Racc/Tacc values."""
        keys = [k for k in all_scene_metrics[0]
                if k not in ('rError', 'tError', 'num_pairs')]
        result = {'num_scenes': len(all_scene_metrics)}
        for k in keys:
            result[k] = float(np.mean([m[k] for m in all_scene_metrics]))
        return result

    @staticmethod
    def strip_raw_errors(metrics: Dict) -> Dict:
        """Return metrics without rError/tError, for Layer 2 comparison view."""
        return {k: v for k, v in metrics.items() if k not in ('rError', 'tError')}
