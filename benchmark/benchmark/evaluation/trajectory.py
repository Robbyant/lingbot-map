"""Trajectory evaluation using evo library (ATE and RPE metrics)."""

import copy
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib.pyplot as plt
import numpy as np


def _filter_valid_pose_pairs(
    gt_poses: np.ndarray,
    pred_poses: np.ndarray,
    timestamps: np.ndarray,
    logger=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop frame pairs where GT or pred pose contains NaN/Inf.

    Logs a WARNING if any pairs are dropped (indicates some GT frames lack
    valid poses, e.g. TUM RGB-D timestamp association gaps).

    Returns:
        Filtered (gt_poses, pred_poses, timestamps), all with NaN-free poses.
    Raises:
        ValueError: If no valid pairs remain after filtering.
    """
    gt_valid   = np.isfinite(gt_poses.reshape(len(gt_poses), -1)).all(axis=1)
    pred_valid = np.isfinite(pred_poses.reshape(len(pred_poses), -1)).all(axis=1)
    mask = gt_valid & pred_valid
    n_dropped = int((~mask).sum())
    if n_dropped > 0:
        msg = (f"Dropped {n_dropped}/{len(mask)} frame(s) with NaN GT/pred pose "
               "(GT trajectory is incomplete for these frames)")
        if logger:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")
    gt_poses   = gt_poses[mask]
    pred_poses = pred_poses[mask]
    timestamps = timestamps[mask]
    if len(gt_poses) == 0:
        raise ValueError("No valid GT/pred pose pairs after NaN filtering")
    return gt_poses, pred_poses, timestamps


def _orthogonalize_se3(pose: np.ndarray) -> np.ndarray:
    """Orthogonalize the rotation part of a 4x4 SE3 matrix via SVD.

    Required because evo validates SO(3) membership, but some datasets
    (e.g., 7Scenes KinectFusion) have slightly non-orthogonal R.

    Args:
        pose: 4x4 transformation matrix

    Returns:
        4x4 matrix with orthogonalized rotation
    """
    result = pose.copy()
    R = result[:3, :3]
    if not np.all(np.isfinite(R)):
        raise ValueError(f"Rotation matrix contains non-finite values (NaN/Inf): {R}")
    try:
        U, _, Vh = np.linalg.svd(R)
    except np.linalg.LinAlgError:
        # Fall back to QR decomposition if SVD fails to converge
        Q, _ = np.linalg.qr(R)
        result[:3, :3] = Q
        return result
    R_ortho = U @ Vh
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vh
    result[:3, :3] = R_ortho
    return result


def _array_to_evo_trajectory(poses: np.ndarray, timestamps: np.ndarray) -> PoseTrajectory3D:
    """Convert valid pose array to evo PoseTrajectory3D.

    Orthogonalizes rotations at the evo boundary to satisfy SO(3) validation.

    Args:
        poses:      (M, 4, 4) array of valid (non-NaN) poses
        timestamps: (M,) float array of timestamps (frame indices as floats)

    Returns:
        evo PoseTrajectory3D object
    """
    poses_se3 = [_orthogonalize_se3(poses[i]) for i in range(len(poses))]
    return PoseTrajectory3D(poses_se3=poses_se3, timestamps=timestamps)


class TrajectoryEvaluator:
    """Evaluates camera trajectories using ATE and RPE metrics."""

    def __init__(self, align: bool = True, correct_scale: bool = True):
        """Initialize trajectory evaluator.

        Args:
            align: Whether to align trajectories before evaluation
            correct_scale: Whether to correct scale during alignment
        """
        self.align = align
        self.correct_scale = correct_scale

    def evaluate(self, gt_loader, pred_loader,
                 logger: Optional[logging.Logger] = None) -> Dict[str, float]:
        """Evaluate trajectory using evo library (ATE and RPE).

        Args:
            gt_loader:   BSSLoader for ground truth directory
            pred_loader: BSSLoader for predicted directory
            logger:      Optional logger

        Returns:
            Dictionary containing:
            - 'ate': Absolute Trajectory Error (RMSE)
            - 'rpe_trans': RPE translation RMSE
            - 'rpe_rot': RPE rotation RMSE (degrees)
            - 'traj_transform': 4x4 Sim(3) alignment matrix (np.ndarray)

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

        if len(gt_poses) == 0 or len(pred_poses) == 0:
            raise ValueError("Empty trajectory — no frames to evaluate")

        timestamps_float = np.array(pred_frame_indices, dtype=float)

        gt_poses, pred_poses, timestamps_float = _filter_valid_pose_pairs(
            gt_poses, pred_poses, timestamps_float, logger)

        logger.info(f"Evaluating trajectory on {len(gt_poses)} frame pairs")

        traj_ref = _array_to_evo_trajectory(gt_poses, timestamps_float)
        traj_est = _array_to_evo_trajectory(pred_poses, timestamps_float)

        # Align estimated trajectory if requested
        traj_est_aligned = copy.deepcopy(traj_est)
        alignment_result = None
        T_align = np.eye(4)
        if self.align:
            alignment_result = traj_est_aligned.align(traj_ref, correct_scale=self.correct_scale)

        if self.align and alignment_result is not None:
            R, t, scale = alignment_result  # evo returns (rotation, translation, scale)
            T_align = np.eye(4)
            T_align[:3, :3] = scale * R
            T_align[:3, 3] = t.flatten()

        # Compute ATE (Absolute Trajectory Error)
        ape_result = main_ape.ape(
            traj_ref,
            traj_est_aligned,
            est_name='traj',
            pose_relation=PoseRelation.translation_part,
            align=False,
            correct_scale=False
        )
        ate = ape_result.stats["rmse"]

        # Compute RPE (Relative Pose Error) - rotation
        rpe_rot_result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=self.align,
            correct_scale=self.correct_scale,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
        rpe_rot = rpe_rot_result.stats["rmse"]

        # Compute RPE (Relative Pose Error) - translation
        rpe_trans_result = main_rpe.rpe(
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=self.align,
            correct_scale=self.correct_scale,
            delta=1,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )
        rpe_trans = rpe_trans_result.stats["rmse"]

        return {
            'ate': float(ate),
            'rpe_trans': float(rpe_trans),
            'rpe_rot': float(rpe_rot),
            'traj_transform': T_align,
        }

    def save_visualization(self, gt_loader, pred_loader, output_dir: Path,
                           logger: Optional[logging.Logger] = None) -> None:
        """Visualize aligned trajectories in 4 views (xyz, xy, xz, yz).

        Args:
            gt_loader:   BSSLoader for ground truth directory
            pred_loader: BSSLoader for predicted directory
            output_dir:  Directory to save the visualization (file: trajectory_visualization.png)
            logger:      Optional logger
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

        if len(gt_poses) == 0:
            raise ValueError("No valid pose pairs found between reference and estimated trajectories")

        timestamps_float = np.array(pred_frame_indices, dtype=float)

        gt_poses, pred_poses, timestamps_float = _filter_valid_pose_pairs(
            gt_poses, pred_poses, timestamps_float, logger)

        traj_ref = _array_to_evo_trajectory(gt_poses, timestamps_float)
        traj_est = _array_to_evo_trajectory(pred_poses, timestamps_float)

        traj_est_aligned = copy.deepcopy(traj_est)
        if self.align:
            traj_est_aligned.align(traj_ref, correct_scale=self.correct_scale)

        # Build full GT trajectory (all valid poses) for visualization so that a method
        # which only processed a small portion of frames cannot look deceptively good.
        # Alignment above is still computed against the matched subset (traj_ref), which
        # is correct; only the reference drawn in grey uses the full GT here.
        gt_full_valid_mask = np.isfinite(gt_traj.reshape(len(gt_traj), -1)).all(axis=1)
        gt_full_poses = gt_traj[gt_full_valid_mask]
        gt_full_timestamps = np.where(gt_full_valid_mask)[0].astype(float)
        traj_ref_full = _array_to_evo_trajectory(gt_full_poses, gt_full_timestamps)

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(12, 12))

        ax = plot.prepare_axis(fig, plot.PlotMode.xyz, subplot_arg=221)
        ax.set_title("XYZ")
        plot.traj(ax, plot.PlotMode.xyz, traj_ref_full, '--', 'gray', label="ref", plot_start_end_markers=True)
        plot.traj(ax, plot.PlotMode.xyz, traj_est_aligned, '-', 'blue', label="est", plot_start_end_markers=True)
        fig.axes.append(ax)

        ax = plot.prepare_axis(fig, plot.PlotMode.xy, subplot_arg=222)
        ax.set_title("XY")
        plot.traj(ax, plot.PlotMode.xy, traj_ref_full, '--', 'gray', label="ref", plot_start_end_markers=True)
        plot.traj(ax, plot.PlotMode.xy, traj_est_aligned, '-', 'blue', label="est", plot_start_end_markers=True)
        fig.axes.append(ax)

        ax = plot.prepare_axis(fig, plot.PlotMode.xz, subplot_arg=223)
        ax.set_title("XZ")
        plot.traj(ax, plot.PlotMode.xz, traj_ref_full, '--', 'gray', label="ref", plot_start_end_markers=True)
        plot.traj(ax, plot.PlotMode.xz, traj_est_aligned, '-', 'blue', label="est", plot_start_end_markers=True)
        fig.axes.append(ax)

        ax = plot.prepare_axis(fig, plot.PlotMode.yz, subplot_arg=224)
        ax.set_title("YZ")
        plot.traj(ax, plot.PlotMode.yz, traj_ref_full, '--', 'gray', label="ref", plot_start_end_markers=True)
        plot.traj(ax, plot.PlotMode.yz, traj_est_aligned, '-', 'blue', label="est", plot_start_end_markers=True)
        fig.axes.append(ax)

        plt.legend()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'trajectory_visualization.png'
        plt.savefig(output_path, dpi=120)
        plt.close(fig)

        logger.info(f"Trajectory visualization saved to {output_path}")
