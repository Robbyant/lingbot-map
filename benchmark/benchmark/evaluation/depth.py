"""Depth evaluation module for computing depth metrics."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm
import numpy as np
import cv2

from benchmark.core.loader import BSSLoader


class DepthEvaluator:
    """Evaluates depth prediction quality using standard metrics.

    Computes metrics like absolute relative error, RMSE, and delta accuracies
    with support for different alignment methods (lstsq, scale_only, none).
    """

    def __init__(self,
                 align: str = 'scale_only',
                 gt_clip: Tuple[float, float] = (0, 80),
                 pre_clip: Optional[Tuple[float, float]] = None,
                 post_clip: Optional[Tuple[float, float]] = None):
        """Initialize depth evaluator.

        Args:
            align:     Alignment method - 'lstsq', 'scale_only', or 'none'
            gt_clip:   (min, max) for ground truth depth range
            pre_clip:  Optional (min, max) for pre-alignment clipping
            post_clip: Optional (min, max) for post-alignment clipping
        """
        if align not in ['lstsq', 'scale_only', 'none']:
            raise ValueError(f"Invalid align method: {align}. Must be 'lstsq', 'scale_only', or 'none'")

        self.align = align
        self.gt_clip = gt_clip
        self.pre_clip = pre_clip
        self.post_clip = post_clip

    def evaluate(self,
                 gt_loader: BSSLoader,
                 pred_loader: BSSLoader,
                 logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """Evaluate depth predictions against ground truth.

        Args:
            gt_loader:   BSSLoader for the ground truth directory
            pred_loader: BSSLoader for the method output directory
            logger:      Optional logger for status messages

        Returns:
            Dictionary containing depth metrics
        """
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            logger = logging.getLogger(__name__)

        if not gt_loader.has_frame_key('depth'):
            logger.warning("Ground truth depth directory not found, skipping depth evaluation")
            return {}

        if not pred_loader.has_frame_key('depth'):
            logger.warning("Method depth directory not found, skipping depth evaluation")
            return {}

        try:
            N = gt_loader.get_num_frames()
            logger.info(f"Evaluating depth on {N} frames")

            # Load depth lists at native resolution
            gt_depth_list = gt_loader.load_depth_list()
            method_depth_list = pred_loader.load_depth_list()

            if gt_depth_list is None:
                logger.warning("Ground truth depth list is None")
                return {}

            if method_depth_list is None:
                logger.warning("Method depth list is None")
                return {}

        except Exception as e:
            logger.error(f"Error loading depth data: {e}")
            return {}

        # Accumulate all valid pixels across frames
        all_pred_pixels = []
        all_gt_pixels = []
        num_frames_evaluated = 0

        for idx in range(N):
            try:
                gt_depth = gt_depth_list[idx] if idx < len(gt_depth_list) else None
                pred_depth = method_depth_list[idx] if idx < len(method_depth_list) else None

                if gt_depth is None or pred_depth is None:
                    continue

                # Defensive resize if shapes differ (should not happen with unified GT resolution)
                if gt_depth.shape != pred_depth.shape:
                    logger.debug(
                        f"Shape mismatch at idx {idx}: GT {gt_depth.shape} vs pred {pred_depth.shape},"
                        f" resizing GT"
                    )
                    gt_depth = cv2.resize(
                        gt_depth, (pred_depth.shape[1], pred_depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                # Process single frame
                frame_metrics, aligned_pred = self.evaluate_single_frame(pred_depth, gt_depth)

                if frame_metrics is None:
                    continue

                # Collect valid pixels
                valid_mask = frame_metrics['valid_mask']
                all_pred_pixels.append(aligned_pred[valid_mask])
                all_gt_pixels.append(gt_depth[valid_mask])
                num_frames_evaluated += 1

            except Exception as e:
                logger.warning(f"Error evaluating frame {idx}: {e}")
                continue

        if num_frames_evaluated == 0:
            logger.warning("No valid frames found for depth evaluation")
            return {}

        # Concatenate all pixels
        all_pred = np.concatenate(all_pred_pixels)
        all_gt = np.concatenate(all_gt_pixels)
        num_valid_pixels = len(all_pred)

        logger.info(f"Computing metrics on {num_valid_pixels} valid pixels from {num_frames_evaluated} frames")

        # Compute aggregate metrics
        metrics = self._compute_metrics(all_pred, all_gt)

        return metrics

    def evaluate_single_frame(self,
                              pred_depth: np.ndarray,
                              gt_depth: np.ndarray) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """Evaluate a single frame.

        Args:
            pred_depth: Predicted depth map
            gt_depth:   Ground truth depth map

        Returns:
            Tuple of (frame_metrics_dict, aligned_pred_depth)
            Returns (None, None) if frame is invalid
        """
        gt_min, gt_max = self.gt_clip
        valid_mask = (gt_depth > gt_min) & (gt_depth < gt_max)
        valid_mask = valid_mask & np.isfinite(gt_depth) & np.isfinite(pred_depth)
        valid_mask = valid_mask & (pred_depth > 0)

        if np.sum(valid_mask) < 100:
            return None, None

        pred_valid = pred_depth[valid_mask].copy()
        gt_valid = gt_depth[valid_mask].copy()

        if self.pre_clip is not None:
            pre_min, pre_max = self.pre_clip
            pred_valid = np.clip(pred_valid, pre_min, pre_max)

        if self.align == 'lstsq':
            try:
                s, t = self._align_lstsq(pred_valid, gt_valid)
                pred_aligned = s * pred_depth + t
            except Exception:
                s = self._align_scale_only(pred_valid, gt_valid)
                pred_aligned = s * pred_depth
        elif self.align == 'scale_only':
            s = self._align_scale_only(pred_valid, gt_valid)
            pred_aligned = s * pred_depth
        else:  # 'none'
            pred_aligned = pred_depth.copy()

        if self.post_clip is not None:
            post_min, post_max = self.post_clip
            pred_aligned = np.clip(pred_aligned, post_min, post_max)

        return {'valid_mask': valid_mask}, pred_aligned

    @staticmethod
    def _align_lstsq(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
        """Solve gt = s * pred + t using least squares."""
        A = np.hstack([pred.reshape(-1, 1), np.ones((len(pred), 1))])
        result = np.linalg.lstsq(A, gt.reshape(-1, 1), rcond=None)
        s, t = result[0][0, 0], result[0][1, 0]
        return s, t

    @staticmethod
    def _align_scale_only(pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute scale using median ratio."""
        return np.median(gt) / np.median(pred)

    @staticmethod
    def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """Compute depth metrics on concatenated valid pixels."""
        abs_rel = np.mean(np.abs(pred - gt) / gt)
        sq_rel = np.mean((pred - gt) ** 2 / gt)
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        pred_clipped = np.clip(pred, 1e-5, None)
        gt_clipped = np.clip(gt, 1e-5, None)
        log_rmse = np.sqrt(np.mean((np.log(pred_clipped) - np.log(gt_clipped)) ** 2))
        max_ratio = np.maximum(pred / gt, gt / pred)
        delta_1_25 = np.mean((max_ratio < 1.25).astype(float)) * 100
        delta_1_25_2 = np.mean((max_ratio < 1.5625).astype(float)) * 100
        delta_1_25_3 = np.mean((max_ratio < 1.953125).astype(float)) * 100

        return {
            'abs_rel': float(abs_rel),
            'sq_rel': float(sq_rel),
            'rmse': float(rmse),
            'log_rmse': float(log_rmse),
            'delta_1_25': float(delta_1_25),
            'delta_1_25_2': float(delta_1_25_2),
            'delta_1_25_3': float(delta_1_25_3),
        }

    def save_visualization(self,
                           gt_loader: BSSLoader,
                           pred_loader: BSSLoader,
                           output_dir: Path,
                           logger: Optional[logging.Logger] = None) -> None:
        """Generate per-frame depth visualizations with RGB, GT, prediction, and error.

        Args:
            gt_loader:   BSSLoader for ground truth directory
            pred_loader: BSSLoader for method predictions directory
            output_dir:  Directory to save frame visualizations
            logger:      Optional logger for status messages
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            logger = logging.getLogger(__name__)

        try:
            N = gt_loader.get_num_frames()

            gt_depth_list = gt_loader.load_depth_list()
            method_depth_list = pred_loader.load_depth_list()

            if gt_depth_list is None or method_depth_list is None:
                logger.warning("Depth lists not available for visualization")
                return

            try:
                rgb_list = gt_loader.load_rgb_list()
            except Exception as e:
                logger.warning(f"Failed to load RGB images: {e}")
                rgb_list = None

            vis_dir = Path(output_dir)
            vis_dir.mkdir(exist_ok=True, parents=True)

            num_saved = 0
            for idx in tqdm(
                range(N), total=N,
                dynamic_ncols=True, desc="Generating visualizations"
            ):
                gt_depth = gt_depth_list[idx] if idx < len(gt_depth_list) else None
                pred_depth = method_depth_list[idx] if idx < len(method_depth_list) else None

                if gt_depth is None or pred_depth is None:
                    continue

                rgb_image = None
                if rgb_list is not None and idx < len(rgb_list):
                    rgb_image = rgb_list[idx]

                # Defensive resize if shapes differ
                if gt_depth.shape != pred_depth.shape:
                    pred_depth = cv2.resize(
                        pred_depth, (gt_depth.shape[1], gt_depth.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )

                frame_metrics, aligned_pred = self.evaluate_single_frame(pred_depth, gt_depth)

                if frame_metrics is None:
                    continue

                valid_mask = frame_metrics['valid_mask']

                abs_rel_error = np.full_like(gt_depth, np.nan)
                with np.errstate(divide='ignore', invalid='ignore'):
                    abs_rel_error[valid_mask] = (
                        np.abs(aligned_pred[valid_mask] - gt_depth[valid_mask]) / gt_depth[valid_mask]
                    )

                img_height, img_width = gt_depth.shape
                aspect_ratio = img_height / img_width
                total_width = 12
                single_width = total_width / 2
                single_height = single_width * aspect_ratio
                total_height = single_height * 2

                fig = plt.figure(figsize=(total_width, total_height), constrained_layout=True)

                valid_gt = gt_depth[valid_mask]
                if len(valid_gt) > 0:
                    depth_min = np.percentile(valid_gt, 5)
                    depth_max = np.percentile(valid_gt, 95)
                else:
                    depth_min, depth_max = self.gt_clip

                ax2 = plt.subplot(2, 2, 1)
                im2 = ax2.imshow(gt_depth, cmap='viridis', vmin=depth_min, vmax=depth_max)
                ax2.axis('off')
                plt.colorbar(im2, ax=ax2, label='Depth (m)', fraction=0.046, pad=0.04)

                ax3 = plt.subplot(2, 2, 2)
                im3 = ax3.imshow(aligned_pred, cmap='viridis', vmin=depth_min, vmax=depth_max)
                ax3.axis('off')
                plt.colorbar(im3, ax=ax3, label='Depth (m)', fraction=0.046, pad=0.04)

                ax1 = plt.subplot(2, 2, 3)
                if rgb_image is not None:
                    ax1.imshow(rgb_image)
                else:
                    ax1.text(0.5, 0.5, 'RGB Not Available', ha='center', va='center',
                             transform=ax1.transAxes)
                ax1.axis('off')

                ax4 = plt.subplot(2, 2, 4)
                im4 = ax4.imshow(abs_rel_error, cmap='hot', vmin=0, vmax=0.5)
                ax4.axis('off')
                plt.colorbar(im4, ax=ax4, label='Abs Rel Error', fraction=0.046, pad=0.04)

                output_file = vis_dir / f'{idx:06d}.jpg'
                plt.savefig(output_file, format='jpg', dpi=100, bbox_inches='tight')
                plt.close()

                num_saved += 1

            logger.info(f"Saved {num_saved} depth frame visualizations to {vis_dir}")

        except Exception as e:
            logger.error(f"Error generating depth visualization: {e}")
