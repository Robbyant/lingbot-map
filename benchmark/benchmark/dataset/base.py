"""Base dataset interface for the benchmark framework.

All datasets must inherit from BaseDataset and implement the required methods.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import random


class BaseDataset(ABC):
    """Abstract base class for all datasets.

    This class defines the interface that all dataset implementations must follow.
    Datasets are responsible for loading raw data and converting it to the
    standardized format expected by the framework.

    Custom Saver Extension:
    ----------------------
    To save custom data types (e.g., 'semantic', 'normal'), implement
    methods with the naming pattern: __save_<key>_file__

    Example for saving semantic segmentation maps:

        def __save_semantic_file__(self, output_dir: Path, base_name: str,
                                   data: np.ndarray) -> None:
            '''Save semantic segmentation map.'''
            semantic_dir = output_dir / 'semantic'
            semantic_dir.mkdir(exist_ok=True)
            Image.fromarray(data).save(
                semantic_dir / f"{base_name}.png"
            )

    The framework will automatically detect and validate these methods
    during BSSSaver initialization.
    """

    def __init__(self, raw_data_root: str, logger: Optional[logging.Logger] = None):
        """Initialize dataset.

        Args:
            raw_data_root: Path to raw dataset directory
            logger:        Optional logger instance
        """
        self.raw_data_root = Path(raw_data_root)
        self.logger = logger

    @abstractmethod
    def get_scenes(self) -> List[str]:
        """Get list of scene identifiers in the dataset.

        Returns:
            List of scene names/identifiers

        Note:
            Scene identifiers should be filesystem-safe strings.
        """
        pass

    @abstractmethod
    def get_frame_list(self, scene: str) -> List[int]:
        """Get list of frame IDs for a specific scene.

        Args:
            scene: Scene identifier

        Returns:
            List of frame IDs (integers)

        Note:
            Frame IDs are used as indices and don't need to be consecutive.
        """
        pass

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        """Load data for a single frame.

        Args:
            scene: Scene identifier
            frame_id: Frame ID (position in the post-sampling frame list)

        Returns:
            Dictionary containing frame data with the following keys:
            - 'rgb' (np.ndarray): HxWx3 RGB image, uint8, 0-255 range (REQUIRED)
            - 'depth' (np.ndarray, optional): HxW depth map, float32, meters
            - 'mask' (np.ndarray, optional): HxW boolean mask (True for valid)
            - 'pose' (np.ndarray or None, optional): 4x4 C2W transformation matrix.
              Return None if no GT pose is available for this frame; the framework
              writes a NaN row in traj.txt.
            - 'intrinsics' (np.ndarray, optional): [fx, fy, cx, cy] or 3x3 matrix
            - Additional custom keys for extended data types

        Note:
            - 'rgb' is the only REQUIRED field
            - 'pose' may be None or omitted if no ground truth pose is available
            - Custom keys require implementing corresponding __save_{key}_file__ methods
            - Frame index in BSS is assigned by position in the post-sampling list
        """
        return {}

    def load_global_data(self, scene: str) -> Dict[str, Any]:
        """Load global scene-level data.

        Args:
            scene: Scene identifier

        Returns:
            Dictionary containing global data with optional keys:
            - 'points' (np.ndarray): Nx3 or Nx6 point cloud (xyz or xyz+rgb)
            - Additional custom keys for extended data types

        Note:
            Return empty dict {} if no global data is available.
            Custom keys require implementing corresponding __save_{key}_file__ methods.
        """
        return {}

    @staticmethod
    def evaluate_pointcloud(gt_loader, pred_loader,
                            logger,
                            options: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Optional method for custom point cloud evaluation.

        Args:
            gt_loader:   BSSLoader for the ground truth directory (no resize_context)
            pred_loader: BSSLoader for the method output directory (no resize_context)
            logger:      Logger for printing evaluation information
            options:     Optional dataset-specific evaluation options from config

        Returns:
            Dictionary containing:
            - 'gt_points': Nx3 or Nx6 ground truth point cloud (with distance appended)
            - 'pred_points': Mx3 or Mx6 predicted point cloud (with distance appended)
            - 'thresholds': list of distance thresholds used
            - Additional metric keys

        Note:
            This method is called during evaluation if point cloud evaluation
            is enabled in the configuration. Return None to skip evaluation.

            The options parameter receives dataset-specific settings from
            evaluation.points.options in the config.
        """
        return None

    def apply_sampling(
        self,
        frames: List[int],
        sampling_config: Dict[str, Any]
    ) -> List[int]:
        """Apply sampling strategy to frame list.

        Args:
            frames: Original frame ID list (integers)
            sampling_config: Sampling configuration dictionary with keys:
                - strategy: 'sequence' or 'random' (default: 'sequence')
                - For sequence:
                    - start_idx: Start index (default: 0)
                    - end_idx: End index, -1 for all (default: -1)
                    - stride: Step size (default: 1)
                    - num_frames: Target number for uniform sampling (optional)
                - For random:
                    - num_frames: Number of frames to sample (required)
                    - seed: Random seed for reproducibility (default: 42)

        Returns:
            Sampled frame ID list

        Note:
            If sampling_config is empty or None, returns all frames.

        Example:
            # Uniform sampling to 100 frames
            config = {'strategy': 'sequence', 'num_frames': 100}
            sampled = dataset.apply_sampling(frames, config)

            # Every 10th frame
            config = {'strategy': 'sequence', 'stride': 10}
            sampled = dataset.apply_sampling(frames, config)

            # Random 50 frames
            config = {'strategy': 'random', 'num_frames': 50, 'seed': 42}
            sampled = dataset.apply_sampling(frames, config)
        """
        if not sampling_config:
            return frames

        strategy = sampling_config.get('strategy', 'sequence')

        if strategy == 'sequence':
            return self._apply_sequence_sampling(frames, sampling_config)
        elif strategy == 'random':
            return self._apply_random_sampling(frames, sampling_config)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def _apply_sequence_sampling(
        self,
        frames: List[int],
        config: Dict[str, Any]
    ) -> List[int]:
        """Apply sequence (stride or uniform) sampling.

        Args:
            frames: Original frame list
            config: Sequence sampling configuration

        Returns:
            Sampled frame list
        """
        start_idx = config.get('start_idx', 0)
        end_idx = config.get('end_idx', -1)
        stride = config.get('stride', 1)
        num_frames = config.get('num_frames', None)

        # Normalize end_idx
        if end_idx == -1:
            end_idx = len(frames)
        else:
            end_idx = min(end_idx, len(frames))

        # Validate range
        if start_idx >= end_idx:
            raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}")

        # If num_frames specified, use uniform sampling
        if num_frames is not None:
            frame_range = end_idx - start_idx
            if num_frames >= frame_range:
                # Return all frames in range
                sampled = frames[start_idx:end_idx]
            else:
                # Use linspace for uniform sampling
                indices = np.linspace(start_idx, end_idx - 1, num_frames, dtype=int)
                sampled = [frames[i] for i in indices]
        else:
            # Use stride sampling
            sampled = frames[start_idx:end_idx:stride]

        return sampled

    def _apply_random_sampling(
        self,
        frames: List[int],
        config: Dict[str, Any]
    ) -> List[int]:
        """Apply random sampling.

        Args:
            frames: Original frame list
            config: Random sampling configuration

        Returns:
            Sampled frame list (sorted)
        """
        num_frames = config.get('num_frames', 10)
        seed = config.get('seed', 42)

        if num_frames > len(frames):
            num_frames = len(frames)

        # Random sampling with fixed seed
        random.seed(seed)
        indices = list(range(len(frames)))
        random.shuffle(indices)
        indices = sorted(indices[:num_frames])

        # Return sorted frame IDs
        sampled = [frames[i] for i in sorted(indices)]
        return sampled
