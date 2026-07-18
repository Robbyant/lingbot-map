"""Base method interface for the benchmark framework.

All methods must inherit from BaseMethod and implement the required methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

from benchmark.geometry.resize import ResizeContext
from benchmark.core.storage import BSSArtifact


class BaseMethod(ABC):
    """Abstract base class for all methods.

    This class defines the interface that all method implementations must follow.
    Methods are responsible for processing scenes and returning results in the
    standardized format.
    """

    def __init__(
        self,
        area_budget: Optional[int] = None,
        align: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize method.

        Args:
            area_budget:  Maximum pixel area for 'area_budget' resize mode.
                          When provided, BSSLoader will downscale images so that
                          width * height <= area_budget (with alignment applied).
                          When None, images are loaded at native GT resolution.
            align:        Alignment requirement (output dims must be multiples of this)
            logger:       Optional logger instance
        """
        
        if area_budget is not None:
            self.resize_context = ResizeContext(
                mode='area_budget', align=align, area_budget=area_budget
            )
        else:
            self.resize_context = ResizeContext(mode='none')
        
        self.align = align
        self.logger = logger

    @abstractmethod
    def process_scene(self, gt_artifact: BSSArtifact) -> Dict[str, Any]:
        """Process a scene and return results.

        Args:
            gt_artifact: BSSArtifact for the ground truth directory

        Returns:
            Dictionary with two keys:
            - 'frame': Dict of per-frame outputs
                - 'rgb': List[np.ndarray] - HxWx3 RGB images, uint8 (REQUIRED)
                - 'depth': List[np.ndarray] - HxW depth maps, float32, meters (optional)
                - 'pose': List[np.ndarray] - 4x4 C2W matrices (optional)
                - 'intrinsics': List[np.ndarray] - [fx,fy,cx,cy] arrays (optional)
                - 'confidence': List[np.ndarray] - HxW confidence maps (optional)
                - Custom keys for extended data types (optional)
            - 'global': Dict of global outputs (optional)
                - 'points': np.ndarray - Nx3 or Nx6 point cloud (optional)
                - Custom keys for extended data types (optional)

        Note:
            - 'frame' key is REQUIRED
            - 'frame['rgb']' is REQUIRED (typically copied from input)
            - All list lengths in 'frame' must match the number of input frames
            - 'global' key can be empty dict {} if no global outputs
            - Custom keys require implementing corresponding __save_{key}_file__ methods

        Example:
            {
                'frame': {
                    'rgb': [rgb1, rgb2, ...],        # Required
                    'depth': [depth1, depth2, ...],  # Optional
                    'pose': [pose1, pose2, ...]      # Optional
                },
                'global': {
                    'points': merged_pointcloud  # Optional
                }
            }
        """
        pass

    def validate_output(self, output: Dict[str, Any], num_frames: int) -> None:
        """Validate method output format.

        Args:
            output:     Output dictionary from process_scene()
            num_frames: Expected number of GT input frames

        Raises:
            ValueError: If output format is invalid
        """
        if 'frame' not in output:
            raise ValueError("Output must contain 'frame' key")

        frame_data = output['frame']
        if not isinstance(frame_data, dict):
            raise ValueError("'frame' must be a dictionary")

        if 'rgb' not in frame_data:
            raise ValueError("'frame' must contain 'rgb' key")

        rgb_list = frame_data['rgb']
        if not isinstance(rgb_list, list):
            raise ValueError("'frame['rgb']' must be a list")

        # For sparse SLAM outputs, frame_indices specifies which GT frames were
        # processed, so the output length K may differ from num_frames N.
        frame_indices = output.get('frame_indices')
        if frame_indices is not None:
            expected_len = len(frame_indices)
            if len(rgb_list) != expected_len:
                raise ValueError(
                    f"'frame['rgb']' length ({len(rgb_list)}) doesn't match "
                    f"len(frame_indices) ({expected_len})"
                )
        else:
            if len(rgb_list) != num_frames:
                raise ValueError(
                    f"'frame['rgb']' length ({len(rgb_list)}) doesn't match "
                    f"expected ({num_frames})"
                )
            expected_len = num_frames

        for key, value in frame_data.items():
            if isinstance(value, list) and len(value) != expected_len:
                raise ValueError(
                    f"'frame['{key}']' length ({len(value)}) doesn't match "
                    f"expected ({expected_len})"
                )

        if 'global' in output:
            global_data = output['global']
            if not isinstance(global_data, dict):
                raise ValueError("'global' must be a dictionary")
