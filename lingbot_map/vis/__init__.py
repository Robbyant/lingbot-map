# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GCT Visualization Module

This module provides visualization utilities for 3D reconstruction results:
- PointCloudViewer: Interactive point cloud viewer with camera visualization
- viser_wrapper: Quick visualization wrapper for predictions
- predictions_to_glb: Export predictions to GLB 3D format
- Colorization and utility functions

Usage:
    from lingbot_map.vis import PointCloudViewer, viser_wrapper, predictions_to_glb

    # Interactive visualization
    viewer = PointCloudViewer(pred_dict=predictions, port=8080)
    viewer.run()

    # Quick visualization
    viser_wrapper(predictions, port=8080)

    # Export to GLB
    scene = predictions_to_glb(predictions)
    scene.export("output.glb")
"""

try:
    from lingbot_map.vis.point_cloud_viewer import PointCloudViewer
    from lingbot_map.vis.viser_wrapper import viser_wrapper
except ImportError:
    PointCloudViewer = None
    viser_wrapper = None
from lingbot_map.vis.utils import CameraState, colorize, colorize_np, get_vertical_colorbar
from lingbot_map.vis.sky_segmentation import (
    apply_sky_segmentation,
    download_skyseg_model,
    load_or_create_sky_masks,
    segment_sky,
)
from lingbot_map.vis.glb_export import predictions_to_glb

__all__ = [
    # Main viewer
    "PointCloudViewer",
    # Quick visualization
    "viser_wrapper",
    # GLB export
    "predictions_to_glb",
    # Utilities
    "CameraState",
    "colorize",
    "colorize_np",
    "get_vertical_colorbar",
    # Sky segmentation
    "apply_sky_segmentation",
    "segment_sky",
    "download_skyseg_model",
    "load_or_create_sky_masks",
]
