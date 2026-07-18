"""Interactive 3D viewer for benchmark results using viser.

This viewer is designed for the benchmark project and uses the BSS
(Benchmark Storage Structure) format.
"""

import argparse
import asyncio
import json
import logging
import shutil
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import viser
import viser.transforms as tf
from PIL import Image

from benchmark.core.loader import BSSLoader
from benchmark.core.storage import BSSArtifact, BSSManager


# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_BASE_POINT_SIZE = 0.001  # Base point size in meters (will be scaled by GUI)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ViewerData:
    """Represents a single dataset/scene/method combination for viewing."""
    dataset: str
    scene: str
    method: str  # "gt" or method name
    artifact: BSSArtifact  # BSS directory for this dataset/scene/method

    @property
    def display_name(self) -> str:
        """Return a display name for GUI."""
        return f"{self.dataset}/{self.scene}/{self.method}"


@dataclass
class GlobalContext:
    """Global viewer configuration."""
    workspace: Path
    available_data: List[ViewerData]
    # Pre-built nested structure: dataset -> scene -> [methods] (gt always first).
    # Built once at startup from the workspace scan so dropdown updates are O(1)
    # lookups instead of repeated list comprehensions over available_data.
    structure: Dict[str, Dict[str, List[str]]]
    default_data_idx: int = 0
    temporal_subsample: int = 1
    spatial_subsample: int = 8


_global_context: Optional[GlobalContext] = None
_global_context_verbose: bool = False
_camera_clipboard: Optional[Dict[str, np.ndarray]] = None  # Shared camera clipboard across all clients


# =============================================================================
# Point Cloud Cache
# =============================================================================

class SceneCache:
    """Manages persistent scene data caching (trajectory, point clouds) for faster loading."""

    def __init__(self, data_dir: Path, spatial_subsample: int, workspace_root: Path, remove_sky: bool = False):
        """Initialize cache manager.

        Args:
            data_dir: Path to method directory (e.g., workspace/dataset/scene/method)
            spatial_subsample: Spatial subsampling factor (used as cache key)
            workspace_root: Path to workspace root directory
            remove_sky: Whether sky removal is enabled (determines which cache file to use)
        """
        self.data_dir = data_dir
        self.spatial_subsample = spatial_subsample

        # Compute relative path from workspace
        try:
            rel_path = data_dir.relative_to(workspace_root)
        except ValueError:
            # Fallback if data_dir is not under workspace_root
            rel_path = Path(data_dir.name)

        # Store cache in workspace/viewer_cache/{dataset}/{scene}/{method}/
        self.cache_dir = workspace_root / "viewer_cache" / rel_path
        sky_suffix = "_sky" if remove_sky else ""
        self.cache_file = self.cache_dir / f"scene_data_{spatial_subsample}{sky_suffix}.npz"
        self.params_file = self.cache_dir / f"cache_params_{spatial_subsample}{sky_suffix}.json"
        # RGB thumbnail cache (independent of sky removal and spatial_subsample)
        self.rgb_cache_file = self.cache_dir / "rgb_thumbnails.npz"
        self.rgb_params_file = self.cache_dir / "rgb_params.json"
        # Sky mask cache (independent of spatial_subsample)
        self.sky_masks_file = self.cache_dir / "sky_masks.npz"

    def is_valid(self, num_frames: int, image_width: int, image_height: int, has_transform: bool) -> bool:
        """Check if cache exists and parameters match.

        Args:
            num_frames:    Number of frames in the scene
            image_width:   Width of RGB images
            image_height:  Height of RGB images
            has_transform: Whether traj_transform.txt exists

        Returns:
            True if cache is valid and can be used
        """
        if not self.cache_file.exists() or not self.params_file.exists():
            return False

        try:
            with open(self.params_file, 'r') as f:
                params = json.load(f)

            # Check parameters
            if params.get('spatial_subsample') != self.spatial_subsample:
                return False
            if params.get('image_width') != image_width:
                return False
            if params.get('image_height') != image_height:
                return False
            if params.get('num_frames') != num_frames:
                return False
            # Check if alignment state matches
            if params.get('is_aligned', False) != has_transform:
                return False

            return True
        except Exception as e:
            logger.warning(f"Cache validation failed: {e}")
            return False

    def load(self) -> tuple[Dict[str, np.ndarray], List[Dict[str, np.ndarray]], bool]:
        """Load cached scene data.

        Returns:
            Tuple of (trajectory_dict, point_clouds_list, is_aligned)
        """
        try:
            data = np.load(self.cache_file, allow_pickle=True)
            
            # Load trajectory
            traj_dict = {}
            num_frames = int(data.get('num_frames', 0))
            for i in range(num_frames):
                if f'c2w_{i}' in data:
                    c2w = data[f'c2w_{i}']
                    traj_dict[i] = c2w
            
            # Load point clouds
            point_clouds = []
            for i in range(num_frames):
                pcd_dict = {}
                if f'xyz_{i}' in data:
                    pcd_dict['xyz'] = data[f'xyz_{i}']
                if f'rgb_{i}' in data:
                    pcd_dict['rgb'] = data[f'rgb_{i}']
                point_clouds.append(pcd_dict)

            # Load alignment status
            with open(self.params_file, 'r') as f:
                params = json.load(f)
            is_aligned = params.get('is_aligned', False)

            logger.info(f"Loaded {len(traj_dict)} poses and {len(point_clouds)} point clouds from cache (aligned={is_aligned})")
            return traj_dict, point_clouds, is_aligned
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return {}, [], False

    def save(
        self,
        traj_dict: Dict[int, np.ndarray],
        point_clouds: List[Dict[str, np.ndarray]],
        num_frames: int,
        image_width: int,
        image_height: int,
        is_aligned: bool
    ):
        """Save scene data to cache.

        Args:
            traj_dict:    Dictionary mapping frame index (int) to C2W matrices
            point_clouds: List of point cloud dictionaries
            num_frames:   Total number of frames
            image_width:  Width of RGB images
            image_height: Height of RGB images
            is_aligned:   Whether data has been aligned with Umeyama transform
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Save trajectory and point clouds to npz
            save_dict = {'num_frames': num_frames}

            # Save trajectory (only valid frames)
            for frame_idx, c2w in traj_dict.items():
                save_dict[f'c2w_{frame_idx}'] = c2w

            # Save point clouds
            for i, pcd_dict in enumerate(point_clouds):
                if 'xyz' in pcd_dict:
                    save_dict[f'xyz_{i}'] = pcd_dict['xyz']
                if 'rgb' in pcd_dict:
                    save_dict[f'rgb_{i}'] = pcd_dict['rgb']

            np.savez_compressed(self.cache_file, **save_dict)

            # Save parameters
            params = {
                'spatial_subsample': self.spatial_subsample,
                'num_frames': num_frames,
                'image_width': image_width,
                'image_height': image_height,
                'is_aligned': is_aligned,
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%S')
            }

            with open(self.params_file, 'w') as f:
                json.dump(params, f, indent=2)

            logger.info(f"Saved {len(traj_dict)} poses and {len(point_clouds)} point clouds to cache (aligned={is_aligned})")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    # ------------------------------------------------------------------
    # RGB thumbnail cache
    # ------------------------------------------------------------------

    def is_valid_rgb(self, num_frames: int, image_width: int, image_height: int) -> bool:
        """Check if the RGB thumbnail cache exists and matches current parameters."""
        if not self.rgb_cache_file.exists() or not self.rgb_params_file.exists():
            return False
        try:
            with open(self.rgb_params_file, 'r') as f:
                params = json.load(f)
            return (params.get('num_frames') == num_frames and
                    params.get('image_width') == image_width and
                    params.get('image_height') == image_height)
        except Exception:
            return False

    def save_rgb_thumbnails(self, thumbnails: List[np.ndarray], image_width: int, image_height: int):
        """Save RGB thumbnails to cache as a compressed npz archive."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            save_dict: Dict[str, np.ndarray] = {'num_thumbnails': np.array(len(thumbnails))}
            for i, thumb in enumerate(thumbnails):
                save_dict[f'thumb_{i}'] = thumb
            np.savez_compressed(self.rgb_cache_file, **save_dict)
            params = {
                'num_frames': len(thumbnails),
                'image_width': image_width,
                'image_height': image_height,
            }
            with open(self.rgb_params_file, 'w') as f:
                json.dump(params, f, indent=2)
            logger.info(f"Saved {len(thumbnails)} RGB thumbnails to cache")
        except Exception as e:
            logger.error(f"Failed to save RGB thumbnails: {e}")

    def load_rgb_thumbnails(self) -> Optional[List[np.ndarray]]:
        """Load RGB thumbnails from the npz cache."""
        try:
            data = np.load(self.rgb_cache_file)
            n = int(data['num_thumbnails'])
            thumbnails = [data[f'thumb_{i}'] for i in range(n)]
            logger.info(f"Loaded {n} RGB thumbnails from cache")
            return thumbnails
        except Exception as e:
            logger.error(f"Failed to load RGB thumbnails: {e}")
            return None

    # ------------------------------------------------------------------
    # Sky mask cache
    # ------------------------------------------------------------------

    def has_sky_masks(self) -> bool:
        """Return True if a sky mask cache file already exists."""
        return self.sky_masks_file.exists()

    def save_sky_masks(self, masks: List[np.ndarray]):
        """Save per-frame binary sky masks (True = sky pixel) to cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            save_dict: Dict[str, np.ndarray] = {'num_masks': np.array(len(masks))}
            for i, mask in enumerate(masks):
                save_dict[f'mask_{i}'] = mask.astype(np.uint8)
            np.savez_compressed(self.sky_masks_file, **save_dict)
            logger.info(f"Saved {len(masks)} sky masks to cache")
        except Exception as e:
            logger.error(f"Failed to save sky masks: {e}")

    def load_sky_masks(self) -> Optional[List[np.ndarray]]:
        """Load per-frame binary sky masks from cache."""
        try:
            data = np.load(self.sky_masks_file)
            n = int(data['num_masks'])
            masks = [data[f'mask_{i}'].astype(bool) for i in range(n)]
            logger.info(f"Loaded {n} sky masks from cache")
            return masks
        except Exception as e:
            logger.error(f"Failed to load sky masks: {e}")
            return None


# =============================================================================
# Utility Functions
# =============================================================================

def scan_workspace(
    workspace: Path,
) -> Tuple[List[ViewerData], Dict[str, Dict[str, List[str]]]]:
    """Scan workspace and return all valid BSS entries plus a pre-built structure.

    Validity is determined by the presence of a .complete.json file in the
    method directory (written by CompletionManager after a successful run/prepare).

    Returns:
        available_data: Flat list of ViewerData (used for data_dir lookups).
        structure: Nested dict  dataset -> scene -> sorted method list,
                   with 'gt' always placed first when present.
    """
    available_data: List[ViewerData] = []
    structure: Dict[str, Dict[str, List[str]]] = {}

    if not workspace.exists():
        logger.warning(f"Workspace directory not found: {workspace}")
        return available_data, structure

    skip_dirs = {'logs', 'report', 'viewer_cache'}

    for dataset_dir in sorted(workspace.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in skip_dirs:
            continue
        dataset_name = dataset_dir.name

        for scene_dir in sorted(dataset_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_name = scene_dir.name

            for method_dir in sorted(scene_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method_name = method_dir.name

                # Only include directories that have been successfully completed.
                if not (method_dir / '.complete.json').exists():
                    continue

                available_data.append(ViewerData(
                    dataset=dataset_name,
                    scene=scene_name,
                    method=method_name,
                    artifact=BSSArtifact(method_dir),
                ))

                # Build nested structure.
                dataset_entry = structure.setdefault(dataset_name, {})
                dataset_entry.setdefault(scene_name, []).append(method_name)

    # Ensure 'gt' is always the first method in every scene.
    for dataset_entry in structure.values():
        for scene_methods in dataset_entry.values():
            if 'gt' in scene_methods:
                scene_methods.remove('gt')
                scene_methods.insert(0, 'gt')

    total = sum(
        len(methods)
        for dataset_entry in structure.values()
        for methods in dataset_entry.values()
    )
    logger.info(
        f"Found {total} valid entries across "
        f"{len(structure)} dataset(s) in workspace"
    )
    return available_data, structure


def get_host_ip() -> str:
    """Get the host IP address for the viser server."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 1))
            internal_ip = s.getsockname()[0]
        except Exception:
            internal_ip = "127.0.0.1"
    return internal_ip


# =============================================================================
# Scene Handling
# =============================================================================

@dataclass
class SceneFrameHandle:
    """Handles for a single frame in the scene."""
    frame_handle: viser.FrameHandle
    frustum_handle: viser.CameraFrustumHandle
    pcd_handle: Optional[viser.PointCloudHandle] = None
    frame_idx: int = 0

    def __post_init__(self):
        self.visible = False

    @property
    def visible(self) -> bool:
        return self.frame_handle.visible

    @visible.setter
    def visible(self, value: bool):
        self.frame_handle.visible = value
        self.frustum_handle.visible = value
        if self.pcd_handle is not None:
            self.pcd_handle.visible = value


# =============================================================================
# Client Closures
# =============================================================================

class ClientClosures:
    """Handles client-specific UI and rendering logic."""

    def __init__(self, client: viser.ClientHandle):
        self.client = client

        async def _run():
            try:
                await self.run()
            except asyncio.CancelledError:
                pass
            finally:
                self.cleanup()

        self.task = asyncio.create_task(_run())

        # GUI handles
        self.gui_dataset: Optional[viser.GuiDropdownHandle] = None
        self.gui_scene: Optional[viser.GuiDropdownHandle] = None
        self.gui_method: Optional[viser.GuiDropdownHandle] = None
        self.gui_load_button: Optional[viser.GuiButtonHandle] = None
        self.gui_clear_cache: Optional[viser.GuiButtonHandle] = None
        self.gui_playback_handle: Optional[viser.GuiFolderHandle] = None
        self.gui_timestep: Optional[viser.GuiSliderHandle] = None
        self.gui_framerate: Optional[viser.GuiSliderHandle] = None
        self.gui_history_cameras: Optional[viser.GuiSliderHandle] = None
        self.gui_history_points: Optional[viser.GuiSliderHandle] = None
        self.gui_play_pause: Optional[viser.GuiButtonHandle] = None
        self.gui_frame_display: Optional[viser.GuiImageHandle] = None
        self.gui_t_sub: Optional[viser.GuiSliderHandle] = None
        self.gui_s_sub: Optional[viser.GuiSliderHandle] = None
        self.gui_point_scale: Optional[viser.GuiSliderHandle] = None
        self.gui_point_downsample: Optional[viser.GuiSliderHandle] = None
        self.gui_frustum_size: Optional[viser.GuiSliderHandle] = None
        self.gui_show_trajectory: Optional[viser.GuiHandle] = None
        self.gui_show_cameras: Optional[viser.GuiHandle] = None
        self.gui_remove_sky: Optional[viser.GuiCheckboxHandle] = None
        self.gui_fov: Optional[viser.GuiSliderHandle] = None
        self.gui_alignment_status: Optional[viser.GuiHandle] = None
        self.gui_confidence_threshold: Optional[viser.GuiSliderHandle] = None


        # Scene state
        self.is_aligned: bool = False
        self.scene_frame_handles: List[SceneFrameHandle] = []
        self.trajectory_handles: List[viser.SceneNodeHandle] = []  # Multiple segments
        self.current_displayed_timestep: int = 0
        self.is_playing: bool = True
        self.first_frame_pose: Optional[np.ndarray] = None
        self.scene_min = np.array([np.inf, np.inf, np.inf])
        self.scene_max = np.array([-np.inf, -np.inf, -np.inf])
        self.base_point_size: float = DEFAULT_BASE_POINT_SIZE  # Will be computed adaptively
        self.rgb_thumbnails: List[np.ndarray] = []
        self.cached_point_clouds: List[Dict[str, np.ndarray]] = []
        self.sky_masks: Optional[List[np.ndarray]] = None  # Per-frame binary sky masks
        self.loop_playback: bool = True
        self.has_confidence_data: bool = False  # Track if current method has confidence data
        self.scene_loaded: bool = False  # Two-state toggle: False=config mode, True=viewing mode
        self._load_task: Optional[asyncio.Task] = None  # Task for debounced loading
        self._cancel_loading: bool = False  # Flag to interrupt _rebuild_scene
        self.use_global_pointcloud: bool = False  # Track if using global point cloud
        self.global_pointcloud_handle: Optional[viser.PointCloudHandle] = None  # Handle for global point cloud


        # Camera clipboard
        self.gui_camera_clipboard_folder: Optional[viser.GuiFolderHandle] = None

    async def stop(self):
        self.task.cancel()
        await self.task

    async def run(self):
        logger.info(f"Client {self.client.client_id} connected")

        structure = self.global_context().structure
        if not structure:
            logger.error("No data available to display")
            return

        # Build dataset/scene/method options from pre-built structure.
        datasets = sorted(structure.keys())

        with self.client.gui.add_folder("Data Selection"):
            self.gui_dataset = self.client.gui.add_dropdown(
                "Dataset",
                options=datasets,
                initial_value=datasets[0] if datasets else ""
            )

            # Initial scene list
            scenes = self._get_scenes_for_dataset(self.gui_dataset.value)
            self.gui_scene = self.client.gui.add_dropdown(
                "Scene",
                options=scenes,
                initial_value=scenes[0] if scenes else ""
            )

            # Initial method list
            methods = self._get_methods_for_dataset_scene(self.gui_dataset.value, self.gui_scene.value)
            self.gui_method = self.client.gui.add_dropdown(
                "Method",
                options=methods,
                initial_value=methods[0] if methods else ""
            )

            # Alignment status display
            self.gui_alignment_status = self.client.gui.add_markdown("")

            # Remove Sky option — takes effect on next Load click
            self.gui_remove_sky = self.client.gui.add_checkbox(
                "Remove Sky",
                initial_value=False,
                hint="Apply sky segmentation to remove sky pixels from point cloud (result is cached after first run)"
            )

            # Load button — data is loaded only when this is clicked
            self.gui_load_button = self.client.gui.add_button(
                "Load",
                hint="Load and visualize the selected dataset / scene / method"
            )

            @self.gui_load_button.on_click
            async def _(_) -> None:
                if self._load_task is not None and not self._load_task.done():
                    # Cancel in-progress loading
                    self._cancel_loading = True
                    self._load_task.cancel()
                else:
                    await self.on_data_update(None)

            # Cache management button (disabled until a scene with cache is selected)
            self.gui_clear_cache = self.client.gui.add_button(
                "Clear Cache",
                hint="Delete cached scene data for current data source",
                disabled=True,
            )

            @self.gui_clear_cache.on_click
            async def _(_):
                await self._clear_cache()

            # Set initial button state based on whether cache exists for default selection
            self._update_clear_cache_button()

            # Update callbacks — cascade dropdown options + return to config mode
            @self.gui_dataset.on_update
            async def _(_) -> None:
                scenes = self._get_scenes_for_dataset(self.gui_dataset.value)
                current_scene = self.gui_scene.value
                self.gui_scene.options = scenes
                if scenes:
                    new_scene = current_scene if current_scene in scenes else scenes[0]
                    self.gui_scene.value = new_scene
                else:
                    new_scene = ""

                methods = self._get_methods_for_dataset_scene(self.gui_dataset.value, new_scene)
                current_method = self.gui_method.value
                self.gui_method.options = methods
                if methods:
                    self.gui_method.value = current_method if current_method in methods else methods[0]

            @self.gui_scene.on_update
            async def _(_) -> None:
                methods = self._get_methods_for_dataset_scene(self.gui_dataset.value, self.gui_scene.value)
                current_method = self.gui_method.value
                self.gui_method.options = methods
                if methods:
                    self.gui_method.value = current_method if current_method in methods else methods[0]


        with self.client.gui.add_folder("Sampling", expand_by_default=False):
            self.gui_t_sub = self.client.gui.add_slider(
                "Temporal", min=1, max=250, step=1,
                initial_value=self.global_context().temporal_subsample
            )
            self.gui_s_sub = self.client.gui.add_slider(
                "Spatial", min=1, max=32, step=1,
                initial_value=self.global_context().spatial_subsample
            )

        with self.client.gui.add_folder("Scene", expand_by_default=False):
            self.gui_confidence_threshold = self.client.gui.add_slider(
                "Confidence Threshold",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.3,
                hint="Percentile threshold: keep points with top X% confidence (disabled if no confidence data)",
                disabled=False  # Will be updated when data loads
            )

            self.gui_point_scale = self.client.gui.add_slider(
                "Point Scale (log)",
                min=-1.0,
                max=1.0,
                step=0.05,
                initial_value=0.0,
                hint="Logarithmic scale: 10^value (0.1x to 10x)"
            )

            @self.gui_point_scale.on_update
            async def _(_) -> None:
                actual_size = self.base_point_size * (10 ** self.gui_point_scale.value)
                # Update per-frame point clouds
                for frame_node in self.scene_frame_handles:
                    if frame_node.pcd_handle is not None:
                        frame_node.pcd_handle.point_size = actual_size
                # Update global point cloud
                if self.global_pointcloud_handle is not None:
                    self.global_pointcloud_handle.point_size = actual_size

            self.gui_point_downsample = self.client.gui.add_slider(
                "Downsample",
                min=1,
                max=16,
                step=1,
                initial_value=1,
                hint="Additional downsampling on cached points (1=use all)"
            )

            self.gui_frustum_size = self.client.gui.add_slider(
                "Frustum Size", min=0.01, max=1.0, step=0.01, initial_value=0.03
            )

            @self.gui_frustum_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    frame_node.frustum_handle.scale = self.gui_frustum_size.value
                    frame_node.frame_handle.axes_length = self.gui_frustum_size.value / 3
                    frame_node.frame_handle.axes_radius = self.gui_frustum_size.value / 30

            self.gui_show_trajectory = self.client.gui.add_checkbox(
                "Trajectory",
                initial_value=False,
                hint="Display camera trajectory path"
            )

            @self.gui_show_trajectory.on_update
            async def _(_) -> None:
                self._update_trajectory_visibility()

            self.gui_show_cameras = self.client.gui.add_checkbox(
                "Cameras",
                initial_value=True,
                hint="Display camera frustums and frames"
            )

            @self.gui_show_cameras.on_update
            async def _(_) -> None:
                self._update_cameras_visibility()

            self.gui_fov = self.client.gui.add_slider("FoV", min=30.0, max=120.0, step=1.0, initial_value=60.0)

            @self.gui_fov.on_update
            async def _(_) -> None:
                self.client.camera.fov = np.deg2rad(self.gui_fov.value)

        # Camera Clipboard (top-level)
        self.gui_camera_clipboard_folder = self.client.gui.add_folder("Camera Clipboard", expand_by_default=False)
        with self.gui_camera_clipboard_folder:
            gui_cam_actions = self.client.gui.add_button_group(
                "",
                options=["Copy", "Paste"]
            )

            @gui_cam_actions.on_click
            def _(event: viser.GuiEvent) -> None:
                if gui_cam_actions.value == "Copy":
                    self._copy_camera_view()
                elif gui_cam_actions.value == "Paste":
                    self._paste_camera_view()

        # No initial load — user must select data and click the Load button

        # Playback loop
        while True:
            if self.is_playing and self.gui_framerate is not None and self.gui_framerate.value > 0:
                max_frame = len(self.scene_frame_handles) - 1
                if self.gui_timestep.value >= max_frame:
                    if self.loop_playback:
                        self.gui_timestep.value = 0
                    else:
                        self.is_playing = False
                        self._rebuild_playback_gui()
                else:
                    self._incr_timestep()
                await asyncio.sleep(1.0 / self.gui_framerate.value)
            else:
                await asyncio.sleep(1.0)

    def _get_scenes_for_dataset(self, dataset: str) -> List[str]:
        """Return scenes for a dataset (pre-built at startup)."""
        return list(self.global_context().structure.get(dataset, {}).keys())

    def _get_methods_for_dataset_scene(self, dataset: str, scene: str) -> List[str]:
        """Return methods for a dataset/scene (pre-built at startup, gt always first)."""
        return list(self.global_context().structure.get(dataset, {}).get(scene, []))

    def _get_current_data(self) -> Optional[ViewerData]:
        """Get currently selected ViewerData."""
        all_data = self.global_context().available_data
        for data in all_data:
            if (data.dataset == self.gui_dataset.value and
                data.scene == self.gui_scene.value and
                data.method == self.gui_method.value):
                return data
        return None

    def _get_current_cache_dir(self) -> Optional[Path]:
        """Return the cache directory for the current data selection, or None."""
        current_data = self._get_current_data()
        if current_data is None:
            return None
        workspace_root = self.global_context().workspace
        try:
            rel_path = current_data.artifact.root.relative_to(workspace_root)
        except ValueError:
            rel_path = Path(current_data.artifact.root.name)
        return workspace_root / "viewer_cache" / rel_path

    def _update_clear_cache_button(self):
        """Enable/disable the Clear Cache button based on whether a cache exists."""
        if self.gui_clear_cache is None:
            return
        cache_dir = self._get_current_cache_dir()
        self.gui_clear_cache.disabled = (cache_dir is None or not cache_dir.exists())

    def _set_config_mode(self, config_mode: bool):
        """Toggle between config mode (True) and viewing mode (False).

        Config mode:  Load enabled, rebuild params enabled, dropdowns enabled, Clear Cache disabled.
        Viewing mode: Load disabled, rebuild params disabled, dropdowns disabled, Clear Cache enabled.
        """
        self.scene_loaded = not config_mode

        # Load button — label reflects state
        if self.gui_load_button is not None:
            self.gui_load_button.disabled = not config_mode
            self.gui_load_button.label = "Load" if config_mode else "Loaded"

        # Dropdowns — prevent switching during viewing/loading
        for dropdown in [self.gui_dataset, self.gui_scene, self.gui_method]:
            if dropdown is not None:
                dropdown.disabled = not config_mode

        # Rebuild-triggering parameters
        rebuild_controls = [
            self.gui_t_sub,
            self.gui_s_sub,
            self.gui_confidence_threshold,
            self.gui_point_downsample,
            self.gui_remove_sky,
        ]
        for ctrl in rebuild_controls:
            if ctrl is not None:
                ctrl.disabled = not config_mode

        # Clear Cache — enabled in viewing mode (if cache exists)
        if not config_mode:
            self._update_clear_cache_button()
        elif self.gui_clear_cache is not None:
            self.gui_clear_cache.disabled = True

    async def _clear_cache(self):
        """Clear cache for current data source and reset the displayed scene."""
        cache_dir = self._get_current_cache_dir()
        if cache_dir is not None and cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared cache: {cache_dir}")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")

        # Clear displayed scene
        self.client.scene.reset()
        self.scene_frame_handles = []
        self.trajectory_handles = []
        self.cached_point_clouds = []
        self.global_pointcloud_handle = None
        self.rgb_thumbnails = []
        self.sky_masks = None
        self.is_playing = True
        self.scene_min = np.array([np.inf, np.inf, np.inf])
        self.scene_max = np.array([-np.inf, -np.inf, -np.inf])

        # Remove playback GUI
        if self.gui_playback_handle is not None:
            self.gui_playback_handle.remove()
            self.gui_playback_handle = None

        # Switch back to config mode
        self._set_config_mode(True)

    async def on_data_update(self, _):
        """Reload scene data when Load is clicked."""
        # Cancel any existing pending load task to avoid redundant loads
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
            try:
                await self._load_task
            except asyncio.CancelledError:
                pass

        async def _reload_job():
            self._cancel_loading = False
            # Disable all config controls; only Cancel button remains active
            self._set_config_mode(False)
            if self.gui_load_button is not None:
                self.gui_load_button.label = "Cancel"
                self.gui_load_button.disabled = False
            # Clear old scene and playback immediately so user sees feedback
            self.client.scene.reset()
            self.scene_frame_handles = []
            self.trajectory_handles = []
            self.global_pointcloud_handle = None
            if self.gui_playback_handle is not None:
                self.gui_playback_handle.remove()
                self.gui_playback_handle = None
            # Yield to event loop so updates reach the frontend
            await asyncio.sleep(0)
            try:
                await asyncio.to_thread(self._rebuild_scene)
                if self._cancel_loading:
                    # Thread finished but was cancelled — clean up
                    self.client.scene.reset()
                    self.scene_frame_handles = []
                    self.trajectory_handles = []
                    self.global_pointcloud_handle = None
                    self._set_config_mode(True)
                    logger.info("Loading cancelled")
                    return
                self._rebuild_playback_gui()
                self._reset_camera_view()

                # Switch to viewing mode
                self._set_config_mode(False)
            except asyncio.CancelledError:
                # Task was cancelled — return to config mode
                self._set_config_mode(True)
            except Exception:
                # On failure, return to config mode so user can retry
                self._set_config_mode(True)
                raise

        # Create and track the new task
        self._load_task = asyncio.create_task(_reload_job())
        
        try:
            await self._load_task
        except asyncio.CancelledError:
            pass

    def _get_gradient_color(self, progress: float) -> tuple[int, int, int]:
        """Get rainbow gradient color based on progress (0.0 to 1.0).
        
        Color progression: Red -> Orange -> Yellow -> Green -> Cyan -> Blue -> Purple
        """
        # Define key points in the rainbow gradient
        colors = [
            (255, 0, 0),      # Red
            (255, 128, 0),    # Orange
            (255, 255, 0),    # Yellow
            (0, 255, 0),      # Green
            (0, 255, 255),    # Cyan
            (0, 128, 255),    # Blue
            (128, 0, 255),    # Purple
        ]
        
        # Scale progress to color index
        n_colors = len(colors)
        scaled = progress * (n_colors - 1)
        idx = int(scaled)
        
        # Handle edge case
        if idx >= n_colors - 1:
            return colors[-1]
        
        # Interpolate between two adjacent colors
        local_progress = scaled - idx
        c1 = colors[idx]
        c2 = colors[idx + 1]
        
        r = int(c1[0] + (c2[0] - c1[0]) * local_progress)
        g = int(c1[1] + (c2[1] - c1[1]) * local_progress)
        b = int(c1[2] + (c2[2] - c1[2]) * local_progress)
        
        return (r, g, b)

    def _load_trajectory(
        self,
        loader: BSSLoader
    ) -> Dict[int, np.ndarray]:
        """Load trajectory, returns dict mapping frame index (int) to C2W matrices.

        Args:
            loader: BSSLoader instance

        Returns:
            Dictionary mapping frame index to C2W matrices (NaN frames excluded)
        """
        traj_array = loader.load_trajectory()
        if traj_array is None:
            return {}

        result = {}
        for i, pose in enumerate(traj_array):
            if not np.any(np.isnan(pose)):
                result[i] = pose
        return result

    def _load_intrinsics_as_dict(
        self,
        loader: BSSLoader
    ) -> Optional[Dict[int, np.ndarray]]:
        """Load intrinsics, returns dict mapping frame index (int) to [fx,fy,cx,cy].

        Args:
            loader: BSSLoader instance

        Returns:
            Dictionary mapping frame index to [fx, fy, cx, cy] arrays, or None if unavailable
        """
        intr_array = loader.load_intrinsics()
        if intr_array is None:
            return None

        result = {}
        for i, intr in enumerate(intr_array):
            if not np.any(np.isnan(intr)):
                result[i] = intr
        return result if result else None

    def _compute_base_point_size(
        self,
        depth_list: List[Optional[np.ndarray]],
        intrinsics_dict: Dict[int, np.ndarray]
    ) -> float:
        """Compute base point size from average pixel spacing using first frame.

        Args:
            depth_list:     List of depth maps
            intrinsics_dict: Dictionary mapping frame index (int) to [fx,fy,cx,cy]

        Returns:
            Base point size (average pixel spacing at mean depth) * 0.1
        """
        # Use only the first valid depth frame
        first_depth = None
        first_intrinsics = None

        for i, depth in enumerate(depth_list):
            if depth is not None:
                valid = (depth > 0) & np.isfinite(depth)
                if np.any(valid) and i in intrinsics_dict:
                    first_depth = depth
                    first_intrinsics = intrinsics_dict[i]
                    break

        if first_depth is None or first_intrinsics is None:
            return DEFAULT_BASE_POINT_SIZE  # Default fallback

        # Extract valid depths from first frame
        valid = (first_depth > 0) & np.isfinite(first_depth)
        valid_depths = first_depth[valid].flatten()
        mean_depth = np.mean(valid_depths)

        # Extract focal lengths
        if first_intrinsics.shape == (3, 3):
            fx, fy = first_intrinsics[0, 0], first_intrinsics[1, 1]
        else:
            fx, fy = first_intrinsics[0], first_intrinsics[1]
        mean_focal = (fx + fy) / 2

        base_size = mean_depth / mean_focal
        logger.info(f"Computed base point size: {base_size:.6f} (mean_depth={mean_depth:.2f}, mean_focal={mean_focal:.1f})")

        return float(base_size) * 0.1

    def _generate_point_clouds(
        self,
        loader: BSSLoader,
        spatial_subsample: int,
        umeyama_transform: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.0,
        sky_masks: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[Dict[str, np.ndarray]], bool]:
        """Generate point clouds using loader's load_point_cloud_grid or global points.

        Args:
            loader: BSSLoader instance
            spatial_subsample: Spatial subsampling factor
            umeyama_transform: Optional 4x4 Umeyama matrix (with scale)
            confidence_threshold: Percentile threshold (0-1) for filtering by confidence
            sky_masks: Optional per-frame binary masks (True = sky pixel to remove)

        Returns:
            Tuple of (point_clouds_list, use_global_pointcloud)
            - point_clouds_list: List of point cloud dictionaries (per-frame or single global)
            - use_global_pointcloud: True if using global point cloud, False for per-frame
        """
        # Try to load point cloud grid (per-frame) with confidence filtering
        try:
            xyzrgb, mask = loader.load_point_cloud_grid(confidence_threshold=confidence_threshold)
            logger.info("Successfully loaded per-frame point clouds")

            point_clouds = []
            V, H, W = xyzrgb.shape[:3]

            for frame_idx in range(V):
                # Extract frame data with spatial subsampling
                xyz_grid = xyzrgb[frame_idx, ::spatial_subsample, ::spatial_subsample, :3]
                rgb_grid = xyzrgb[frame_idx, ::spatial_subsample, ::spatial_subsample, 3:]
                mask_grid = mask[frame_idx, ::spatial_subsample, ::spatial_subsample]

                # Apply sky mask: remove pixels classified as sky
                if sky_masks is not None and frame_idx < len(sky_masks):
                    sky_mask_full = sky_masks[frame_idx]  # (H, W) bool
                    sky_mask_sub = sky_mask_full[::spatial_subsample, ::spatial_subsample]
                    if sky_mask_sub.shape == mask_grid.shape:
                        mask_grid = mask_grid & ~sky_mask_sub

                # Convert grid to point cloud format
                xyz_flat = xyz_grid[mask_grid]
                rgb_flat = (rgb_grid[mask_grid] * 255).astype(np.uint8)

                pcd_dict = {}
                if len(xyz_flat) > 0:
                    # Apply Umeyama transformation (including scale) to point cloud
                    if umeyama_transform is not None:
                        sR = umeyama_transform[:3, :3]  # (3, 3)
                        t = umeyama_transform[:3, 3]     # (3,)
                        xyz_transformed = (xyz_flat.astype(np.float64) @ sR.T) + t
                        pcd_dict['xyz'] = xyz_transformed
                    else:
                        pcd_dict['xyz'] = xyz_flat

                    pcd_dict['rgb'] = rgb_flat

                point_clouds.append(pcd_dict)

            return point_clouds, False

        except Exception as e:
            logger.warning(f"Failed to load per-frame point cloud grid: {e}")
            logger.info("Attempting to load global point cloud...")

            # Try to load global point cloud as fallback
            try:
                global_points = loader.load_global_points()
                if global_points is None:
                    raise ValueError("Global point cloud file not found")

                logger.info(f"Successfully loaded global point cloud with {len(global_points)} points")

                # Apply spatial subsampling to global point cloud
                global_points = global_points[::spatial_subsample]

                # Parse global points (Nx3 or Nx6)
                if global_points.shape[1] == 6:
                    xyz = global_points[:, :3]
                    rgb = (global_points[:, 3:6] * 255).astype(np.uint8)
                elif global_points.shape[1] == 3:
                    xyz = global_points
                    rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)  # Default gray color
                else:
                    raise ValueError(f"Invalid global point cloud shape: {global_points.shape}")

                # Apply Umeyama transformation if provided
                if umeyama_transform is not None:
                    sR = umeyama_transform[:3, :3]
                    t = umeyama_transform[:3, 3]
                    xyz = (xyz.astype(np.float64) @ sR.T) + t
                
                # Return single point cloud in list format
                pcd_dict = {'xyz': xyz, 'rgb': rgb}
                return [pcd_dict], True
                
            except Exception as e2:
                logger.warning(
                    f"No point cloud data available (per-frame: {e}; global: {e2}). "
                    "Showing trajectory only."
                )
                return [], False

    def _rebuild_scene(self):
        """Rebuild the 3D scene from currently selected data."""
        current_data = self._get_current_data()
        if current_data is None:
            logger.warning("No data selected")
            return

        spatial_subsample: int = self.gui_s_sub.value
        temporal_subsample: int = self.gui_t_sub.value

        logger.info(f"Loading: {current_data.display_name}")

        # rgb_list is loaded lazily — only when actually needed
        rgb_list: Optional[List[np.ndarray]] = None

        # Load data using BSSLoader
        bss_manager = BSSManager(_global_context.workspace)
        method_name = None if current_data.method == 'gt' else current_data.method
        try:
            artifact = bss_manager.get_artifact(current_data.dataset, current_data.scene, method_name)
            loader = BSSLoader(artifact)
            N = loader.get_num_frames()
            traj_dict = self._load_trajectory(loader)
            intrinsics_dict = self._load_intrinsics_as_dict(loader)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return

        # Validate required data
        if traj_dict is None or len(traj_dict) == 0:
            logger.error("No trajectory data found")
            return

        # Fallback to GT intrinsics if not available (e.g., PI3 method)
        if intrinsics_dict is None:
            logger.warning(f"No intrinsics found for {current_data.method}, attempting to load from GT")
            try:
                gt_artifact = bss_manager.get_artifact(current_data.dataset, current_data.scene, None)
                gt_loader = BSSLoader(gt_artifact)
                intrinsics_dict = self._load_intrinsics_as_dict(gt_loader)
                if intrinsics_dict is not None:
                    logger.info(f"Loaded intrinsics from GT for visualization")
                else:
                    # Use default intrinsics based on image dimensions
                    logger.warning("No intrinsics found in GT either, using default intrinsics")
                    metadata = artifact.read_metadata()
                    image_width = metadata.get('image_width', 0)
                    image_height = metadata.get('image_height', 0)
                    
                    if image_width == 0 or image_height == 0:
                        # Load first RGB frame to get dimensions
                        rgb_list = loader.load_rgb_list()
                        if rgb_list and len(rgb_list) > 0:
                            image_height, image_width = rgb_list[0].shape[:2]
                        else:
                            logger.error("Cannot determine image dimensions for default intrinsics")
                            return
                    
                    # Default intrinsics: assume FOV ~= 60 degrees
                    # fx = fy = max(width, height)
                    # cx = width / 2, cy = height / 2
                    fx = fy = float(max(image_width, image_height))
                    cx = image_width / 2.0
                    cy = image_height / 2.0
                    default_intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)
                    
                    # Create intrinsics dict for all frames
                    intrinsics_dict = {i: default_intrinsics.copy() for i in range(N)}
                    logger.warning(f"Using default intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            except Exception as e:
                logger.error(f"Failed to load intrinsics from GT: {e}")
                return

        # Check for Umeyama transformation
        umeyama_transform = loader.load_traj_transform()
        has_transform = umeyama_transform is not None

        # Read image dimensions from metadata (avoids loading all RGB images upfront)
        metadata = artifact.read_metadata()
        image_width = metadata.get('image_width', 0)
        image_height = metadata.get('image_height', 0)

        # Check if confidence data exists
        self.has_confidence_data = artifact.confidence_dir.exists()
        if self.has_confidence_data:
            logger.info(f"Confidence data detected: {artifact.confidence_dir}")
        else:
            logger.info("No confidence data found - confidence filtering disabled")

        # Determine whether sky removal is active
        remove_sky = self.gui_remove_sky.value if self.gui_remove_sky is not None else False

        # Initialize cache (sky-removal creates a separate cache file)
        workspace_root = self.global_context().workspace
        cache = SceneCache(artifact.root, spatial_subsample, workspace_root, remove_sky=remove_sky)

        # ----------------------------------------------------------------
        # Scene data (trajectory + point clouds)
        # ----------------------------------------------------------------
        if cache.is_valid(N, image_width, image_height, has_transform):
            logger.info("Loading scene data from cache...")
            traj_dict, self.cached_point_clouds, self.is_aligned = cache.load()
            self.use_global_pointcloud = False
        else:
            if self._cancel_loading:
                return
            rgb_list = loader.load_rgb_list()
            if rgb_list:
                image_height, image_width = rgb_list[0].shape[:2]
            logger.info("Generating scene data...")

            # Resolve sky masks
            sky_masks: Optional[List[np.ndarray]] = None
            if remove_sky:
                if self._cancel_loading:
                    return
                if cache.has_sky_masks():
                    logger.info("Loading sky masks from cache...")
                    sky_masks = cache.load_sky_masks()
                else:
                    logger.info("Running sky segmentation (result will be cached)...")
                    sky_masks = self._run_sky_segmentation(rgb_list)
                    cache.save_sky_masks(sky_masks)

            if self._cancel_loading:
                return

            # Get confidence threshold
            confidence_threshold = self.gui_confidence_threshold.value if (
                self.gui_confidence_threshold is not None and self.has_confidence_data
            ) else 0.0

            # Generate point clouds (with optional sky filtering)
            try:
                self.cached_point_clouds, self.use_global_pointcloud = self._generate_point_clouds(
                    loader, spatial_subsample,
                    umeyama_transform,
                    confidence_threshold=confidence_threshold,
                    sky_masks=sky_masks,
                )
            except Exception as e:
                logger.warning(f"Point cloud generation failed: {e}. Showing trajectory only.")
                self.cached_point_clouds = []
                self.use_global_pointcloud = False

            # Apply Umeyama transformation to trajectory for camera display
            if umeyama_transform is not None:
                logger.info("Applying Umeyama transform to trajectory")
                traj_dict_transformed = {}
                for frame_idx, c2w in traj_dict.items():
                    c2w_transformed = umeyama_transform @ c2w
                    # Umeyama transform contains scale, so the rotation part is no longer
                    # a pure rotation matrix. Use SVD to extract the closest rotation matrix.
                    R_scaled = c2w_transformed[:3, :3]
                    U, _, Vh = np.linalg.svd(R_scaled)
                    R_pure = U @ Vh
                    # Ensure proper rotation (det = +1, not reflection)
                    if np.linalg.det(R_pure) < 0:
                        R_pure = -R_pure
                    c2w_transformed[:3, :3] = R_pure
                    traj_dict_transformed[frame_idx] = c2w_transformed
                traj_dict = traj_dict_transformed
                self.is_aligned = True
            else:
                self.is_aligned = False

            # Save to cache (traj_dict is now transformed)
            # Note: Don't cache global point cloud (only cache per-frame clouds)
            if not self.use_global_pointcloud:
                cache.save(traj_dict, self.cached_point_clouds, N, image_width, image_height, self.is_aligned)
            else:
                logger.info("Using global point cloud - skipping cache save")

        # Update alignment status in GUI
        if self.gui_alignment_status is not None:
            # Check if this is GT data
            current_data = self._get_current_data()
            is_gt = current_data is not None and current_data.method.lower() == 'gt'

            if is_gt:
                self.gui_alignment_status.content = "**Alignment:** GT"
            elif self.is_aligned:
                self.gui_alignment_status.content = "**Alignment:** ✓ Aligned (Umeyama)"
            else:
                self.gui_alignment_status.content = "**Alignment:** ✗ Not aligned"

        # Compute adaptive point size
        depth_list = loader.load_depth_list()
        if depth_list is not None:
            self.base_point_size = self._compute_base_point_size(depth_list, intrinsics_dict)
        else:
            self.base_point_size = DEFAULT_BASE_POINT_SIZE

        # Set frustum size to 50x base point size
        if self.gui_frustum_size is not None:
            self.gui_frustum_size.value = self.base_point_size * 50

        # ----------------------------------------------------------------
        # RGB thumbnails — load from cache or generate and cache
        # ----------------------------------------------------------------
        if cache.is_valid_rgb(N, image_width, image_height):
            logger.info("Loading RGB thumbnails from cache...")
            loaded = cache.load_rgb_thumbnails()
            self.rgb_thumbnails = loaded if loaded is not None else []
        else:
            # Load rgb_list if not already loaded (may have been loaded for sky seg above)
            if rgb_list is None:
                rgb_list = loader.load_rgb_list()
                if rgb_list and (image_width == 0 or image_height == 0):
                    image_height, image_width = rgb_list[0].shape[:2]
            logger.info("Generating RGB thumbnails and saving to cache...")
            self.rgb_thumbnails = []
            for rgb in rgb_list:
                img_pil = Image.fromarray(rgb)
                aspect_ratio = float(img_pil.width) / img_pil.height
                thumbnail_width = int(400 * aspect_ratio)
                img_pil = img_pil.resize((thumbnail_width, 400), Image.Resampling.LANCZOS)
                self.rgb_thumbnails.append(np.array(img_pil))
            cache.save_rgb_thumbnails(self.rgb_thumbnails, image_width, image_height)

        # Scene already cleared by caller (_reload_job); just reset bounds
        self.client.camera.fov = np.deg2rad(self.gui_fov.value)
        self.scene_min = np.array([np.inf, np.inf, np.inf])
        self.scene_max = np.array([-np.inf, -np.inf, -np.inf])

        first_frame_y: Optional[np.ndarray] = None
        frames_to_process = (N + temporal_subsample - 1) // temporal_subsample

        logger.info(f"Loading {frames_to_process} frames (subsample={temporal_subsample})")

        # Use atomic() here (not in the caller) so Loading... label can be sent first
        with self.client.atomic():
            # If using global point cloud, display it now (not per-frame)
            if self.use_global_pointcloud and len(self.cached_point_clouds) > 0:
                global_pcd = self.cached_point_clouds[0]
                if 'xyz' in global_pcd and len(global_pcd['xyz']) > 0:
                    actual_size = self.base_point_size * (10 ** self.gui_point_scale.value)
                    self.global_pointcloud_handle = self.client.scene.add_point_cloud(
                        name="/global_point_cloud",
                        points=global_pcd['xyz'],
                        colors=global_pcd.get('rgb'),
                        point_size=actual_size,
                        point_shape="rounded",
                    )
                    # Update scene bounds with global point cloud
                    self._update_scene_bounds(np.eye(4), global_pcd['xyz'])
                    logger.info(f"Displayed global point cloud with {len(global_pcd['xyz'])} points")

            # Need rgb_list for frame rendering — load if not already in memory
            if rgb_list is None:
                rgb_list = loader.load_rgb_list()
            # Process frames
            for frame_idx in range(N):
                if self._cancel_loading:
                    logger.info("Loading cancelled by user")
                    return

                if frame_idx % temporal_subsample != 0:
                    continue

                # Skip frames without valid trajectory
                if frame_idx not in traj_dict:
                    continue

                # Skip frames without valid intrinsics (use first valid as fallback)
                if frame_idx not in intrinsics_dict:
                    if not intrinsics_dict:
                        continue
                    fallback_idx = min(intrinsics_dict.keys())
                    intrinsics = intrinsics_dict[fallback_idx]
                else:
                    intrinsics = intrinsics_dict[frame_idx]

                # Print progress
                processed = frame_idx // temporal_subsample + 1
                progress = processed / frames_to_process
                bar_len = 40
                filled = int(bar_len * progress)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  Loading: [{bar}] {processed}/{frames_to_process} ({progress*100:.1f}%)", end="", flush=True)

                # Get frame data
                rgb = rgb_list[frame_idx]
                c2w = traj_dict[frame_idx]

                # Parse intrinsics
                if intrinsics.shape == (3, 3):
                    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                else:
                    fx, fy, cx, cy = intrinsics

                # Get current frame dimensions
                frame_height, frame_width = rgb.shape[0], rgb.shape[1]

                # Calculate vertical FOV
                fov = 2 * np.arctan2(float(frame_height / 2), fy)
                aspect_ratio = float(frame_width) / float(frame_height)

                # Subsample RGB
                sampled_rgb = rgb[::spatial_subsample, ::spatial_subsample]

                # Set up direction from first frame (only if not already saved)
                if first_frame_y is None:
                    first_frame_y = c2w[:3, 1]
                    self.client.scene.set_up_direction(-first_frame_y)
                    self.first_frame_pose = c2w.copy()

                # Get cached point cloud (only for per-frame clouds, not global)
                pcd_dict = {}
                if not self.use_global_pointcloud:
                    pcd_dict = self.cached_point_clouds[frame_idx] if frame_idx < len(self.cached_point_clouds) else {}

                frame_node = self._make_frame_nodes(
                    frame_idx,
                    c2w,
                    sampled_rgb,
                    fov,
                    pcd_dict,
                    aspect_ratio
                )
                self.scene_frame_handles.append(frame_node)

            print(f"\r  Loading: [{'█' * 40}] {frames_to_process}/{frames_to_process} (100.0%) ✓")
            logger.info(f"Loaded {len(self.scene_frame_handles)} frames")

            if len(self.scene_frame_handles) == 0:
                logger.warning("No valid frames to display")
                return

            # Apply gradient colors to frustums
            total_frames = len(self.scene_frame_handles)
            for idx, frame_node in enumerate(self.scene_frame_handles):
                progress = idx / max(1, total_frames - 1)
                color = self._get_gradient_color(progress)
                frame_node.frustum_handle.color = color

            # Create trajectory
            self._create_trajectory()

            # Set camera clipping planes
            self._update_camera_clipping_planes()

    def _make_frame_nodes(
        self,
        frame_idx: int,
        c2w: np.ndarray,
        rgb: np.ndarray,
        fov: float,
        pcd_dict: Dict[str, np.ndarray],
        aspect_ratio: float
    ) -> SceneFrameHandle:
        """Create viser handles for a single frame."""
        axes_length = self.gui_frustum_size.value / 3
        axes_radius = self.gui_frustum_size.value / 30

        handle = self.client.scene.add_frame(
            f"/frames/t{frame_idx}",
            axes_length=axes_length,
            axes_radius=axes_radius,
            wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
        )

        # Create thumbnail with aspect ratio EXACTLY matching the frustum
        # viser stretches the image to fill the frustum, so we must resize (not thumbnail)
        # to prevent distortion
        thumbnail_height = 200
        thumbnail_width = int(thumbnail_height * aspect_ratio)
        
        frame_thumbnail = Image.fromarray(rgb).resize(
            (thumbnail_width, thumbnail_height), 
            Image.Resampling.LANCZOS
        )

        frustum_handle = self.client.scene.add_camera_frustum(
            f"/frames/t{frame_idx}/frustum",
            fov=fov,
            aspect=aspect_ratio,
            scale=self.gui_frustum_size.value,
            image=np.array(frame_thumbnail),
        )

        # Create point cloud handle if data exists
        pcd_handle = None
        if 'xyz' in pcd_dict and len(pcd_dict['xyz']) > 0:
            # Apply runtime downsampling
            stride = self.gui_point_downsample.value
            pcd_points = pcd_dict['xyz'][::stride]
            pcd_colors = pcd_dict.get('rgb', None)
            if pcd_colors is not None:
                pcd_colors = pcd_colors[::stride]
                # Convert to uint8 if needed
                if pcd_colors.max() <= 1.0:
                    pcd_colors = (pcd_colors * 255).astype(np.uint8)
                else:
                    pcd_colors = pcd_colors.astype(np.uint8)

            actual_size = self.base_point_size * (10 ** self.gui_point_scale.value)

            # Point cloud is already in world coordinates (from load_point_cloud_grid),
            # so we add it to the scene root, not as a child of the frame
            pcd_handle = self.client.scene.add_point_cloud(
                name=f"/point_clouds/t{frame_idx}",
                points=pcd_points,
                colors=pcd_colors,
                point_size=actual_size,
                point_shape="rounded",
            )
            pcd_handle.visible = False

        return SceneFrameHandle(
            frame_handle=handle,
            frustum_handle=frustum_handle,
            pcd_handle=pcd_handle,
            frame_idx=frame_idx,
        )

    def _update_scene_bounds(self, c2w: np.ndarray, pcd: Optional[np.ndarray]):
        """Update scene bounding box."""
        cam_pos = c2w[:3, 3]
        self.scene_min = np.minimum(self.scene_min, cam_pos)
        self.scene_max = np.maximum(self.scene_max, cam_pos)

        if pcd is not None:
            valid_mask = np.all(np.isfinite(pcd), axis=1)
            if np.any(valid_mask):
                valid_pcd = pcd[valid_mask]
                pcd_min = np.min(valid_pcd, axis=0)
                pcd_max = np.max(valid_pcd, axis=0)
                self.scene_min = np.minimum(self.scene_min, pcd_min)
                self.scene_max = np.maximum(self.scene_max, pcd_max)

    def _update_camera_clipping_planes(self):
        """Update camera clipping planes based on scene bounds."""
        if np.any(np.isinf(self.scene_min)) or np.any(np.isinf(self.scene_max)):
            return

        scene_diagonal = np.linalg.norm(self.scene_max - self.scene_min)
        near_clip = min(0.0001, scene_diagonal * 0.0001)
        far_clip = scene_diagonal * 10.0

        try:
            self.client.camera.znear = near_clip
            self.client.camera.zfar = far_clip
            logger.info(f"Camera clipping: near={near_clip:.4f}, far={far_clip:.4f}")
        except AttributeError:
            logger.warning("Camera clipping not supported in this viser version")

    def _reset_camera_view(self):
        """Reset camera to view the first frame."""
        if len(self.scene_frame_handles) == 0:
            return

        first_frame = self.scene_frame_handles[0]
        c2w = np.eye(4)
        c2w[:3, :3] = tf.SO3(first_frame.frame_handle.wxyz).as_matrix()
        c2w[:3, 3] = first_frame.frame_handle.position

        look_dir = c2w[:3, 2]
        up_dir = -c2w[:3, 1]
        camera_pos = c2w[:3, 3] - look_dir * 2.0
        look_at = c2w[:3, 3]

        self.client.camera.position = camera_pos
        self.client.camera.look_at = look_at
        self.client.camera.up_direction = up_dir

    def _create_trajectory(self):
        """Create trajectory path with rainbow gradient colors (segmented lines)."""
        if len(self.scene_frame_handles) < 2:
            return

        # Clear old trajectory handles (scene.reset() already removed them from scene)
        self.trajectory_handles = []

        positions = []
        for frame_node in self.scene_frame_handles:
            pos = frame_node.frame_handle.position
            positions.append(pos)

        positions = np.array(positions)

        total_segments = len(positions) - 1
        for i in range(total_segments):
            progress = i / max(1, total_segments - 1)
            color = self._get_gradient_color(progress)
            color_normalized = tuple(c / 255.0 for c in color)

            segment_handle = self.client.scene.add_spline_catmull_rom(
                f"/trajectory/segment_{i}",
                positions[i:i+2].tolist(),
                color=color_normalized,
                line_width=3.0,
                segments=10,
            )
            self.trajectory_handles.append(segment_handle)

        self._update_trajectory_visibility()

    def _update_trajectory_visibility(self):
        """Update trajectory visibility based on checkbox."""
        if len(self.trajectory_handles) > 0 and hasattr(self, 'gui_show_trajectory'):
            visible = self.gui_show_trajectory.value

            if self.gui_timestep is not None:
                current_idx = self.gui_timestep.value
                for i, segment_handle in enumerate(self.trajectory_handles):
                    segment_handle.visible = visible and (i < current_idx)
            else:
                for segment_handle in self.trajectory_handles:
                    segment_handle.visible = visible

    def _incr_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % len(self.scene_frame_handles)

    def _decr_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % len(self.scene_frame_handles)

    def _first_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = 0

    def _last_timestep(self):
        if self.gui_timestep is not None and len(self.scene_frame_handles) > 0:
            self.gui_timestep.value = len(self.scene_frame_handles) - 1

    def _rebuild_playback_gui(self):
        """Rebuild playback controls."""
        current_timeline_value = self.gui_timestep.value if self.gui_timestep is not None else 0
        default_history = max(0, min(20, len(self.scene_frame_handles) - 1))
        current_camera_history = self.gui_history_cameras.value if self.gui_history_cameras is not None else default_history
        current_points_history = self.gui_history_points.value if self.gui_history_points is not None else default_history
        current_framerate_value = self.gui_framerate.value if self.gui_framerate is not None else 15

        if self.gui_playback_handle is not None:
            self.gui_playback_handle.remove()
        self.gui_playback_handle = self.client.gui.add_folder("Playback")

        with self.gui_playback_handle:
            self.gui_play_pause = self.client.gui.add_button(
                "Pause" if self.is_playing else "Play",
                hint="Toggle between play and pause"
            )

            gui_reset_view = self.client.gui.add_button(
                "Reset View",
                hint="Reset camera to view the first frame",
            )

            @gui_reset_view.on_click
            def _(_) -> None:
                self._reset_camera_view()

            max_frame_idx = max(0, len(self.scene_frame_handles) - 1)
            current_timeline_value = min(current_timeline_value, max_frame_idx)
            self.gui_timestep = self.client.gui.add_slider(
                "Timeline", min=0, max=max_frame_idx, step=1, initial_value=current_timeline_value
            )
            gui_frame_control = self.client.gui.add_button_group(
                "Control", options=["First", "Prev", "Next", "End"]
            )
            self.gui_framerate = self.client.gui.add_slider("FPS", min=0, max=30, step=1.0, initial_value=current_framerate_value)

            gui_loop = self.client.gui.add_checkbox(
                "Loop",
                initial_value=self.loop_playback,
                hint="Enable loop playback (restart from beginning when reaching end)"
            )

            @gui_loop.on_update
            async def _(_) -> None:
                self.loop_playback = gui_loop.value

            # History frames — split into Camera and Points
            max_history = max(0, len(self.scene_frame_handles) - 1)
            current_camera_history = max(0, min(current_camera_history, max_history))
            current_points_history = max(0, min(current_points_history, max_history))
            with self.client.gui.add_folder("History Frames", expand_by_default=True):
                self.gui_history_cameras = self.client.gui.add_slider(
                    "Camera", min=0, max=max_history, step=1, initial_value=current_camera_history,
                    hint="Number of historical camera frustums to display (0 = current frame only)"
                )
                self.gui_history_points = self.client.gui.add_slider(
                    "Points", min=0, max=max_history, step=1, initial_value=current_points_history,
                    hint="Number of historical point cloud frames to display (0 = current frame only)"
                )

            # Add RGB frame display
            if len(self.rgb_thumbnails) > 0:
                initial_idx = min(current_timeline_value, len(self.rgb_thumbnails) - 1)
                self.gui_frame_display = self.client.gui.add_image(
                    image=self.rgb_thumbnails[initial_idx],
                )

            @self.gui_play_pause.on_click
            async def _(_) -> None:
                self.is_playing = not self.is_playing
                self._rebuild_playback_gui()
                self._update_frame_visibility()

            @gui_frame_control.on_click
            async def _(_) -> None:
                if gui_frame_control.value == "First":
                    self._first_timestep()
                elif gui_frame_control.value == "Prev":
                    self._decr_timestep()
                elif gui_frame_control.value == "Next":
                    self._incr_timestep()
                else:  # "End"
                    self._last_timestep()

            @self.gui_timestep.on_update
            async def _(_) -> None:
                current_idx = self.gui_timestep.value

                # Update RGB thumbnail
                if self.gui_frame_display is not None and current_idx < len(self.rgb_thumbnails):
                    self.gui_frame_display.image = self.rgb_thumbnails[current_idx]

                # Update frame visibility
                self._update_frame_visibility()

                # Update trajectory visibility
                self._update_trajectory_visibility()

            @self.gui_history_cameras.on_update
            async def _(_) -> None:
                self._update_frame_visibility()

            @self.gui_history_points.on_update
            async def _(_) -> None:
                self._update_frame_visibility()

    def _update_frame_visibility(self):
        """Update frame visibility based on timeline."""
        if self.gui_timestep is None:
            return

        with self.client.atomic():
            current_idx = self.gui_timestep.value

            camera_history = self.gui_history_cameras.value if self.gui_history_cameras else 0
            points_history = self.gui_history_points.value if self.gui_history_points else 0
            show_cameras = self.gui_show_cameras.value if self.gui_show_cameras is not None else True

            # Show current frame + history frames (controlled by separate sliders)
            for i, frame in enumerate(self.scene_frame_handles):
                # Camera frustum/frame visibility
                if camera_history == 0:
                    camera_visible = (i == current_idx)
                else:
                    camera_visible = (current_idx - camera_history <= i <= current_idx)

                # Point cloud visibility
                if points_history == 0:
                    points_visible = (i == current_idx)
                else:
                    points_visible = (current_idx - points_history <= i <= current_idx)

                frame.frame_handle.visible = camera_visible and show_cameras
                frame.frustum_handle.visible = camera_visible and show_cameras
                if frame.pcd_handle is not None:
                    frame.pcd_handle.visible = points_visible

    def _update_cameras_visibility(self):
        """Update camera frustums visibility based on checkbox."""
        if self.gui_show_cameras is None:
            return

        show_cameras = self.gui_show_cameras.value

        with self.client.atomic():
            for frame in self.scene_frame_handles:
                if self.gui_timestep is not None:
                    current_idx = self.gui_timestep.value
                    camera_history = self.gui_history_cameras.value if self.gui_history_cameras else 0

                    if camera_history == 0:
                        timeline_visible = (frame.frame_idx == current_idx)
                    else:
                        timeline_visible = (current_idx - camera_history <= frame.frame_idx <= current_idx)

                    frame.frame_handle.visible = timeline_visible and show_cameras
                    frame.frustum_handle.visible = timeline_visible and show_cameras
                else:
                    frame.frame_handle.visible = show_cameras
                    frame.frustum_handle.visible = show_cameras

    def _copy_camera_view(self):
        """Copy current camera view parameters to global clipboard."""
        global _camera_clipboard
        
        try:
            _camera_clipboard = {
                'position': np.array(self.client.camera.position),
                'look_at': np.array(self.client.camera.look_at),
                'up_direction': np.array(self.client.camera.up_direction),
                'fov': self.client.camera.fov,
            }

            logger.info(f"Camera view copied: pos={_camera_clipboard['position']}, "
                       f"look_at={_camera_clipboard['look_at']}")
        except Exception as e:
            logger.error(f"Failed to copy camera view: {e}")

    def _paste_camera_view(self):
        """Paste camera view parameters from global clipboard."""
        global _camera_clipboard
        
        if _camera_clipboard is None:
            logger.warning("No camera data in clipboard")
            return

        try:
            self.client.camera.position = _camera_clipboard['position']
            self.client.camera.look_at = _camera_clipboard['look_at']
            self.client.camera.up_direction = _camera_clipboard['up_direction']
            self.client.camera.fov = _camera_clipboard['fov']

            if self.gui_fov is not None:
                self.gui_fov.value = float(np.rad2deg(_camera_clipboard['fov']))

            logger.info(f"Camera view pasted: pos={_camera_clipboard['position']}, "
                       f"look_at={_camera_clipboard['look_at']}")
        except Exception as e:
            logger.error(f"Failed to paste camera view: {e}")

    # ------------------------------------------------------------------
    # Sky segmentation helpers
    # ------------------------------------------------------------------

    def _run_sky_segmentation(self, rgb_list: List[np.ndarray]) -> List[np.ndarray]:
        """Run batched sky segmentation using ONNX-based SkySegmenter.

        Args:
            rgb_list: List of uint8 RGB images.

        Returns:
            List of boolean arrays (True = sky pixel to remove).
        """
        logger.info(f"Running sky segmentation on {len(rgb_list)} frames…")
        from vis.sky_seg import SkySegmenter
        segmenter = SkySegmenter(batch_size=16)
        masks = segmenter.segment_batch(list(rgb_list))
        logger.info("Sky segmentation complete (ONNX)")
        return masks

    def cleanup(self):
        """Clean up resources when client disconnects."""
        logger.info(f"Client {self.client.client_id} disconnected")

    @classmethod
    def global_context(cls) -> GlobalContext:
        global _global_context
        assert _global_context is not None, "Global context not initialized"
        return _global_context


# =============================================================================
# Main Server
# =============================================================================

def run_viser(
    workspace: Path,
    port: int = 20540,
    temporal_subsample: int = 1,
    spatial_subsample: int = 2,
    verbose: bool = False,
):
    """Run viser server for benchmark visualization.

    Args:
        workspace: Path to workspace directory.
        port: Port number for viser server.
        temporal_subsample: Initial temporal subsample value.
        spatial_subsample: Initial spatial subsample value.
        verbose: Enable verbose output.
    """
    logger.info(f"Scanning workspace: {workspace}")
    available_data, structure = scan_workspace(workspace)

    if len(available_data) == 0:
        logger.error("No valid data found in workspace (no .complete.json entries). Exiting.")
        return

    global _global_context, _global_context_verbose
    _global_context = GlobalContext(
        workspace=workspace,
        available_data=available_data,
        structure=structure,
        default_data_idx=0,
        temporal_subsample=temporal_subsample,
        spatial_subsample=spatial_subsample,
    )
    _global_context_verbose = verbose

    server = viser.ViserServer(host=get_host_ip(), port=port, verbose=False)
    server.gui.configure_theme(dark_mode=True, show_logo=False)
    client_closures: dict[int, ClientClosures] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle):
        client_closures[client.client_id] = ClientClosures(client)

    @server.on_client_disconnect
    async def _(client: viser.ClientHandle):
        await client_closures[client.client_id].stop()
        del client_closures[client.client_id]

    while True:
        try:
            time.sleep(10.0)
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Shutting down server...")
            break
    server.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D viewer for benchmark results",
        epilog="""
Examples:
  # View all data in workspace
  python viewer.py ./workspace

  # Custom port and subsampling
  python viewer.py ./workspace -p 8080 -t 5 -s 4
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workspace", type=Path, nargs="?", default=Path("."),
                        help="Path to workspace directory (default: current dir)")
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=20540,
        help="Port number for viser server (default: 20540)",
    )
    parser.add_argument(
        "-t", "--temporal-subsample",
        type=int,
        default=1,
        help="Initial temporal subsample value (default: 1)",
    )
    parser.add_argument(
        "-s", "--spatial-subsample",
        type=int,
        default=2,
        help="Initial spatial subsample value (default: 2)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    run_viser(
        args.workspace,
        args.port,
        args.temporal_subsample,
        args.spatial_subsample,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
