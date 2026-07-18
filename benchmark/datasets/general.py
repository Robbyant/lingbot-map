"""General dataset loader for arbitrary image folders or video files.

Usage:
  raw_data_root points directly to either:
    1. A directory of images (png/jpg/jpeg/bmp/tiff)
    2. A video file (.mp4/.avi/.mov/.mkv)

  The input type is auto-detected. No dataset/scene hierarchy is assumed;
  a single "scene" is created automatically from the input name.

  For video input, frames are extracted via ffmpeg into a sibling directory
  named {video_stem}_frames/.  Extracted frames are cached for reuse.

Optional COLMAP integration:
  When use_colmap=True, runs COLMAP (feature_extractor -> sequential_matcher
  -> mapper) to compute camera intrinsics and extrinsics.  Results are cached
  under {image_dir}/colmap_workspace/ and reused on subsequent runs.

Config example:
  datasets:
    my_images:
      dataset: general
      raw_data_root: /path/to/image_folder
      _use_colmap: true

    my_video:
      dataset: general
      raw_data_root: /path/to/video.mp4
      _video_fps: 2.0
      _use_colmap: true
"""

import logging
import subprocess
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional

from benchmark.dataset.base import BaseDataset


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


class GeneralDataset(BaseDataset):
    """General-purpose dataset loader for a single image folder or video file."""

    def __init__(
        self,
        raw_data_root: str,
        load_img_size: Optional[int] = None,
        video_fps: Optional[float] = None,
        use_colmap: bool = False,
        colmap_binary: str = 'colmap',
        colmap_camera_model: str = 'SIMPLE_PINHOLE',
        colmap_use_gpu: bool = True,
        colmap_max_num_features: int = 8192,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize general dataset loader.

        Args:
            raw_data_root: Path to an image directory OR a video file.
            load_img_size: Target image width for resizing at load time.
                           Height is computed to preserve aspect ratio and
                           rounded to a multiple of 14. None = no resizing.
            video_fps: Extract frames at this FPS from video.
                       None = extract all frames.
            use_colmap: Run COLMAP to compute intrinsics and poses.
            colmap_binary: Path to COLMAP binary.
            colmap_camera_model: COLMAP camera model
                (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV).
            colmap_use_gpu: Use GPU for COLMAP feature extraction / matching.
            colmap_max_num_features: Max SIFT features per image.
            logger: Optional logger instance.
        """
        super().__init__(raw_data_root, logger=logger)

        self._input_path = Path(raw_data_root)
        self._load_img_size = load_img_size
        self._video_fps = video_fps
        self._use_colmap = use_colmap
        self._colmap_binary = colmap_binary
        self._colmap_camera_model = colmap_camera_model
        self._colmap_use_gpu = colmap_use_gpu
        self._colmap_max_num_features = colmap_max_num_features

        # Detect input type
        self._is_video = (
            self._input_path.is_file()
            and self._input_path.suffix.lower() in VIDEO_EXTENSIONS
        )

        # Resolve the actual image directory
        if self._is_video:
            self._image_dir = self._input_path.parent / f"{self._input_path.stem}_frames"
            self._scene_name = self._input_path.stem
        else:
            if not self._input_path.is_dir():
                raise FileNotFoundError(
                    f"raw_data_root must be an image directory or video file, "
                    f"got: {self._input_path}"
                )
            self._image_dir = self._input_path
            self._scene_name = self._input_path.name

        # Lazy-loaded cache
        self._image_paths: Optional[List[Path]] = None
        self._colmap_data: Optional[Dict[str, Any]] = None
        self._colmap_loaded = False

    # ------------------------------------------------------------------ #
    #  BaseDataset interface                                              #
    # ------------------------------------------------------------------ #

    def get_scenes(self) -> List[str]:
        """Return a single scene derived from the input name."""
        return [self._scene_name]

    def get_frame_list(self, scene: str) -> List[int]:
        """Return sequential frame indices [0 .. N-1]."""
        paths = self._get_image_paths()
        return list(range(len(paths)))

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        """Load RGB and optional COLMAP intrinsics / pose for one frame.

        When load_img_size is set, the image is resized to that target width
        with aspect ratio preserved and height rounded to a multiple of 14.
        COLMAP intrinsics are scaled accordingly; if no COLMAP intrinsics
        exist, a default estimate (fx=fy=1.2*width, cx/cy at center) is used.

        Returns:
            Dict with 'rgb' (required), optionally 'intrinsics' and 'pose'.
        """
        paths = self._get_image_paths()
        img_path = paths[frame_id]

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # Resize if load_img_size is specified
        if self._load_img_size is not None:
            target_w = self._load_img_size
            aspect_ratio = float(orig_h) / orig_w
            target_h = int(target_w * aspect_ratio)
            # Round height to nearest multiple of 14 for patch-based models
            target_h = (target_h // 14) * 14
            img = img.resize((target_w, target_h), Image.LANCZOS)
        else:
            target_w, target_h = orig_w, orig_h

        rgb = np.array(img, dtype=np.uint8)
        result: Dict[str, Any] = {'rgb': rgb}

        colmap_data = self._get_colmap_data()
        if colmap_data is not None:
            img_name = img_path.name
            if img_name in colmap_data['poses']:
                result['pose'] = colmap_data['poses'][img_name]
            else:
                result['pose'] = None

            if img_name in colmap_data['intrinsics']:
                intr = colmap_data['intrinsics'][img_name].copy()
                # Scale intrinsics to match resized image
                if self._load_img_size is not None:
                    scale_x = target_w / orig_w
                    scale_y = target_h / orig_h
                    intr[0] *= scale_x  # fx
                    intr[1] *= scale_y  # fy
                    intr[2] *= scale_x  # cx
                    intr[3] *= scale_y  # cy
                result['intrinsics'] = intr
            elif self._load_img_size is not None:
                # No COLMAP intrinsics: use default estimate on resized size
                fx = fy = 1.2 * target_w
                cx, cy = target_w / 2.0, target_h / 2.0
                result['intrinsics'] = np.array(
                    [fx, fy, cx, cy], dtype=np.float32
                )

        return result

    # ------------------------------------------------------------------ #
    #  Image collection                                                   #
    # ------------------------------------------------------------------ #

    def _get_image_paths(self) -> List[Path]:
        """Return sorted image paths, extracting video frames if needed."""
        if self._image_paths is not None:
            return self._image_paths

        if self._is_video:
            self._image_paths = self._extract_video_frames()
        else:
            self._image_paths = self._collect_image_paths()

        if not self._image_paths:
            raise RuntimeError(f"No images found in {self._image_dir}")

        return self._image_paths

    def _collect_image_paths(self) -> List[Path]:
        """Collect sorted image paths from the image directory."""
        return sorted(
            p for p in self._image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _extract_video_frames(self) -> List[Path]:
        """Extract frames from video using ffmpeg, cached in a sibling dir."""
        self._image_dir.mkdir(parents=True, exist_ok=True)

        # Reuse previously extracted frames
        existing = self._collect_images_from(self._image_dir)
        if existing:
            if self.logger:
                self.logger.info(
                    f"Reusing {len(existing)} previously extracted frames "
                    f"from {self._image_dir}"
                )
            return existing

        if self.logger:
            self.logger.info(f"Extracting frames from {self._input_path}")

        cmd = ['ffmpeg', '-i', str(self._input_path)]
        if self._video_fps is not None:
            cmd.extend(['-vf', f'fps={self._video_fps}'])
        cmd.extend([
            '-start_number', '0',
            '-q:v', '1',
            str(self._image_dir / '%06d.jpg'),
        ])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg to extract video frames."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")

        extracted = self._collect_images_from(self._image_dir)
        if self.logger:
            self.logger.info(f"Extracted {len(extracted)} frames")
        return extracted

    @staticmethod
    def _collect_images_from(directory: Path) -> List[Path]:
        """Collect sorted image files from a directory."""
        return sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    # ------------------------------------------------------------------ #
    #  COLMAP integration                                                 #
    # ------------------------------------------------------------------ #

    def _get_colmap_data(self) -> Optional[Dict[str, Any]]:
        """Lazy-load COLMAP results (run pipeline if not cached)."""
        if self._colmap_loaded:
            return self._colmap_data
        self._colmap_loaded = True

        if not self._use_colmap:
            return None

        self._colmap_data = self._run_colmap()
        if self._colmap_data is None and self.logger:
            self.logger.warning(
                "COLMAP failed or unavailable, proceeding without poses"
            )
        return self._colmap_data

    def _run_colmap(self) -> Optional[Dict[str, Any]]:
        """Run COLMAP SfM pipeline and parse results.

        Returns:
            Dict with 'poses' and 'intrinsics' keyed by image filename,
            or None on failure.
        """
        if not self._check_colmap_available():
            if self.logger:
                self.logger.warning(
                    f"COLMAP binary '{self._colmap_binary}' not found"
                )
            return None

        colmap_ws = self._image_dir / 'colmap_workspace'
        sparse_dir = colmap_ws / 'sparse' / '0'

        # Reuse cached reconstruction
        if (sparse_dir / 'images.bin').exists() or \
           (sparse_dir / 'images.txt').exists():
            if self.logger:
                self.logger.info("Using cached COLMAP results")
            return self._parse_colmap_results(sparse_dir)

        colmap_ws.mkdir(parents=True, exist_ok=True)
        db_path = colmap_ws / 'database.db'
        use_gpu_str = '1' if self._colmap_use_gpu else '0'

        # 1) Feature extraction
        if self.logger:
            self.logger.info("Running COLMAP feature extraction...")
        if not self._run_colmap_cmd([
            self._colmap_binary, 'feature_extractor',
            '--database_path', str(db_path),
            '--image_path', str(self._image_dir),
            '--ImageReader.camera_model', self._colmap_camera_model,
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', use_gpu_str,
            '--SiftExtraction.max_num_features',
            str(self._colmap_max_num_features),
        ]):
            return None

        # 2) Sequential matching
        if self.logger:
            self.logger.info("Running COLMAP sequential matching...")
        if not self._run_colmap_cmd([
            self._colmap_binary, 'sequential_matcher',
            '--database_path', str(db_path),
            '--SiftMatching.use_gpu', use_gpu_str,
        ]):
            return None

        # 3) Mapper
        if self.logger:
            self.logger.info("Running COLMAP mapper...")
        (colmap_ws / 'sparse').mkdir(exist_ok=True)
        if not self._run_colmap_cmd([
            self._colmap_binary, 'mapper',
            '--database_path', str(db_path),
            '--image_path', str(self._image_dir),
            '--output_path', str(colmap_ws / 'sparse'),
        ]):
            return None

        if not sparse_dir.exists():
            if self.logger:
                self.logger.warning("COLMAP mapper produced no output")
            return None

        # 4) Convert binary to text for parsing
        if not (sparse_dir / 'images.txt').exists():
            self._run_colmap_cmd([
                self._colmap_binary, 'model_converter',
                '--input_path', str(sparse_dir),
                '--output_path', str(sparse_dir),
                '--output_type', 'TXT',
            ])

        if self.logger:
            self.logger.info("COLMAP reconstruction complete")
        return self._parse_colmap_results(sparse_dir)

    def _check_colmap_available(self) -> bool:
        """Check if COLMAP binary exists on PATH."""
        return shutil.which(self._colmap_binary) is not None

    def _run_colmap_cmd(self, cmd: List[str]) -> bool:
        """Execute a COLMAP sub-command; return True on success."""
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=3600)
            if result.returncode != 0:
                if self.logger:
                    stderr = result.stderr.decode(errors='replace')
                    self.logger.warning(
                        f"COLMAP failed: {' '.join(cmd[:2])}\n{stderr[:500]}"
                    )
                return False
            return True
        except subprocess.TimeoutExpired:
            if self.logger:
                self.logger.warning("COLMAP timed out (1 h limit)")
            return False
        except Exception as e:
            if self.logger:
                self.logger.warning(f"COLMAP error: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  COLMAP result parsing                                              #
    # ------------------------------------------------------------------ #

    def _parse_colmap_results(
        self, sparse_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Parse COLMAP sparse reconstruction into poses + intrinsics."""
        images_txt = sparse_dir / 'images.txt'
        cameras_txt = sparse_dir / 'cameras.txt'

        if images_txt.exists() and cameras_txt.exists():
            return self._parse_colmap_text(cameras_txt, images_txt)

        # Try converting binary -> text
        if (sparse_dir / 'images.bin').exists():
            self._run_colmap_cmd([
                self._colmap_binary, 'model_converter',
                '--input_path', str(sparse_dir),
                '--output_path', str(sparse_dir),
                '--output_type', 'TXT',
            ])
            if images_txt.exists() and cameras_txt.exists():
                return self._parse_colmap_text(cameras_txt, images_txt)

        if self.logger:
            self.logger.warning("No parseable COLMAP output found")
        return None

    def _parse_colmap_text(
        self,
        cameras_txt: Path,
        images_txt: Path,
    ) -> Dict[str, Any]:
        """Parse COLMAP text-format cameras.txt + images.txt.

        Returns dict with:
          'poses':      {image_name: 4x4 C2W float32}
          'intrinsics': {image_name: [fx, fy, cx, cy] float32}
        """
        from scipy.spatial.transform import Rotation

        # --- cameras.txt ---
        cameras: Dict[int, np.ndarray] = {}
        with open(cameras_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                model = parts[1]
                params = [float(x) for x in parts[4:]]
                cameras[cam_id] = _colmap_camera_to_intrinsics(model, params)

        # --- images.txt (pairs of lines: header + points2d) ---
        poses: Dict[str, np.ndarray] = {}
        intrinsics: Dict[str, np.ndarray] = {}

        with open(images_txt, 'r') as f:
            lines = [
                l.strip() for l in f
                if l.strip() and not l.strip().startswith('#')
            ]

        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            qw = float(parts[1])
            qx, qy, qz = float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            img_name = parts[9]

            # COLMAP stores W2C; convert to C2W
            rot_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t_w2c = np.array([tx, ty, tz], dtype=np.float64)
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = rot_w2c.T
            c2w[:3, 3] = -rot_w2c.T @ t_w2c

            poses[img_name] = c2w.astype(np.float32)
            if cam_id in cameras:
                intrinsics[img_name] = cameras[cam_id]

        if self.logger:
            self.logger.info(
                f"COLMAP: {len(poses)} images registered"
            )
        return {'poses': poses, 'intrinsics': intrinsics}


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _colmap_camera_to_intrinsics(
    model: str, params: List[float],
) -> np.ndarray:
    """Convert COLMAP camera parameters to [fx, fy, cx, cy].

    Supports: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV.
    """
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
        f, cx, cy = params[0], params[1], params[2]
        return np.array([f, f, cx, cy], dtype=np.float32)
    elif model in ('PINHOLE', 'OPENCV'):
        return np.array(params[:4], dtype=np.float32)
    else:
        if len(params) >= 4:
            return np.array(params[:4], dtype=np.float32)
        raise ValueError(f"Unsupported COLMAP camera model: {model}")
