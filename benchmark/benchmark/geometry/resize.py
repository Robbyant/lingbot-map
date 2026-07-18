"""Image resizing utilities for the benchmark framework.

Supports two resize modes:
  - 'none':        No resizing; images are used at native resolution.
  - 'area_budget': Uniform downscale so total pixel count <= area_budget;
                   output dimensions are aligned to a given multiple.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2


@dataclass
class ResizeTransform:
    """Encapsulates a single resize/crop transformation.

    Attributes:
        original_size:     (W, H) of the original image
        intermediate_size: (W, H) after pure scaling (before crop)
        final_size:        (W, H) of the output image seen by the method
        scale_x:           horizontal scale factor (intermediate / original)
        scale_y:           vertical scale factor (intermediate / original)
        content_x0:        left edge of the crop region in the intermediate image
        content_y0:        top edge of the crop region in the intermediate image
        content_w:         width of the crop region
        content_h:         height of the crop region
        dest_x0:           x offset where content is placed in the final canvas
        dest_y0:           y offset where content is placed in the final canvas
    """
    original_size: Tuple[int, int]       # (W, H)
    intermediate_size: Tuple[int, int]   # (W, H) after scaling
    final_size: Tuple[int, int]          # (W, H) what the method receives
    scale_x: float
    scale_y: float
    content_x0: int   # crop region in intermediate image
    content_y0: int
    content_w: int
    content_h: int
    dest_x0: int      # where content is placed in final canvas
    dest_y0: int


def _compute_resize_transform(
    orig_w: int,
    orig_h: int,
    mode: str,
    align: int = 1,
    area_budget: Optional[int] = None,
) -> ResizeTransform:
    """Compute a ResizeTransform for the given mode.

    Args:
        orig_w:      Original image width
        orig_h:      Original image height
        mode:        One of 'none', 'area_budget'
        align:       Alignment divisor (output dimensions must be multiples of this)
        area_budget: Maximum pixel count for 'area_budget' mode

    Returns:
        ResizeTransform describing the full transformation pipeline
    """
    if mode == 'none':
        return ResizeTransform(
            original_size=(orig_w, orig_h),
            intermediate_size=(orig_w, orig_h),
            final_size=(orig_w, orig_h),
            scale_x=1.0,
            scale_y=1.0,
            content_x0=0,
            content_y0=0,
            content_w=orig_w,
            content_h=orig_h,
            dest_x0=0,
            dest_y0=0,
        )

    elif mode == 'area_budget':
        # Scale down so total pixel count <= area_budget; align dimensions
        assert area_budget is not None, "area_budget mode requires area_budget"
        scale = min(np.sqrt(area_budget / (orig_w * orig_h)), 1.0)
        new_w = (int(orig_w * scale) // align) * align
        new_h = (int(orig_h * scale) // align) * align
        # Ensure at least align×align
        new_w = max(new_w, align)
        new_h = max(new_h, align)
        return ResizeTransform(
            original_size=(orig_w, orig_h),
            intermediate_size=(new_w, new_h),
            final_size=(new_w, new_h),
            scale_x=new_w / orig_w,
            scale_y=new_h / orig_h,
            content_x0=0,
            content_y0=0,
            content_w=new_w,
            content_h=new_h,
            dest_x0=0,
            dest_y0=0,
        )

    else:
        raise ValueError(f"Unknown resize mode: '{mode}'. "
                         f"Supported: none, area_budget")


def _apply_resize_transform(
    image: np.ndarray,
    transform: ResizeTransform,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Apply a ResizeTransform to an image.

    Pipeline:
      1. Resize to intermediate_size
      2. Crop [content_y0:content_y0+content_h, content_x0:content_x0+content_w]
      3. Place into a zero canvas of final_size at (dest_x0, dest_y0)

    Args:
        image:         Input image (HxW or HxWxC)
        transform:     Transformation parameters
        interpolation: OpenCV interpolation method

    Returns:
        Transformed image at final_size (H_final x W_final [x C])
    """
    inter_w, inter_h = transform.intermediate_size
    final_w, final_h = transform.final_size

    # Step 1: Resize to intermediate size
    resized = cv2.resize(image, (inter_w, inter_h), interpolation=interpolation)

    # Step 2: Crop content region
    x0, y0 = transform.content_x0, transform.content_y0
    cw, ch = transform.content_w, transform.content_h
    if resized.ndim == 2:
        content = resized[y0:y0 + ch, x0:x0 + cw]
    else:
        content = resized[y0:y0 + ch, x0:x0 + cw, :]

    # Step 3: Place into final canvas
    if (transform.dest_x0 == 0 and transform.dest_y0 == 0
            and cw == final_w and ch == final_h):
        # Fast path: no padding needed
        return content

    if resized.ndim == 2:
        canvas = np.zeros((final_h, final_w), dtype=image.dtype)
    else:
        canvas = np.zeros((final_h, final_w, image.shape[2]), dtype=image.dtype)

    dx, dy = transform.dest_x0, transform.dest_y0
    canvas[dy:dy + ch, dx:dx + cw] = content
    return canvas


def adjust_intrinsics(
    intrinsics: np.ndarray,
    transform: ResizeTransform,
) -> np.ndarray:
    """Adjust camera intrinsics for a ResizeTransform.

    Applies:
      1. Scale:  fx *= scale_x,  fy *= scale_y,  cx *= scale_x,  cy *= scale_y
      2. Crop:   cx -= content_x0,  cy -= content_y0
      3. Place:  cx += dest_x0,    cy += dest_y0

    Args:
        intrinsics: [fx, fy, cx, cy] array or 3x3 camera matrix
        transform:  Transformation parameters

    Returns:
        Adjusted intrinsics in the same format as input
    """
    is_matrix = (intrinsics.ndim == 2 and intrinsics.shape == (3, 3))

    if is_matrix:
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    else:
        fx, fy, cx, cy = float(intrinsics[0]), float(intrinsics[1]), \
                          float(intrinsics[2]), float(intrinsics[3])

    fx_new = fx * transform.scale_x
    fy_new = fy * transform.scale_y
    cx_new = cx * transform.scale_x - transform.content_x0 + transform.dest_x0
    cy_new = cy * transform.scale_y - transform.content_y0 + transform.dest_y0

    if is_matrix:
        result = intrinsics.copy().astype(float)
        result[0, 0] = fx_new
        result[1, 1] = fy_new
        result[0, 2] = cx_new
        result[1, 2] = cy_new
        return result
    else:
        return np.array([fx_new, fy_new, cx_new, cy_new], dtype=intrinsics.dtype)


class ResizeContext:
    """Manages resize operations for a scene.

    Lazily computes the ResizeTransform on first use and caches it.
    All resize operations are owned by this class.
    """

    def __init__(
        self,
        mode: str = 'none',
        align: int = 1,
        area_budget: Optional[int] = None,
    ):
        """Initialize resize context.

        Args:
            mode:        Resize strategy. One of: 'none', 'area_budget'
            align:       Alignment requirement (output dims must be multiples of this)
            area_budget: Maximum pixel area for 'area_budget' mode
        """
        self.mode = mode
        self.align = align
        self.area_budget = area_budget
        self._transform: Optional[ResizeTransform] = None

    @property
    def enabled(self) -> bool:
        """Whether resizing is active (mode != 'none')."""
        return self.mode != 'none'

    @property
    def transform(self) -> Optional[ResizeTransform]:
        """The cached ResizeTransform, or None if not yet computed."""
        return self._transform

    @property
    def final_size(self) -> Optional[Tuple[int, int]]:
        """Output (W, H) after transform, or None if not yet computed."""
        if self._transform is not None:
            return self._transform.final_size
        return None

    def get_transform(self, orig_w: int, orig_h: int) -> ResizeTransform:
        """Compute (or return cached) transform for the given input dimensions.

        Args:
            orig_w: Original image width
            orig_h: Original image height

        Returns:
            ResizeTransform for these dimensions
        """
        if (self._transform is not None
                and self._transform.original_size == (orig_w, orig_h)):
            return self._transform
        self._transform = _compute_resize_transform(
            orig_w, orig_h,
            mode=self.mode,
            align=self.align,
            area_budget=self.area_budget,
        )
        return self._transform

    def apply(self, image: np.ndarray,
              interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """Apply resize transform to an image.

        Determines image dimensions automatically and computes the transform
        if not yet cached.

        Args:
            image:         Input image (HxW or HxWxC)
            interpolation: OpenCV interpolation flag

        Returns:
            Transformed image at final_size
        """
        if not self.enabled:
            return image
        h, w = image.shape[:2]
        transform = self.get_transform(w, h)
        return _apply_resize_transform(image, transform, interpolation)

    def apply_nearest(self, image: np.ndarray) -> np.ndarray:
        """Apply resize transform with nearest-neighbor interpolation.

        Suitable for depth maps, masks, and integer-valued arrays.

        Args:
            image: Input image (HxW or HxWxC)

        Returns:
            Transformed image at final_size
        """
        return self.apply(image, cv2.INTER_NEAREST)

    def adjust_intrinsics(self, intrinsics: np.ndarray) -> np.ndarray:
        """Adjust camera intrinsics for this resize transform.

        Note: get_transform() must have been called first to initialize the
        cached transform (e.g., after apply() is called on at least one image).
        If the transform is not yet cached, returns the intrinsics unchanged.

        Args:
            intrinsics: [fx, fy, cx, cy] array or 3x3 camera matrix

        Returns:
            Adjusted intrinsics in the same format as input
        """
        if not self.enabled or self._transform is None:
            return intrinsics
        return adjust_intrinsics(intrinsics, self._transform)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        d = {
            'mode': self.mode,
            'enabled': self.enabled,
            'align': self.align,
            'area_budget': self.area_budget,
        }
        if self._transform is not None:
            t = self._transform
            d['transform'] = {
                'original_size': list(t.original_size),
                'intermediate_size': list(t.intermediate_size),
                'final_size': list(t.final_size),
                'scale_x': t.scale_x,
                'scale_y': t.scale_y,
                'content_x0': t.content_x0,
                'content_y0': t.content_y0,
                'content_w': t.content_w,
                'content_h': t.content_h,
                'dest_x0': t.dest_x0,
                'dest_y0': t.dest_y0,
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ResizeContext':
        """Deserialize from a dict (e.g., loaded from JSON).

        Args:
            d: Dictionary produced by to_dict()

        Returns:
            ResizeContext instance with cached transform if present
        """
        ctx = cls(
            mode=d.get('mode', 'none'),
            align=d.get('align', 1),
            area_budget=d.get('area_budget'),
        )
        if 'transform' in d:
            td = d['transform']
            ctx._transform = ResizeTransform(
                original_size=tuple(td['original_size']),
                intermediate_size=tuple(td['intermediate_size']),
                final_size=tuple(td['final_size']),
                scale_x=td['scale_x'],
                scale_y=td['scale_y'],
                content_x0=td['content_x0'],
                content_y0=td['content_y0'],
                content_w=td['content_w'],
                content_h=td['content_h'],
                dest_x0=td['dest_x0'],
                dest_y0=td['dest_y0'],
            )
        return ctx

    @classmethod
    def none(cls) -> 'ResizeContext':
        """Create a no-op resize context (mode='none')."""
        return cls(mode='none')

    def __repr__(self) -> str:
        if self.enabled and self._transform is not None:
            return (f"ResizeContext(mode={self.mode!r}, "
                    f"{self._transform.original_size} -> {self._transform.final_size})")
        return f"ResizeContext(mode={self.mode!r})"
