"""RGB and mask image I/O utilities."""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union


def load_rgb(image_path: Path) -> np.ndarray:
    """Load RGB image.

    Args:
        image_path: Path to image file

    Returns:
        HxWx3 RGB image array (uint8, 0-255)
    """
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.uint8)


def save_rgb(image: np.ndarray, output_path: Path) -> None:
    """Save RGB image.

    Args:
        image: HxWx3 RGB image array (uint8, 0-255)
        output_path: Output file path
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    img = Image.fromarray(image, mode='RGB')
    img.save(output_path)


def load_mask(mask_path: Path) -> np.ndarray:
    """Load binary mask image.

    Args:
        mask_path: Path to mask file

    Returns:
        HxW boolean mask array (True for valid)
    """
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask, dtype=np.uint8)
    return mask_array > 0


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save binary mask image.

    Args:
        mask: HxW boolean mask array (True for valid)
        output_path: Output file path
    """
    # Convert boolean to uint8
    mask_uint8 = (mask.astype(np.uint8) * 255)

    img = Image.fromarray(mask_uint8, mode='L')
    img.save(output_path)


"""EXR I/O utilities."""

def save_exr(image: np.ndarray, output_file: Path) -> None:
    """Save image to EXR format with adaptive channel support.

    Args:
        image: Image array in one of the following formats:
            - HxW (single channel, grayscale)
            - HxWx3 (three channels, RGB)
            - HxWx4 (four channels, RGBA)
        output_file: Output EXR file path

    Raises:
        ImportError: If OpenEXR is not installed
        ValueError: If image array has unsupported shape or type
    """
    import OpenEXR
    import Imath

    # Check and convert to float32
    if image.dtype != np.float32:
        raise ValueError(f"Input image must be float32, got {image.dtype}")
    
    image = image.astype(np.float32)
    
    # Determine shape and channels
    if image.ndim == 2:
        # Single channel: HxW
        height, width = image.shape
        num_channels = 1
        channel_names = ['Y']
        channel_data = [image]
    elif image.ndim == 3:
        # Multi-channel: HxWxC
        height, width, num_channels = image.shape
        if num_channels == 3:
            channel_names = ['R', 'G', 'B']
            channel_data = [image[:, :, i] for i in range(3)]
        elif num_channels == 4:
            channel_names = ['R', 'G', 'B', 'A']
            channel_data = [image[:, :, i] for i in range(4)]
        elif num_channels == 1:
            channel_names = ['Y']
            channel_data = [image[:, :, 0]]
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}. Expected 1, 3, or 4.")
    else:
        raise ValueError(f"Unsupported image array dimension: {image.ndim}. Expected 2 or 3.")

    # Create EXR header with appropriate channels
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        for name in channel_names
    }

    # Write EXR file
    exr = OpenEXR.OutputFile(str(output_file), header)
    pixel_data = {name: data.tobytes() for name, data in zip(channel_names, channel_data)}
    exr.writePixels(pixel_data)
    exr.close()


def load_exr(input_file: Path, return_channels: Union[str, None] = None) -> np.ndarray:
    """Load image from EXR format with adaptive channel support.

    Args:
        input_file: Input EXR file path
        return_channels: How to return the data:
            - None (default): Auto-detect and return original format
            - 'single': Always return single channel (HxW)
            - 'rgb': Always return 3 channels (HxWx3)
            - 'rgba': Always return 4 channels (HxWx4)
            - 'all': Return all available channels

    Returns:
        Image array (float32) in format:
            - HxW for single channel
            - HxWx3 for RGB
            - HxWx4 for RGBA

    Raises:
        ImportError: If OpenEXR is not installed
        FileNotFoundError: If input file doesn't exist
        ValueError: If requested format cannot be satisfied
    """
    import OpenEXR
    import Imath

    exr_file = OpenEXR.InputFile(str(input_file))
    header = exr_file.header()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    available_channels = list(header['channels'].keys())
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Detect channel configuration
    has_rgba = all(ch in available_channels for ch in ['R', 'G', 'B', 'A'])
    has_rgb = all(ch in available_channels for ch in ['R', 'G', 'B'])
    has_y = 'Y' in available_channels
    
    # Determine which channels to read
    if return_channels == 'rgba' or (return_channels is None and has_rgba):
        # Read RGBA
        channels_to_read = ['R', 'G', 'B', 'A']
        if not has_rgba:
            raise ValueError(f"RGBA channels requested but not all available. Found: {available_channels}")
    elif return_channels == 'rgb' or (return_channels is None and has_rgb and not has_rgba):
        # Read RGB
        channels_to_read = ['R', 'G', 'B']
        if not has_rgb:
            raise ValueError(f"RGB channels requested but not all available. Found: {available_channels}")
    elif return_channels == 'single' or return_channels is None:
        # Read single channel - try Y first, then R, then first available
        if has_y:
            channels_to_read = ['Y']
        elif 'R' in available_channels:
            channels_to_read = ['R']
        elif 'Z' in available_channels:
            channels_to_read = ['Z']
        elif 'depth' in available_channels:
            channels_to_read = ['depth']
        else:
            # Use first available channel
            channels_to_read = [available_channels[0]]
    elif return_channels == 'all':
        # Read all available channels
        channels_to_read = available_channels
    else:
        raise ValueError(f"Invalid return_channels: {return_channels}")

    # Read channel data
    channel_arrays = []
    for ch in channels_to_read:
        if ch not in available_channels:
            raise ValueError(f"Channel '{ch}' not found in EXR. Available: {available_channels}")
        
        data_str = exr_file.channel(ch, pt)
        data = np.frombuffer(data_str, dtype=np.float32)
        data = data.reshape(height, width)
        channel_arrays.append(data)

    # Return appropriate format
    if len(channel_arrays) == 1:
        return channel_arrays[0]  # HxW
    else:
        return np.stack(channel_arrays, axis=-1)  # HxWxC
