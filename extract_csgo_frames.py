"""
Extract RGB frames from a CS:GO imitation-learning HDF5 file (frame_i_x datasets,
150x280x3 uint8) into a numbered PNG folder that lingbot-map's demo.py/batch_demo.py
expects for --image_folder.
"""

import argparse
from pathlib import Path

import h5py
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--hdf5_path", default=r"C:\csgo_data\hdf5_dm_july2021_1.hdf5")
parser.add_argument("--output_folder", default=r"C:\csgo_data\lingbot_map_example\courthouse_replacement")
parser.add_argument("--num_frames", type=int, default=500)
parser.add_argument("--stride", type=int, default=1)
args = parser.parse_args()

output_folder = Path(args.output_folder)
output_folder.mkdir(parents=True, exist_ok=True)

with h5py.File(args.hdf5_path, "r") as f:
    n_written = 0
    frame_idx = 0
    while n_written < args.num_frames:
        key = f"frame_{frame_idx}_x"
        if key not in f:
            break
        frame = f[key][()]
        Image.fromarray(frame).save(output_folder / f"{n_written:06d}.png")
        n_written += 1
        frame_idx += args.stride

print(f"Wrote {n_written} frames to {output_folder}")
