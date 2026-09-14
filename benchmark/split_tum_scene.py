"""Split a long TUM RGB-D scene into shorter chunks, each a standalone scene.

The TUM dataset loader (benchmark/datasets/tum.py) globs `<scene>/rgb/*.png`
directly (ignores rgb.txt) and reads the whole `<scene>/groundtruth.txt`,
associating each RGB frame with its nearest-in-time GT pose independently.
So splitting is just: symlink a contiguous subset of the rgb/ images into a
new scene directory (named so it still matches `rgbd_dataset_freiburg*` and
keeps the same freiburgN camera id), and symlink the same groundtruth.txt
into it -- GT association still works correctly per chunk.

This exists because both streaming and windowed inference OOM past ~700-900
frames on a 22GB GPU for this checkpoint (memory scales with total sequence
length, not window size) -- splitting into independent chunks processes
every frame instead of subsampling.

Usage:
    python split_tum_scene.py rgbd_dataset_freiburg1_room --parts 2
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=str, help="Scene name under data/TUM-RGBD/")
    parser.add_argument("--parts", type=int, default=2, help="Number of equal chunks")
    parser.add_argument("--root", type=str, default="data/TUM-RGBD",
                         help="TUM-RGBD raw_data_root")
    args = parser.parse_args()

    root = Path(args.root)
    src = root / args.scene
    images = sorted((src / "rgb").glob("*.png"), key=lambda p: float(p.stem))
    n = len(images)
    print(f"{args.scene}: {n} frames -> {args.parts} parts")

    chunk = (n + args.parts - 1) // args.parts
    for i in range(args.parts):
        chunk_imgs = images[i * chunk: (i + 1) * chunk]
        if not chunk_imgs:
            continue
        dst = root / f"{args.scene}_part{i+1}"
        (dst / "rgb").mkdir(parents=True, exist_ok=True)
        for img in chunk_imgs:
            link = dst / "rgb" / img.name
            if not link.exists():
                link.symlink_to(img.resolve())
        gt_link = dst / "groundtruth.txt"
        if not gt_link.exists():
            gt_link.symlink_to((src / "groundtruth.txt").resolve())
        print(f"  {dst.name}: {len(chunk_imgs)} frames "
              f"({Path(chunk_imgs[0].name).stem} .. {Path(chunk_imgs[-1].name).stem})")


if __name__ == "__main__":
    main()
