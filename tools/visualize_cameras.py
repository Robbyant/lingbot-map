#!/usr/bin/env python3
"""Visualize camera trajectory from cameras.json produced by demo.py.

Usage:
    python tools/visualize_cameras.py output/cameras.json
    python tools/visualize_cameras.py output/cameras.json --save trajectory.png
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def draw_camera(ax, c2w, scale=0.05, color="steelblue"):
    """Draw a small camera frustum: 3 axes + pyramid outline."""
    origin = c2w[:3, 3]
    # Unit axes in camera space (right=x, up=-y, forward=z)
    axes = c2w[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float).T
    colors = ["red", "green", color]
    labels = ["X", "Y", "Z"]
    for i, (col, lbl) in enumerate(zip(colors, labels)):
        tip = origin + axes[:, i] * scale
        ax.plot(*zip(origin, tip), color=col, linewidth=1)

    # Frustum corners (simplified: just 4 corner rays)
    corners_cam = np.array([[1, 1, 2], [-1, 1, 2], [-1, -1, 2], [1, -1, 2]], dtype=float) * scale * 0.5
    corners_world = (c2w[:3, :3] @ corners_cam.T).T + origin
    for corner in corners_world:
        ax.plot(*zip(origin, corner), color=color, linewidth=0.5, alpha=0.5)
    # Close the frustum rectangle
    rect = np.vstack([corners_world, corners_world[0]])
    ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], color=color, linewidth=0.5, alpha=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cameras_json", help="Path to cameras.json")
    parser.add_argument("--save", default=None, help="Save figure to file instead of showing")
    parser.add_argument("--skip", type=int, default=1, help="Draw every N-th camera (default 1 = all)")
    parser.add_argument("--frustum_scale", type=float, default=None, help="Frustum size (auto if unset)")
    args = parser.parse_args()

    with open(args.cameras_json) as f:
        cameras = json.load(f)

    positions = np.array([c["c2w"][i][3] for c in cameras for i in range(3)]).reshape(-1, 3)
    positions = np.array([[c["c2w"][0][3], c["c2w"][1][3], c["c2w"][2][3]] for c in cameras])

    span = np.max(positions, axis=0) - np.min(positions, axis=0)
    scale = float(np.max(span)) * 0.04 if args.frustum_scale is None else args.frustum_scale

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            color="gray", linewidth=1, alpha=0.6, label="trajectory")

    # Start / end markers
    ax.scatter(*positions[0], color="lime", s=80, zorder=5, label="start")
    ax.scatter(*positions[-1], color="red", s=80, zorder=5, label="end")

    # Camera frustums
    for i, cam in enumerate(cameras):
        if i % args.skip != 0:
            continue
        c2w = np.array(cam["c2w"])            # (3, 4)
        c2w_4x4 = np.eye(4)
        c2w_4x4[:3, :] = c2w
        t = i / max(len(cameras) - 1, 1)
        color = plt.cm.cool(t)
        draw_camera(ax, c2w_4x4, scale=scale, color=color)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Camera trajectory  ({len(cameras)} frames)")
    ax.legend()

    # Equal aspect ratio
    center = positions.mean(axis=0)
    half = float(np.max(span)) * 0.55
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"Saved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
