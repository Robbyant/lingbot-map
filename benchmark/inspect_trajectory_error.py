"""Inspect per-frame trajectory error between a prediction and its GT.

Prints the same aggregate ATE/RPE numbers evaluate.py computes, but also
breaks the ATE down *per frame* (not just the RMSE) so you can see exactly
where/how much the predicted trajectory diverges from ground truth, and
saves a plot of that error curve alongside the existing overlay plot.

Run from benchmark/, inside the `lingbot-map` (or `bench`) conda env:

    python inspect_trajectory_error.py \
        --scene_dir data/workspace/tum/tum/rgbd_dataset_freiburg1_desk

--scene_dir is the folder that contains `gt/` and `<method>/` (e.g. `lingbot_map/`).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from evo.core.metrics import PoseRelation
import evo.main_ape as main_ape

from benchmark.core.storage import BSSArtifact
from benchmark.core.loader import BSSLoader
from benchmark.evaluation.trajectory import (
    TrajectoryEvaluator,
    _array_to_evo_trajectory,
    _filter_valid_pose_pairs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", type=str, required=True,
                         help="Scene directory containing gt/ and <method>/ subfolders")
    parser.add_argument("--method", type=str, default="lingbot_map",
                         help="Method subfolder name to compare against gt/ (default: lingbot_map)")
    parser.add_argument("--out_dir", type=str, default=None,
                         help="Where to save the error plot (default: <scene_dir>/<method>/eval/inspect)")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    out_dir = Path(args.out_dir) if args.out_dir else scene_dir / args.method / "eval" / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_loader = BSSLoader(BSSArtifact(scene_dir / "gt"))
    pred_loader = BSSLoader(BSSArtifact(scene_dir / args.method))

    # 1. Aggregate metrics -- same numbers evaluate.py writes to eval/traj.json
    evaluator = TrajectoryEvaluator(align=True, correct_scale=True)
    agg = evaluator.evaluate(gt_loader, pred_loader)
    print("=== Aggregate (matches eval/traj.json) ===")
    print(f"  ATE (RMSE)        : {agg['ate']*100:.2f} cm")
    print(f"  RPE translation   : {agg['rpe_trans']*100:.2f} cm")
    print(f"  RPE rotation      : {agg['rpe_rot']:.3f} deg")

    # 2. Per-frame ATE -- how far off each individual frame's position is,
    #    after the same rigid alignment used for the aggregate ATE above.
    gt_traj = gt_loader.load_trajectory()
    pred_traj = pred_loader.load_trajectory()
    pred_frame_indices = pred_loader.get_frame_indices()
    gt_poses = gt_traj[pred_frame_indices]
    pred_poses = pred_traj
    timestamps = np.array(pred_frame_indices, dtype=float)
    gt_poses, pred_poses, timestamps = _filter_valid_pose_pairs(gt_poses, pred_poses, timestamps)

    traj_ref = _array_to_evo_trajectory(gt_poses, timestamps)
    traj_est = _array_to_evo_trajectory(pred_poses, timestamps)

    ape_result = main_ape.ape(
        traj_ref, traj_est, est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True, correct_scale=True,
    )
    per_frame_error = ape_result.np_arrays["error_array"]  # meters, one value per frame

    print("\n=== Per-frame ATE (meters) ===")
    print(f"  min / mean / max : {per_frame_error.min():.4f} / {per_frame_error.mean():.4f} / {per_frame_error.max():.4f}")
    worst = np.argsort(per_frame_error)[::-1][:5]
    print("  5 worst frames (frame_idx: error_m):")
    for i in worst:
        print(f"    {int(timestamps[i])}: {per_frame_error[i]:.4f}")

    # 3. Plot: error over the sequence.
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, per_frame_error)
    ax.set_xlabel("frame index")
    ax.set_ylabel("ATE (m)")
    ax.set_title(f"Per-frame trajectory error: {args.method} vs gt ({scene_dir.name})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    error_plot_path = out_dir / "per_frame_ate.png"
    fig.savefig(error_plot_path, dpi=150)
    print(f"\nSaved per-frame error plot to: {error_plot_path}")

    # 4. Also refresh the overlay (predicted vs GT trajectory) plot.
    evaluator.save_visualization(gt_loader, pred_loader, out_dir)
    print(f"Saved trajectory overlay plot to: {out_dir / 'trajectory_visualization.png'}")


if __name__ == "__main__":
    main()
