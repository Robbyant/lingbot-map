"""Core evaluation orchestrator.

Orchestrates TrajectoryEvaluator, AUCEvaluator, DepthEvaluator, and PointCloudEvaluator
through a uniform evaluate(gt_loader, pred_loader, logger) interface.

Results are saved per evaluation type to eval/{type}.json within the method artifact.
"""

import logging
from typing import Any, Dict, Optional

from benchmark.core.loader import BSSLoader
from benchmark.evaluation.trajectory import TrajectoryEvaluator
from benchmark.evaluation.auc import AUCEvaluator
from benchmark.evaluation.depth import DepthEvaluator
from benchmark.evaluation.points import PointCloudEvaluator


class Evaluator:
    """Orchestrates all evaluators for a single (gt_artifact, pred_artifact) pair.

    All evaluators share the interface: evaluate(gt_loader, pred_loader, logger).
    Results are saved individually to eval/{type}.json files.
    """

    def __init__(
        self,
        gt_artifact,
        pred_artifact,
        eval_cfg: Dict[str, Any],
        logger: logging.Logger,
        dataset=None,
        force: bool = False,
    ):
        """Initialize evaluator.

        Args:
            gt_artifact:   BSSArtifact for ground truth
            pred_artifact: BSSArtifact for method output
            eval_cfg:      Merged evaluation config dict (after deep-merge with default)
            logger:        Logger instance
            dataset:       Optional dataset instance for point cloud evaluation
        """
        self._gt_artifact = gt_artifact
        self._pred_artifact = pred_artifact
        self.eval_cfg = eval_cfg
        self.logger = logger
        self._dataset = dataset
        self._force = force

        # Always instantiate trajectory and AUC evaluators
        self.trajectory_evaluator = TrajectoryEvaluator(align=True, correct_scale=True)
        self.auc_evaluator = AUCEvaluator()

        # Depth evaluator (instantiated from flattened depth config)
        depth_cfg = eval_cfg.get('depth', {})
        if depth_cfg.get('enable', False):
            gt_clip_cfg = depth_cfg.get('gt_clip', {})
            gt_clip = (
                gt_clip_cfg.get('min', 0.0),
                gt_clip_cfg.get('max', 80.0),
            ) if isinstance(gt_clip_cfg, dict) else (0.0, 80.0)

            pre_clip_cfg = depth_cfg.get('pre_clip', {})
            pre_clip = (
                (pre_clip_cfg.get('min'), pre_clip_cfg.get('max'))
                if isinstance(pre_clip_cfg, dict) and (
                    pre_clip_cfg.get('min') is not None or pre_clip_cfg.get('max') is not None
                ) else None
            )

            post_clip_cfg = depth_cfg.get('post_clip', {})
            post_clip = (
                (post_clip_cfg.get('min'), post_clip_cfg.get('max'))
                if isinstance(post_clip_cfg, dict) and (
                    post_clip_cfg.get('min') is not None or post_clip_cfg.get('max') is not None
                ) else None
            )

            self.depth_evaluator: Optional[DepthEvaluator] = DepthEvaluator(
                align=depth_cfg.get('align', 'scale_only'),
                gt_clip=gt_clip,
                pre_clip=pre_clip,
                post_clip=post_clip,
            )
        else:
            self.depth_evaluator = None

        # Point cloud evaluator
        points_cfg = eval_cfg.get('points', {})
        if points_cfg.get('enable', False) and dataset is not None:
            self.pointcloud_evaluator: Optional[PointCloudEvaluator] = PointCloudEvaluator(
                dataset, options={k: v for k, v in points_cfg.items() if k not in {'enable', 'vis'}}
            )
        else:
            self.pointcloud_evaluator = None

    def evaluate(self) -> None:
        """Run all enabled evaluators, save results to individual eval/*.json files."""
        gt_artifact   = self._gt_artifact
        pred_artifact = self._pred_artifact

        gt_loader   = BSSLoader(gt_artifact,   context=self._dataset)
        pred_loader = BSSLoader(pred_artifact, context=self._dataset)

        traj_cfg   = self.eval_cfg.get('traj',   {})
        auc_cfg    = self.eval_cfg.get('auc',    {})
        depth_cfg  = self.eval_cfg.get('depth',  {})
        points_cfg = self.eval_cfg.get('points', {})

        has_traj = (
            gt_artifact.traj_file.exists() and pred_artifact.traj_file.exists()
        )

        # ── Trajectory ───────────────────────────────────────────────────────
        if traj_cfg.get('enable', True) and has_traj:
            if not pred_artifact.has_eval('traj') or self._force:
                try:
                    traj_metrics = self.trajectory_evaluator.evaluate(
                        gt_loader, pred_loader, self.logger
                    )
                    T_align = traj_metrics.pop('traj_transform', None)
                    if T_align is not None:
                        pred_artifact.save_traj_transform(T_align)
                    pred_artifact.save_eval('traj', traj_metrics)

                    if traj_cfg.get('vis', False):
                        self.trajectory_evaluator.save_visualization(
                            gt_loader, pred_loader,
                            pred_artifact.vis_traj_dir,
                            logger=self.logger,
                        )
                except Exception as e:
                    self.logger.error(f"Trajectory evaluation failed: {e}", exc_info=True)

        # ── AUC ──────────────────────────────────────────────────────────────
        if auc_cfg.get('enable', True) and has_traj:
            if not pred_artifact.has_eval('auc') or self._force:
                try:
                    auc_metrics = self.auc_evaluator.evaluate(
                        gt_loader, pred_loader, self.logger
                    )
                    pred_artifact.save_eval('auc', auc_metrics)

                    if auc_cfg.get('vis', False):
                        frame_counts, auc_vals, racc_vals, tacc_vals = (
                            self.auc_evaluator.compute_auc_vs_frames(
                                gt_loader, pred_loader, step=10, threshold=30,
                                logger=self.logger,
                            )
                        )
                        self.auc_evaluator.save_visualization(
                            frame_counts, auc_vals, racc_vals, tacc_vals,
                            output_dir=pred_artifact.vis_auc_dir,
                            threshold=30,
                            logger=self.logger,
                        )
                except Exception as e:
                    self.logger.error(f"AUC evaluation failed: {e}", exc_info=True)

        # ── Depth ─────────────────────────────────────────────────────────────
        if self.depth_evaluator is not None:
            if not pred_artifact.has_eval('depth') or self._force:
                try:
                    depth_metrics = self.depth_evaluator.evaluate(
                        gt_loader, pred_loader, self.logger
                    )
                    if depth_metrics:
                        pred_artifact.save_eval('depth', depth_metrics)

                        if depth_cfg.get('vis', False):
                            self.depth_evaluator.save_visualization(
                                gt_loader, pred_loader,
                                pred_artifact.vis_depth_dir,
                                logger=self.logger,
                            )
                except Exception as e:
                    self.logger.error(f"Depth evaluation failed: {e}", exc_info=True)

        # ── Point cloud ────────────────────────────────────────────────────────
        if self.pointcloud_evaluator is not None:
            if not pred_artifact.has_eval('points') or self._force:
                try:
                    pc_results, gt_pts, pred_pts, thresholds = (
                        self.pointcloud_evaluator.evaluate(gt_loader, pred_loader, self.logger)
                    )
                    if pc_results:
                        pred_artifact.save_eval('points', pc_results)

                        if points_cfg.get('vis', False):
                            if gt_pts is not None and pred_pts is not None:
                                self.pointcloud_evaluator.save_pointcloud_visualization(
                                    gt_pts, pred_pts, thresholds,
                                    vis_dir=pred_artifact.vis_points_dir,
                                    logger=self.logger,
                                )
                except Exception as e:
                    self.logger.error(f"Point cloud evaluation failed: {e}", exc_info=True)
