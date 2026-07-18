"""Point cloud evaluation metrics and PointCloudEvaluator class."""

import logging
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree as KDTree
from typing import Dict, List, Optional, Tuple


def distance(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """Compute nearest neighbor distances from source points to target points.

    For each point in the source point cloud, finds the distance to its nearest
    neighbor in the target point cloud.

    Args:
        source_points: (N, D) array where D >= 3.
            First three dimensions/columns are XYZ coordinates.
        target_points: (M, D) array where D >= 3.
            First three dimensions/columns are XYZ coordinates.

    Returns:
        (N,) array of distances from each source point to its nearest target point.

    Note:
        Only XYZ coordinates are used for distance computation.
        Additional attributes (e.g., RGB, normals) are ignored.

    Example:
        >>> source = np.array([[0, 0, 0], [1, 0, 0]])
        >>> target = np.array([[0, 0, 0.1], [2, 0, 0]])
        >>> dists = distance(source, target)
        >>> dists.shape
        (2,)
    """
    # Extract XYZ coordinates
    source_xyz = source_points[:, :3]
    target_xyz = target_points[:, :3]

    # Build KD-tree for efficient nearest neighbor search
    tree = KDTree(target_xyz)

    # Query nearest neighbors
    # workers: Number of workers to use for parallel processing.
    # We use -1 to use all CPU threads for speedup.
    dist_s2t, _ = tree.query(source_xyz, workers=-1)

    return dist_s2t


def accuracy(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> float:
    """Compute accuracy: mean distance from source to target.

    Measures how close the source points are to the target point cloud.
    Lower values indicate better accuracy.

    Args:
        source_points: (N, D) array where D >= 3.
            First three dimensions/columns are XYZ coordinates.
        target_points: (M, D) array where D >= 3.
            First three dimensions/columns are XYZ coordinates.

    Returns:
        Mean distance from source to target (in same units as input coordinates).

    Note:
        This is directional: accuracy(A, B) != accuracy(B, A).
        For symmetric evaluation, use both accuracy and completeness.
    """
    dist_s2t = distance(source_points, target_points)
    return float(np.mean(dist_s2t))


def completeness(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> float:
    """Compute completeness: mean distance from target to source.

    Measures how well the source points cover the target point cloud.
    Lower values indicate better completeness.

    Args:
        source_points: (N, D) array where D >= 3.
            First three dimensions/columns are XYZ coordinates.
        target_points: (M, D) array where D >= 3.
            First three dimensions/columns are XYZ coordinates.

    Returns:
        Mean distance from target to source (in same units as input coordinates).

    Note:
        This is the reverse direction of accuracy: completeness(A, B) = accuracy(B, A).
    """
    dist_t2s = distance(target_points, source_points)
    return float(np.mean(dist_t2s))


def chamfer_distance(
    accuracy: float,
    completeness: float,
) -> float:
    """Compute Chamfer distance: average of accuracy and completeness.

    The Chamfer distance is a symmetric metric that combines both directions
    of nearest neighbor distances.

    Args:
        accuracy: Mean distance from source to target.
        completeness: Mean distance from target to source.

    Returns:
        Chamfer distance: Average of accuracy and completeness.

    Example:
        >>> source_points = np.random.rand(1000, 3)
        >>> target_points = np.random.rand(1000, 3)
        >>> acc = accuracy(source_points, target_points)
        >>> comp = completeness(source_points, target_points)
        >>> chamfer = chamfer_distance(acc, comp)
        >>> print(f"Chamfer: {chamfer:.4f}, Acc: {acc:.4f}, Comp: {comp:.4f}")
    """
    return (accuracy + completeness) / 2.0

def precision(
    source_points: np.ndarray,
    target_points: np.ndarray,
    thresholds: List[float],
) -> List[float]:
    """Compute precision at multiple distance thresholds.

    Precision measures the percentage of source points that are within
    a given distance threshold to the target point cloud.

    Args:
        source_points: (N, D) array where D >= 3.
            First three dimensions/columns are XYZ coordinates.
        target_points: (M, D) array where D >= 3.
            First three dimensions/columns are XYZ coordinates.
        thresholds: List of distance thresholds to evaluate.

    Returns:
        List of precision values (as percentages 0-100) for each threshold.

    Note:
        Higher precision means more source points are close to the target.
        Precision alone does not measure if the target is fully covered.

    Example:
        >>> prec = precision(pred_points, gt_points, [0.01, 0.05, 0.10])
        >>> print(f"Precision_5cm: {prec[1]:.2f}%")
    """
    dist_s2t = distance(source_points, target_points)

    precisions = []
    for threshold in thresholds:
        ratio = np.mean(dist_s2t < threshold)
        precisions.append(float(ratio * 100.0))

    return precisions


def recall(
    source_points: np.ndarray,
    target_points: np.ndarray,
    thresholds: List[float],
) -> List[float]:
    """Compute recall at multiple distance thresholds.

    Recall measures the percentage of target points that are within
    a given distance threshold to the source point cloud.

    Args:
        source_points: (N, D) array where D >= 3.
            First three dimensions/columns are XYZ coordinates.
        target_points: (M, D) array where D >= 3.
            First three dimensions/columns are XYZ coordinates.
        thresholds: List of distance thresholds to evaluate.

    Returns:
        List of recall values (as percentages 0-100) for each threshold.

    Note:
        Higher recall means more target points are covered by the source.
        Recall alone does not penalize outliers in the source.

    Example:
        >>> rec = recall(pred_points, gt_points, [0.01, 0.05, 0.10])
        >>> print(f"Recall_5cm: {rec[1]:.2f}%")
    """
    dist_t2s = distance(target_points, source_points)

    recalls = []
    for threshold in thresholds:
        ratio = np.mean(dist_t2s < threshold)
        recalls.append(float(ratio * 100.0))

    return recalls


def f1_score(
    precision: List[float],
    recall: List[float],
) -> List[float]:
    """Compute F1-score at multiple distance thresholds.

    F1-score is the harmonic mean of precision and recall, providing
    a balanced measure of reconstruction quality.

    Args:
        source_points: (N, D) array of source point cloud where D >= 3.
            First three dimensions are XYZ coordinates.
        target_points: (M, D) array of target point cloud where D >= 3.
            First three dimensions are XYZ coordinates.
        thresholds: List of distance thresholds to evaluate.

    Returns:
        List of F1-score values (as percentages 0-100) for each threshold.

    Note:
        F1 = 2 * (precision * recall) / (precision + recall)
        If precision + recall = 0, F1 is set to 0.

    Example:
        >>> thresholds = [0.01, 0.05, 0.10]
        >>> precision = precision(pred_points, gt_points, thresholds)
        >>> recall = recall(pred_points, gt_points, thresholds)
        >>> f1 = f1_score(precision, recall)
        >>> print(f"F1_5cm: {f1[1]:.2f}%")
    """

    f1_scores = []
    for prec, rec in zip(precision, recall):
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        f1_scores.append(float(f1))

    return f1_scores


def evaluate_pointcloud(
    source_points: np.ndarray,
    target_points: np.ndarray,
    thresholds: List[float] = None,
) -> dict:
    """Compute comprehensive point cloud evaluation metrics.

    Convenience function that computes all common metrics in one call.

    Args:
        source_points: Array with shape [..., 3] or [..., 6].
        target_points: Array with shape [..., 3] or [..., 6].
        thresholds: List of distance thresholds for precision/recall/F1.
            Default: [0.01, 0.02, 0.05, 0.10] (in same units as coordinates).

    Returns:
        Dictionary containing:
        - chamfer: Chamfer distance
        - accuracy: Mean source-to-target distance
        - completeness: Mean target-to-source distance
        - pred_points: Source points with distance appended as last column
        - gt_points: Target points with distance appended as last column
        - precision_{T}: Precision at threshold T (for each T in thresholds)
        - recall_{T}: Recall at threshold T (for each T in thresholds)
        - f1_{T}: F1-score at threshold T (for each T in thresholds)

    Example:
        >>> pred = np.random.rand(1000, 3)
        >>> gt = np.random.rand(1000, 3)
        >>> metrics = evaluate_pointcloud(pred, gt)
        >>> print(f"Chamfer: {metrics['chamfer']:.4f}")
        >>> print(f"Precision_5cm: {metrics['precision_0.05']:.2f}%")
    """
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.10]
    
    source_points = source_points.reshape(-1, source_points.shape[-1])
    target_points = target_points.reshape(-1, target_points.shape[-1])

    # Compute distances once to avoid redundant KD-tree construction
    dist_s2t = distance(source_points, target_points)  # source → target
    dist_t2s = distance(target_points, source_points)  # target → source

    # Reshape to 2D arrays
    source_with_dist = np.hstack([source_points, dist_s2t.reshape(-1, 1)])
    target_with_dist = np.hstack([target_points, dist_t2s.reshape(-1, 1)])

    # Distance-based metrics (using pre-computed distances)
    acc = float(np.mean(dist_s2t))
    comp = float(np.mean(dist_t2s))
    chamfer = chamfer_distance(acc, comp)

    # Threshold-based metrics (using pre-computed distances)
    prec_list = []
    for threshold in thresholds:
        ratio = np.mean(dist_s2t < threshold)
        prec_list.append(float(ratio * 100.0))

    rec_list = []
    for threshold in thresholds:
        ratio = np.mean(dist_t2s < threshold)
        rec_list.append(float(ratio * 100.0))

    f1_list = f1_score(precision=prec_list, recall=rec_list)

    # Build results dictionary
    # pred_points, gt_points and thresholds are included 
    # for potential visualization and analysis purposes.
    results = {
        'chamfer': chamfer,
        'accuracy': acc,
        'completeness': comp,
        'pred_points': source_with_dist,
        'gt_points': target_with_dist,
        'thresholds': thresholds,
    }

    # Add threshold-specific metrics
    if len(thresholds) == 1:
        # If only one threshold, use generic keys without threshold suffix
        results['precision'] = prec_list[0]
        results['recall'] = rec_list[0]
        results['f1'] = f1_list[0]
    else:
        for i, threshold in enumerate(thresholds):
            threshold_str = f"{threshold:.4f}".rstrip('0').rstrip('.')
            results[f'precision_{threshold_str}'] = prec_list[i]
            results[f'recall_{threshold_str}'] = rec_list[i]
            results[f'f1_{threshold_str}'] = f1_list[i]

    return results


def colorize_by_distance(
    points_with_dist: np.ndarray,
    tau: float,
    colormap: str = 'hot_r',
) -> np.ndarray:
    """Colorize points by distance error using a colormap.

    Maps nearest-neighbor distances to colors, capped at 3*tau.
    Ported from the official TAT toolbox write_color_distances().

    Args:
        points_with_dist: Nx4 (xyz+dist) or Nx7 (xyz+rgb+dist) array.
            Distance must be in the last column.
        tau: Distance threshold used for evaluation. Colors are scaled
            relative to this value; distances > 3*tau are clamped.
        colormap: Matplotlib colormap name. Default 'hot_r'.

    Returns:
        Nx7 array: xyz + colormap_rgb (float32, [0,1]) + dist.
    """
    import matplotlib
    cmap = matplotlib.colormaps[colormap]
    max_dist = 3.0 * tau

    dist = points_with_dist[:, -1]
    xyz = points_with_dist[:, :3]

    colors = cmap(np.minimum(dist, max_dist) / max_dist)[:, :3].astype(np.float32)

    return np.hstack([xyz, colors, dist.reshape(-1, 1)])


def plot_pr_curve(
    dist_pred2gt: np.ndarray,
    dist_gt2pred: np.ndarray,
    tau: float,
    output_path: Path,
) -> None:
    """Plot a precision-recall CDF curve and save to file.

    Ported from the official TAT toolbox plot_graph().
    Red curve = precision (pred→GT distances), Blue curve = recall (GT→pred).

    Args:
        dist_pred2gt: 1-D array of pred-to-GT nearest-neighbour distances.
        dist_gt2pred: 1-D array of GT-to-pred nearest-neighbour distances.
        tau: Distance threshold. Dashed vertical line drawn at this value.
            X-axis spans [0, 10*tau].
        output_path: File path to save the PNG (parent dirs created if needed).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_stretch = 10.0
    bins = np.arange(0, tau * plot_stretch, tau / 100.0)

    hist_s, edges_s = np.histogram(dist_pred2gt, bins)
    cum_s = np.cumsum(hist_s).astype(float) / max(len(dist_pred2gt), 1)

    hist_t, edges_t = np.histogram(dist_gt2pred, bins)
    cum_t = np.cumsum(hist_t).astype(float) / max(len(dist_gt2pred), 1)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(edges_s[1:], cum_s * 100, c='red',  label='precision', linewidth=2.0)
    ax.plot(edges_t[1:], cum_t * 100, c='blue', label='recall',    linewidth=2.0)
    ax.axvline(x=tau, c='black', ls='dashed', linewidth=2.0)
    ax.set_xlim(0, tau * plot_stretch)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Meters', fontsize=15)
    ax.set_ylabel('# of points (%)', fontsize=15)
    title = f"Precision and Recall"
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc='lower right', fontsize='medium', shadow=True, fancybox=True)

    fig.savefig(str(output_path), format='png', bbox_inches='tight')
    plt.close(fig)


class PointCloudEvaluator:
    """Wraps dataset.evaluate_pointcloud() as a uniform evaluator."""

    def __init__(self, dataset, options: dict = None):
        """Initialize point cloud evaluator.

        Args:
            dataset: Dataset instance with optional evaluate_pointcloud() method
            options: Dataset-specific evaluation options from config
        """
        self.dataset = dataset
        self.options = options or {}

    def evaluate(
        self,
        gt_loader,
        pred_loader,
        logger=None,
    ) -> Tuple[Optional[dict], Optional[np.ndarray], Optional[np.ndarray], list]:
        """Run point cloud evaluation via dataset.evaluate_pointcloud().

        Args:
            gt_loader:   BSSLoader for ground truth directory
            pred_loader: BSSLoader for method output directory
            logger:      Optional logger

        Returns:
            (metrics_dict, gt_points, pred_points, thresholds)
            gt_points/pred_points: Nx(3+1) with nearest-neighbor distance appended.
            Returns (None, None, None, []) if not available.
        """
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            logger = logging.getLogger(__name__)

        if self.dataset is None or not hasattr(self.dataset, 'evaluate_pointcloud'):
            return None, None, None, []

        result = self.dataset.evaluate_pointcloud(
            gt_loader, pred_loader, logger, options=self.options
        )
        if result is None:
            return None, None, None, []

        gt_points   = result.pop('gt_points', None)
        pred_points = result.pop('pred_points', None)
        thresholds  = result.pop('thresholds', [])
        return result, gt_points, pred_points, thresholds

    def save_pointcloud_visualization(
        self,
        gt_points: np.ndarray,
        pred_points: np.ndarray,
        thresholds: list,
        vis_dir: Path,
        logger=None,
    ) -> None:
        """Save threshold-colored PLY files and PR curves to vis_dir.

        Files are named with tau in the filename:
            vis_dir / f'gt_points_{tau}.ply'
            vis_dir / f'pred_points_{tau}.ply'
            vis_dir / f'pr_curve_{tau}.png'

        Visualization files (colored PLY, PR curves) are saved to vis_dir
        (typically eval/points/ within the method artifact).

        Args:
            gt_points:   Nx(3+1) GT point cloud with distance in last column
            pred_points: Mx(3+1) predicted point cloud with distance in last column
            thresholds:  List of distance thresholds used during evaluation
            vis_dir:     Directory to write visualization files
            logger:      Optional logger
        """
        if logger is None:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
            )
            logger = logging.getLogger(__name__)

        from benchmark.io.pointcloud import save_point_cloud_ply

        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

        for tau in thresholds:
            tau_str = f"{tau:.4f}".rstrip('0').rstrip('.')

            if gt_points is not None and pred_points is not None:
                try:
                    dist_pred2gt = pred_points[:, -1]
                    dist_gt2pred = gt_points[:, -1]
                    plot_pr_curve(
                        dist_pred2gt, dist_gt2pred, tau,
                        vis_dir / f'pr_curve_{tau_str}.png',
                    )
                except Exception as e:
                    logger.debug(f"PR curve failed (tau={tau}): {e}")

                try:
                    gt_colored = colorize_by_distance(gt_points, tau)
                    save_point_cloud_ply(gt_colored, vis_dir / f'gt_points_{tau_str}.ply')
                except Exception as e:
                    logger.debug(f"GT ply save failed (tau={tau}): {e}")

                try:
                    pred_colored = colorize_by_distance(pred_points, tau)
                    save_point_cloud_ply(pred_colored, vis_dir / f'pred_points_{tau_str}.ply')
                except Exception as e:
                    logger.debug(f"Pred ply save failed (tau={tau}): {e}")
