"""Evaluate phase: Compute metrics for method outputs.

Two-phase evaluation:
  Phase 1: Per-scene evaluation → writes Layer 1 (eval/*.json per method artifact)
  Phase 2: Aggregation → writes Layer 2 (scene-level) and Layer 3 (dataset-level)

Usage:
    python evaluate.py --config configs/base.yaml
    python evaluate.py --config configs/base.yaml --debug
    python evaluate.py --config configs/base.yaml --force
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Any

from benchmark.core.config import ConfigManager
from benchmark.core.registry import ClassLoader
from benchmark.core.storage import BSSManager
from benchmark.core.evaluator import Evaluator
from benchmark.utils.logging import setup_logging


def aggregate_eval(bss_manager, dataset_name, method_name, scenes, eval_cfg, logger):
    """Read Layer 1 json files from disk, generate Layer 2 and Layer 3.

    Layer 2: per-scene cross-method comparison (read-modify-write)
    Layer 3: dataset-level aggregation (read-modify-write)
    """
    from benchmark.evaluation.auc import AUCEvaluator

    all_auc   = []
    all_traj  = []
    all_depth = []
    all_points = []

    eval_types = ['traj', 'auc', 'depth', 'points']
    collectors = {
        'traj': all_traj,
        'auc': all_auc,
        'depth': all_depth,
        'points': all_points,
    }

    for scene in scenes:
        pred_artifact = bss_manager.get_artifact(dataset_name, scene, method_name)
        for eval_type in eval_types:
            data = pred_artifact.load_eval(eval_type)
            if data is None:
                continue

            # Layer 2: write scene-level cross-method comparison
            scene_data = data
            if eval_type == 'auc':
                scene_data = AUCEvaluator.strip_raw_errors(data)
            bss_manager.save_scene_eval(
                dataset_name, scene, method_name, eval_type, scene_data)

            # Collect for Layer 3
            collectors[eval_type].append(data)

    # Layer 3: dataset-level aggregation
    agg_mode = eval_cfg.get('auc', {}).get('aggregation', 'micro')

    if all_auc:
        if agg_mode in ('micro', 'both'):
            bss_manager.save_dataset_eval(
                dataset_name, method_name, 'auc_micro',
                AUCEvaluator.aggregate_micro(all_auc))
        if agg_mode in ('macro', 'both'):
            bss_manager.save_dataset_eval(
                dataset_name, method_name, 'auc_macro',
                AUCEvaluator.aggregate_macro(all_auc))

    if all_traj:
        bss_manager.save_dataset_eval(
            dataset_name, method_name, 'traj',
            _average_dicts(all_traj))

    if all_depth:
        bss_manager.save_dataset_eval(
            dataset_name, method_name, 'depth',
            _average_dicts(all_depth))

    if all_points:
        bss_manager.save_dataset_eval(
            dataset_name, method_name, 'points',
            _average_dicts(all_points))

    logger.info(f"Aggregation complete for {method_name} on {dataset_name}")


def _average_dicts(dicts: List[Dict]) -> Dict:
    """Average all numeric values across a list of dicts. Add num_scenes."""
    result = {'num_scenes': len(dicts)}
    keys = [k for k in dicts[0] if isinstance(dicts[0][k], (int, float))]
    for k in keys:
        values = [d[k] for d in dicts if k in d and isinstance(d[k], (int, float))]
        if values:
            result[k] = sum(values) / len(values)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate method outputs on BSS datasets"
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force re-evaluation by clearing eval/ directories')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: evaluate only the first scene of each dataset')

    args = parser.parse_args()

    try:
        config_manager = ConfigManager(Path(args.config))
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    workspace = Path(config_manager.get_workspace())
    bss_manager = BSSManager(workspace)

    log_dir = workspace / 'logs'
    logger = setup_logging(log_dir, name='evaluate')
    logger.info("Starting evaluation phase")
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Config: {args.config}")
    if args.debug:
        logger.info("Debug mode: will evaluate only the first scene of each dataset")

    try:
        dataset_names = config_manager.get_selected_dataset_names()
        method_names  = config_manager.get_selected_method_names()
    except KeyError as e:
        logger.error(str(e))
        sys.exit(1)

    if not dataset_names:
        logger.warning("No datasets selected. Add dataset names to the 'datasets' list in base.yaml.")
        sys.exit(0)
    if not method_names:
        logger.warning("No methods selected. Add method names to the 'methods' list in base.yaml.")
        sys.exit(0)

    total_combinations = len(dataset_names) * len(method_names)
    logger.info(
        f"Evaluating {len(dataset_names)} dataset(s) × {len(method_names)} method(s) "
        f"= {total_combinations} combination(s)"
    )

    total_success = 0
    total_failed  = 0
    all_failed_combinations = []

    combination_idx = 0
    for dataset_idx, dataset_name in enumerate(dataset_names, 1):
        for method_idx, method_name in enumerate(method_names, 1):
            combination_idx += 1
            logger.info("=" * 60)
            logger.info(
                f"Combination ({combination_idx}/{total_combinations}): "
                f"Evaluating {method_name} on {dataset_name}"
            )
            logger.info(
                f"  [Dataset {dataset_idx}/{len(dataset_names)}, "
                f"Method {method_idx}/{len(method_names)}]"
            )
            logger.info("=" * 60)

            # Load dataset instance (needed for point cloud evaluation and custom loaders)
            dataset = None
            try:
                ds_cfg = config_manager.get_dataset_config(dataset_name)
                dataset_params = dict(ds_cfg['params'])
                dataset_class  = ClassLoader.load_dataset(ds_cfg['dataset_class'])
                dataset = dataset_class(logger=logger, **dataset_params)
                logger.info(f"Dataset instance created: {dataset_class.__name__}")
            except Exception as e:
                logger.warning(f"Failed to load dataset instance: {e}")

            # Build scene list
            dataset_scenes = bss_manager.list_scenes(dataset_name)
            scenes = []
            for scene in dataset_scenes:
                methods = bss_manager.list_methods(dataset_name, scene)
                if method_name in methods:
                    scenes.append(scene)

            if not scenes:
                logger.error(f"No scenes found with outputs from method: {method_name}")
                continue

            if args.debug:
                scenes = scenes[:1]
                logger.info(f"Debug mode: evaluating first scene only: {scenes}")
            else:
                logger.info(f"Found {len(scenes)} scenes with method outputs")

            # Phase 1: per-scene evaluation (Layer 1)
            success_count = 0
            failed_scenes = []

            for scene_idx, scene in enumerate(scenes, 1):
                gt_artifact   = bss_manager.get_artifact(dataset_name, scene)
                pred_artifact = bss_manager.get_artifact(dataset_name, scene, method_name)

                if not gt_artifact.is_complete():
                    logger.warning(f"Scene {scene} GT not complete, skipping")
                    continue

                if not pred_artifact.is_complete():
                    logger.warning(f"Scene {scene} method output not complete, skipping")
                    continue

                if args.force:
                    pred_artifact.clear_eval()

                try:
                    logger.info(f"Evaluating scene ({scene_idx}/{len(scenes)}): {scene}")
                    evaluator = Evaluator(
                        gt_artifact, pred_artifact,
                        config_manager.get_merged_evaluation_config(dataset_name, method_name),
                        logger, dataset=dataset, force=args.force
                    )
                    evaluator.evaluate()
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to evaluate scene {scene}: {e}", exc_info=True)
                    failed_scenes.append(f"{dataset_name}/{scene}")

            # Phase 2: aggregation (Layer 2 + Layer 3)
            if not args.debug and success_count > 0:
                aggregate_eval(
                    bss_manager, dataset_name, method_name, scenes,
                    config_manager.get_merged_evaluation_config(dataset_name, method_name),
                    logger
                )

            total_success += success_count
            total_failed  += len(failed_scenes)
            if failed_scenes:
                all_failed_combinations.append(
                    f"{method_name} on {dataset_name}: {', '.join(failed_scenes)}"
                )

            logger.info("-" * 60)
            logger.info(f"Combination '{method_name}' on '{dataset_name}' completed")
            logger.info(f"Successful: {success_count}/{len(scenes)}")
            if failed_scenes:
                logger.warning(f"Failed scenes: {', '.join(failed_scenes)}")

    logger.info("=" * 60)
    logger.info("Evaluation phase completed")
    logger.info(f"Total successful: {total_success}")
    logger.info(f"Total failed: {total_failed}")
    if all_failed_combinations:
        logger.warning("Failed combinations:")
        for fail_msg in all_failed_combinations:
            logger.warning(f"  {fail_msg}")
    logger.info("=" * 60)

    if all_failed_combinations:
        sys.exit(1)


if __name__ == '__main__':
    main()
