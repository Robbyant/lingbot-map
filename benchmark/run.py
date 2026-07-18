"""Run phase: Execute methods on prepared datasets.

Dataset and method selection is controlled by the 'datasets' and 'methods' lists
in base.yaml.  Methods that declare an ``env`` field are dispatched to a subprocess
via ``conda run -n {env} --no-capture-output python run_worker.py``.

Usage:
    python run.py --config configs/base.yaml
    python run.py --config configs/base.yaml --debug
    python run.py --config configs/base.yaml --force
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

from benchmark.core.config import ConfigManager
from benchmark.core.registry import ClassLoader
from benchmark.core.storage import BSSManager, BSSArtifact
from benchmark.core.loader import BSSLoader
from benchmark.core.saver import BSSSaver
from benchmark.utils.logging import setup_logging


# ---------------------------------------------------------------------------
# In-process scene runner (used when method has no env)
# ---------------------------------------------------------------------------

def run_scene(method, gt_artifact: BSSArtifact, pred_artifact: BSSArtifact,
              logger: logging.Logger):
    """Run method on a single scene in the current process.

    Returns None on success, Exception on failure.
    """
    try:
        pred_artifact.clear_incomplete()
        pred_artifact.root.mkdir(parents=True, exist_ok=True)

        bss_loader = BSSLoader(gt_artifact, resize_context=method.resize_context)
        num_frames = bss_loader.get_num_frames()
        image_width, image_height = bss_loader.get_image_dimensions()

        logger.info(f"Found {num_frames} frames")
        logger.info("Running method...")
        output = method.process_scene(gt_artifact)

        method.validate_output(output, num_frames)

        logger.info("Saving results...")
        saver = BSSSaver(pred_artifact, context=method, logger=logger)

        frame_indices = output.get('frame_indices')
        frame_dict = output['frame']
        output_len = len(frame_dict.get('rgb', []))

        frame_data_list = []
        for i in range(output_len):
            frame_data = {}
            for key, value_list in frame_dict.items():
                if isinstance(value_list, list):
                    frame_data[key] = value_list[i]
            frame_data_list.append(frame_data)

        saver.save_frame_data(frame_data_list, image_width, image_height, frame_indices)

        if 'global' in output:
            saver.save_global_data(output['global'])

        metadata = {
            'num_frames': output_len,
            'image_width': image_width,
            'image_height': image_height,
        }
        metadata.update(saver.get_completion_metadata())
        pred_artifact.mark_complete(metadata=metadata)

        logger.info("Completed successfully")
        return None

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return e


# ---------------------------------------------------------------------------
# Subprocess dispatcher
# ---------------------------------------------------------------------------

def run_via_subprocess(env: str, config_path: str, method_name: str,
                       dataset_name: str, scene: str | None, force: bool,
                       logger: logging.Logger) -> int:
    """Dispatch a (method, dataset[, scene]) job to run_worker.py via conda run.

    Returns the subprocess exit code.
    """
    cmd = [
        'conda', 'run', '-n', env, '--no-capture-output',
        'python', 'run_worker.py',
        '--config', config_path,
        '--method', method_name,
        '--dataset', dataset_name,
    ]
    if scene:
        cmd += ['--scene', scene]
    if force:
        cmd += ['--force']

    logger.info(f"Dispatching to env '{env}': {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run methods on prepared BSS datasets"
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base configuration YAML file')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force re-run even if scene is already complete')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: process only the first scene of each dataset')

    args = parser.parse_args()

    try:
        config_manager = ConfigManager(Path(args.config))
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    workspace = Path(config_manager.get_workspace())
    bss_manager = BSSManager(workspace)

    log_dir = workspace / 'logs'
    logger = setup_logging(log_dir, name='run')
    logger.info("Starting run phase")
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Config: {args.config}")
    if args.force:
        logger.info("Force mode: will re-run completed scenes")
    if args.debug:
        logger.info("Debug mode: will process only the first scene of each dataset")

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
        f"Processing {len(dataset_names)} dataset(s) × {len(method_names)} method(s) "
        f"= {total_combinations} combination(s)"
    )

    total_success = 0
    total_failed  = 0
    all_failed_combinations = []

    combination_idx = 0
    for dataset_name in dataset_names:
        for method_name in method_names:
            combination_idx += 1
            logger.info("=" * 60)
            logger.info(
                f"Combination ({combination_idx}/{total_combinations}): "
                f"Running {method_name} on {dataset_name}"
            )
            logger.info("=" * 60)

            try:
                method_cfg = config_manager.get_method_config(method_name)
            except Exception as e:
                logger.error(f"Failed to load config for {method_name}: {e}")
                total_failed += 1
                all_failed_combinations.append(f"{method_name} on {dataset_name}: config error")
                continue

            env = method_cfg.get('env')

            # In debug mode: resolve the first scene for this dataset
            debug_scene = None
            if args.debug:
                try:
                    ds_cfg = config_manager.get_dataset_config(dataset_name)
                    dataset_class = ClassLoader.load_dataset(ds_cfg['dataset_class'])
                    dataset_inst  = dataset_class(logger=logger, **ds_cfg['params'])
                    scenes_all    = dataset_inst.get_scenes()
                    if not scenes_all:
                        logger.error(f"Dataset {dataset_name} returned no scenes")
                        continue
                    debug_scene = scenes_all[0]
                    logger.info(f"Debug mode: using first scene '{debug_scene}'")
                except Exception as e:
                    logger.error(f"Failed to get scenes for {dataset_name}: {e}")
                    continue

            # ------------------------------------------------------------------
            # Subprocess path: method has an env field
            # ------------------------------------------------------------------
            if env:
                rc = run_via_subprocess(
                    env=env,
                    config_path=args.config,
                    method_name=method_name,
                    dataset_name=dataset_name,
                    scene=debug_scene,
                    force=args.force,
                    logger=logger,
                )
                if rc == 0:
                    total_success += 1
                else:
                    total_failed += 1
                    all_failed_combinations.append(
                        f"{method_name} on {dataset_name}: worker exited with code {rc}"
                    )
                continue

            # ------------------------------------------------------------------
            # In-process path: no env field
            # ------------------------------------------------------------------
            if debug_scene:
                scenes = [debug_scene]
            else:
                scenes = bss_manager.list_scenes(dataset_name)
                if not scenes:
                    logger.error(f"No prepared scenes found for dataset: {dataset_name}")
                    continue

            scenes_to_run = []
            for scene in scenes:
                gt_artifact   = bss_manager.get_artifact(dataset_name, scene)
                pred_artifact = bss_manager.get_artifact(dataset_name, scene, method_name)

                if not gt_artifact.is_complete():
                    logger.warning(f"Scene {scene} not prepared, skipping")
                    continue

                pred_artifact.root.mkdir(parents=True, exist_ok=True)

                if args.force:
                    pred_artifact.clear_directory()
                    scenes_to_run.append((scene, gt_artifact, pred_artifact))
                elif not pred_artifact.is_complete():
                    scenes_to_run.append((scene, gt_artifact, pred_artifact))

            if not scenes_to_run:
                logger.info(
                    f"All scenes already complete for {method_name} on {dataset_name}, "
                    f"skipping model load"
                )
                continue

            logger.info(f"Scenes to process: {len(scenes_to_run)}/{len(scenes)}")

            try:
                method_class = ClassLoader.load_method(method_cfg['method_class'])
                method = method_class(logger=logger, **method_cfg['params'])
                logger.info(f"Loaded method: {method_class.__name__}")
            except Exception as e:
                logger.error(f"Failed to load method {method_name}: {e}")
                total_failed += len(scenes_to_run)
                all_failed_combinations.append(
                    f"{method_name} on {dataset_name}: model load error"
                )
                continue

            success_count = 0
            failed_scenes = []

            for idx, (scene, gt_artifact, pred_artifact) in enumerate(scenes_to_run, 1):
                logger.info(f"Scene ({idx}/{len(scenes_to_run)}): {scene}")
                error = run_scene(method, gt_artifact, pred_artifact, logger)
                if error is None:
                    success_count += 1
                else:
                    failed_scenes.append((scene, error))

            total_success += success_count
            total_failed  += len(failed_scenes)
            if failed_scenes:
                all_failed_combinations.append(
                    f"{method_name} on {dataset_name}: "
                    + ", ".join(f"{s} ({e})" for s, e in failed_scenes)
                )

            logger.info("-" * 60)
            logger.info(f"Combination '{method_name}' on '{dataset_name}' completed")
            logger.info(f"Successful: {success_count}/{len(scenes_to_run)}")
            if failed_scenes:
                for scene, error in failed_scenes:
                    logger.warning(f"  Failed: {scene} — {error}")

    logger.info("=" * 60)
    logger.info("Run phase completed")
    logger.info(f"Total successful: {total_success}")
    logger.info(f"Total failed: {total_failed}")
    if all_failed_combinations:
        logger.warning("Failed combinations:")
        for fail_msg in all_failed_combinations:
            logger.warning(f"  {fail_msg}")
    logger.info("=" * 60)

    if total_failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
