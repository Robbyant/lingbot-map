"""Worker script for running a single method on a single dataset.

Called by run.py via:
    conda run -n {env} --no-capture-output python run_worker.py \
        --config configs/base.yaml \
        --method cut3r \
        --dataset 7scenes_s10 \
        [--scene chess/seq-01]

Exit codes:
    0  - all scenes succeeded
    1  - one or more scenes failed
"""

import argparse
import logging
from pathlib import Path
import sys

from benchmark.core.config import ConfigManager
from benchmark.core.registry import ClassLoader
from benchmark.core.storage import BSSManager, BSSArtifact
from benchmark.core.loader import BSSLoader
from benchmark.core.saver import BSSSaver
from benchmark.utils.logging import setup_logging


def run_scene(method, gt_artifact: BSSArtifact, pred_artifact: BSSArtifact,
              logger: logging.Logger):
    """Run method on a single scene.

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


def main():
    parser = argparse.ArgumentParser(
        description="Worker: run one method on one dataset (called by run.py via conda run)"
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base configuration YAML file')
    parser.add_argument('--method', type=str, required=True,
                        help='Exact method config key (e.g., cut3r)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Exact dataset config key (e.g., 7scenes_s10)')
    parser.add_argument('--scene', type=str, default=None,
                        help='Process only this scene (default: all scenes)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force re-run even if scene is already complete')

    args = parser.parse_args()

    try:
        config_manager = ConfigManager(Path(args.config))
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    workspace = Path(config_manager.get_workspace())
    bss_manager = BSSManager(workspace)

    log_dir = workspace / 'logs'
    logger = setup_logging(log_dir, name=f'worker.{args.method}.{args.dataset}')
    logger.info(f"Worker started: method={args.method} dataset={args.dataset}")

    # Build scene list
    if args.scene:
        scenes = [args.scene]
        logger.info(f"Processing single scene: {args.scene}")
    else:
        scenes = bss_manager.list_scenes(args.dataset)
        if not scenes:
            logger.error(f"No prepared scenes found for dataset: {args.dataset}")
            sys.exit(1)

    # Load method config
    try:
        method_cfg = config_manager.get_method_config(args.method)
    except Exception as e:
        logger.error(f"Failed to load config for {args.method}: {e}")
        sys.exit(1)

    # Filter scenes that need processing
    scenes_to_run = []
    for scene in scenes:
        gt_artifact   = bss_manager.get_artifact(args.dataset, scene)
        pred_artifact = bss_manager.get_artifact(args.dataset, scene, args.method)

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
        logger.info("All scenes already complete, nothing to do")
        sys.exit(0)

    logger.info(f"Scenes to process: {len(scenes_to_run)}/{len(scenes)}")

    # Load method
    try:
        method_class = ClassLoader.load_method(method_cfg['method_class'])
        method = method_class(logger=logger, **method_cfg['params'])
        logger.info(f"Loaded method: {method_class.__name__}")
    except Exception as e:
        logger.error(f"Failed to load method {args.method}: {e}")
        sys.exit(1)

    # Process scenes
    success_count = 0
    failed_scenes = []

    for idx, (scene, gt_artifact, pred_artifact) in enumerate(scenes_to_run, 1):
        logger.info(f"Scene ({idx}/{len(scenes_to_run)}): {scene}")
        error = run_scene(method, gt_artifact, pred_artifact, logger)
        if error is None:
            success_count += 1
        else:
            failed_scenes.append((scene, error))

    logger.info("-" * 60)
    logger.info(f"Worker done: {success_count}/{len(scenes_to_run)} scenes succeeded")
    if failed_scenes:
        for scene, err in failed_scenes:
            logger.warning(f"  Failed: {scene} — {err}")
        sys.exit(1)


if __name__ == '__main__':
    main()
