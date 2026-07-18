"""Prepare phase: Convert raw datasets to standardized BSS format.

This script loads raw dataset using dataset loaders and converts them to
the standardized Benchmark Storage Structure (BSS) format.

Dataset selection is controlled by the 'datasets' list in base.yaml.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Dict, Any

from benchmark.core.config import ConfigManager
from benchmark.core.registry import ClassLoader
from benchmark.core.storage import BSSManager
from benchmark.core.saver import BSSSaver
from benchmark.utils.logging import setup_logging


def prepare_scene(gt_artifact, scene: str,
                  dataset, logger: logging.Logger,
                  sampling_config: Optional[Dict[str, Any]] = None) -> bool:
    """Prepare a single scene.

    Args:
        gt_artifact:     BSSArtifact for the GT directory
        scene:           Scene name (e.g. 'chess/seq-01')
        dataset:         Dataset instance
        logger:          Logger instance
        sampling_config: Optional sampling configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        # Clear incomplete data (no-op if directory doesn't exist or is already complete)
        gt_artifact.clear_incomplete()

        # Ensure GT directory exists
        gt_artifact.root.mkdir(parents=True, exist_ok=True)

        # Get frame list
        frame_list = dataset.get_frame_list(scene)
        original_count = len(frame_list)
        logger.info(f"Found {original_count} frames")

        if not frame_list:
            logger.warning(f"No frames found for scene {scene}")
            return False

        # Apply sampling if configured
        if sampling_config:
            frame_list = dataset.apply_sampling(frame_list, sampling_config)
            logger.info(f"Sampled {len(frame_list)}/{original_count} frames")

        # Load frame data with progress logging and parallel loading
        logger.info(f"Loading {len(frame_list)} frames...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing

        max_workers = min(multiprocessing.cpu_count(), 16)
        frame_data_list = [None] * len(frame_list)
        total_frames = len(frame_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(dataset.load_frame_data, scene, frame_id): (idx, frame_id)
                for idx, frame_id in enumerate(frame_list)
            }

            completed = 0
            for future in as_completed(future_to_idx):
                idx, frame_id = future_to_idx[future]
                try:
                    frame_data = future.result()
                    frame_data_list[idx] = frame_data
                    completed += 1

                    if completed % 500 == 0 or completed % max(1, total_frames // 10) == 0:
                        logger.info(
                            f"Loading progress: {completed}/{total_frames} frames "
                            f"({100*completed//total_frames}%)"
                        )
                except Exception as e:
                    logger.error(f"Failed to load frame {frame_id}: {e}")
                    raise

        logger.info(f"Completed loading all {total_frames} frames")

        # Get image dimensions from first frame
        first_rgb = frame_data_list[0]['rgb']
        image_height, image_width = first_rgb.shape[:2]

        # Initialize saver and save frame data
        saver = BSSSaver(gt_artifact, context=dataset, logger=logger)
        logger.info("Saving frame data...")
        saver.save_frame_data(frame_data_list, image_width, image_height)

        # Save sampling metadata if sampling was applied
        if sampling_config:
            saver.save_sampling_metadata(frame_list, sampling_config)

        # Load and save global data
        logger.info("Saving global data...")
        global_data = dataset.load_global_data(scene)
        saver.save_global_data(global_data)

        # Mark as complete (merge saver metadata for frame_keys/global_keys)
        metadata = {
            'num_frames': len(frame_list),
            'original_num_frames': original_count,
            'image_width': image_width,
            'image_height': image_height,
        }
        if sampling_config:
            metadata['sampling'] = sampling_config
        metadata.update(saver.get_completion_metadata())
        gt_artifact.mark_complete(metadata=metadata)

        logger.info(f"Scene {scene} completed successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to process scene {scene}: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets in standardized BSS format"
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force re-preparation even if scene is already complete')
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
    logger = setup_logging(log_dir, name='prepare')
    logger.info("Starting prepare phase")
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Config: {args.config}")
    if args.debug:
        logger.info("Debug mode: will process only the first scene of each dataset")

    try:
        dataset_names = config_manager.get_selected_dataset_names()
    except KeyError as e:
        logger.error(str(e))
        sys.exit(1)

    if not dataset_names:
        logger.warning("No datasets selected. Add dataset names to the 'datasets' list in base.yaml.")
        sys.exit(0)

    logger.info(f"Processing {len(dataset_names)} dataset configuration(s): {dataset_names}")

    total_success = 0
    total_failed = 0
    all_failed_scenes = []

    for dataset_idx, dataset_name in enumerate(dataset_names, 1):
        logger.info("=" * 60)
        logger.info(f"Dataset ({dataset_idx}/{len(dataset_names)}): Processing {dataset_name}")
        logger.info("=" * 60)

        try:
            ds_cfg = config_manager.get_dataset_config(dataset_name)
            dataset_class_name = ds_cfg['dataset_class']
            dataset_params = ds_cfg['params']
        except Exception as e:
            logger.error(f"Failed to load config for {dataset_name}: {e}")
            continue

        try:
            dataset_class = ClassLoader.load_dataset(dataset_class_name)
            logger.info(f"Loaded dataset class: {dataset_class.__name__}")
        except Exception as e:
            logger.error(f"Failed to load dataset class: {e}")
            continue

        sampling_config = ds_cfg.get('sampling')
        logger.info(f"Sampling config: {sampling_config}" if sampling_config else "No sampling applied")

        try:
            dataset = dataset_class(logger=logger, **dataset_params)
            logger.info("Dataset instance created successfully")
        except Exception as e:
            logger.error(f"Failed to instantiate dataset: {e}")
            continue

        scenes = dataset.get_scenes()
        if args.debug:
            scenes = scenes[:1]
            logger.info(f"Debug mode: using first scene only: {scenes}")

        success_count = 0
        failed_scenes = []

        for scene_idx, scene in enumerate(scenes, 1):
            gt_artifact = bss_manager.get_artifact(dataset_name, scene)

            if gt_artifact.is_complete() and not args.force:
                logger.info(f"Scene {scene} already complete, skipping")
                success_count += 1
                continue

            logger.info(f"Processing scene ({scene_idx}/{len(scenes)}): {scene}")
            if prepare_scene(gt_artifact, scene, dataset, logger, sampling_config):
                success_count += 1
            else:
                failed_scenes.append(f"{dataset_name}/{scene}")

        total_success += success_count
        total_failed += len(failed_scenes)
        all_failed_scenes.extend(failed_scenes)

        logger.info("-" * 60)
        logger.info(f"Dataset config '{dataset_name}' completed")
        logger.info(f"Successful: {success_count}/{len(scenes)}")
        if failed_scenes:
            logger.warning(f"Failed scenes: {', '.join(failed_scenes)}")

    logger.info("=" * 60)
    logger.info("Prepare phase completed")
    logger.info(f"Total successful: {total_success}")
    logger.info(f"Total failed: {total_failed}")
    if all_failed_scenes:
        logger.warning(f"Failed scenes: {', '.join(all_failed_scenes)}")
    logger.info("=" * 60)

    if all_failed_scenes:
        sys.exit(1)


if __name__ == '__main__':
    main()
