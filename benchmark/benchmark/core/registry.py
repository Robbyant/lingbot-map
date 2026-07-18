"""Dynamic class loading via reflection.

Provides registration-free loading of dataset and method classes.
"""

import importlib
import sys
from pathlib import Path
from typing import Type, TypeVar


T = TypeVar('T')


class ClassLoader:
    """Loads classes dynamically without requiring registration."""

    @staticmethod
    def load_dataset(name: str) -> Type:
        """Load a dataset class by name.

        Searches for the class in the datasets/ directory.

        Args:
            name: Dataset name (e.g., 'seven_scenes').

        Returns:
            Dataset class (subclass of BaseDataset).

        Raises:
            ImportError: If module or class cannot be loaded.
            AttributeError: If class doesn't exist in module.
        """
        return ClassLoader._load_class(
            module_path=f"datasets.{name}",
            class_name=ClassLoader.ensure_dataset_class_name(name),
            type_name="Dataset"
        )

    @staticmethod
    def load_method(name: str) -> Type:
        """Load a method class by name.

        Searches for the class in the methods/ directory.

        Args:
            name: Method name (e.g., 'passthrough').

        Returns:
            Method class (subclass of BaseMethod).

        Raises:
            ImportError: If module or class cannot be loaded.
            AttributeError: If class doesn't exist in module.
        """
        return ClassLoader._load_class(
            module_path=f"methods.{name}",
            class_name=ClassLoader.ensure_method_class_name(name),
            type_name="Method"
        )

    @staticmethod
    def _load_class(module_path: str, class_name: str, type_name: str) -> Type:
        """Load a class from a module path.

        Args:
            module_path: Python module path (e.g., 'datasets.seven_scenes').
            class_name: Class name to load (e.g., 'SevenScenesDataset').
            type_name: Type name for error messages (e.g., 'Dataset').

        Returns:
            Loaded class.

        Raises:
            ImportError: If module cannot be imported.
            AttributeError: If class doesn't exist in module.
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Failed to import {type_name} module '{module_path}': {e}\n"
                f"Make sure the file exists in the corresponding directory."
            ) from e

        try:
            cls = getattr(module, class_name)
        except AttributeError as e:
            raise AttributeError(
                f"Module '{module_path}' does not contain class '{class_name}'\n"
                f"Expected class name: {class_name}"
            ) from e

        return cls

    @staticmethod
    def _to_class_name(module_name: str) -> str:
        """Convert module name to expected class name.

        Converts snake_case to PascalCase and appends 'Dataset' or 'Method'.

        Examples:
            'seven_scenes' -> 'SevenScenesDataset'
            'passthrough' -> 'PassthroughMethod'

        Args:
            module_name: Module name in snake_case.

        Returns:
            Class name in PascalCase.
        """
        # Convert snake_case to PascalCase
        words = module_name.split('_')
        pascal_case = ''.join(word.capitalize() for word in words)
        return pascal_case

    @staticmethod
    def ensure_dataset_class_name(name: str) -> str:
        """Get expected dataset class name.

        Args:
            name: Dataset module name.

        Returns:
            Expected class name (PascalCase + 'Dataset').
        """
        return ClassLoader._to_class_name(name) + "Dataset"

    @staticmethod
    def ensure_method_class_name(name: str) -> str:
        """Get expected method class name.

        Args:
            name: Method module name.

        Returns:
            Expected class name (PascalCase + 'Method').
        """
        return ClassLoader._to_class_name(name) + "Method"
