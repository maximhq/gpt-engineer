"""
Disk Memory Module
==================

This module provides a simple file-based key-value database system, where keys are
represented as filenames and values are the contents of these files. The `DiskMemory` class
is responsible for the CRUD operations on the database.

Attributes
----------
None

Functions
---------
None

Classes
-------
DiskMemory
    A file-based key-value store where keys correspond to filenames and values to file contents.
"""

import base64
import json
import shutil

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

from gpt_engineer.core.base_memory import BaseMemory
from gpt_engineer.tools.supported_languages import SUPPORTED_LANGUAGES

# Import observability
try:
    from gpt_engineer.core.maxim_observability import get_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


# This class represents a simple database that stores its tools as files in a directory.
class DiskMemory(BaseMemory):
    """
    A file-based key-value store where keys correspond to filenames and values to file contents.

    This class provides an interface to a file-based database, leveraging file operations to
    facilitate CRUD-like interactions. It allows for quick checks on the existence of keys,
    retrieval of values based on keys, and setting new key-value pairs.

    Attributes
    ----------
    path : Path
        The directory path where the database files are stored.
    """

    def __init__(self, path: Union[str, Path], suppress_observability: bool = False):
        """
        Initialize the DiskMemory class with a specified path.

        Parameters
        ----------
        path : str or Path
            The path to the directory where the database files will be stored.
        suppress_observability : bool
            If True, disables observability logging for this instance.
        """
        self.path: Path = Path(path).absolute()
        self.suppress_observability = suppress_observability
        self.path.mkdir(parents=True, exist_ok=True)

    def __contains__(self, key: str) -> bool:
        """
        Determine whether the database contains a file with the specified key.

        Parameters
        ----------
        key : str
            The key (filename) to check for existence in the database.

        Returns
        -------
        bool
            Returns True if the file exists, False otherwise.

        """
        return (self.path / key).is_file()

    def __getitem__(self, key: Union[str, Path]) -> str:
        """
        Retrieve the content of a file in the database corresponding to the given key.
        If the file is an image with a .png or .jpeg extension, it returns the content
        in Base64-encoded string format.

        Parameters
        ----------
        key : str or Path
            The key (filename) whose content is to be retrieved.

        Returns
        -------
        str
            The content of the file associated with the key, or Base64-encoded string if it's a .png or .jpeg file.

        Raises
        ------
        KeyError
            If the file corresponding to the key does not exist in the database.
        """
        key = Path(key)

        # Determine operation type
        if key.suffix in {".py", ".js", ".ts", ".html", ".css"}:
            pass
        elif key.suffix in {".txt", ".md", ".rst"}:
            pass
        elif key.suffix in {".json", ".yaml", ".yml", ".toml"}:
            pass

        try:
            # Handle images as binary and base64 encode
            if key.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
                with open(self.path / key, "rb") as f:
                    val = base64.b64encode(f.read()).decode("utf-8")
            else:
                with open(self.path / key, "r", encoding="utf-8") as f:
                    val = f.read()

            # Log file read operation with file content attachment if observability is available
            if (
                not getattr(self, "suppress_observability", False)
                and OBSERVABILITY_AVAILABLE
            ):
                try:
                    observability = get_observability()
                    if observability.is_enabled():
                        # Removed log_retrieval call
                        pass
                except Exception:
                    pass  # Don't fail file read if observability fails

            return val
        except FileNotFoundError:
            raise KeyError(key)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve the content of a file in the database, or return a default value if not found.

        Parameters
        ----------
        key : str
            The key (filename) whose content is to be retrieved.
        default : Any, optional
            The default value to return if the file does not exist. Default is None.

        Returns
        -------
        Any
            The content of the file if it exists, a new DiskMemory instance if the key corresponds to a directory.
        """

        item_path = self.path / key
        try:
            if item_path.is_file():
                return self[key]
            elif item_path.is_dir():
                return DiskMemory(item_path)
            else:
                return default
        except:
            return default

    def __setitem__(self, key: Union[str, Path], val: str) -> None:
        """
        Set or update the content of a file in the database corresponding to the given key.

        Parameters
        ----------
        key : str or Path
            The key (filename) where the content is to be set.
        val : str
            The content to be written to the file.

        Raises
        ------
        ValueError
            If the key attempts to access a parent path.
        TypeError
            If the value is not a string.

        """
        if str(key).startswith("../"):
            raise ValueError(f"File name {key} attempted to access parent path.")

        if not isinstance(val, str):
            raise TypeError("val must be str")

        full_path = self.path / key
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to determine if this is create or update
        file_existed = full_path.exists()
        operation_type = "file_update" if file_existed else "file_create"

        full_path.write_text(val, encoding="utf-8")

        # Log file write operation if observability is available
        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
                if observability.is_enabled():
                    from uuid import uuid4

                    observability.log_event(
                        event_id=str(uuid4()),
                        event_type="file_write",
                        metadata={
                            "file_path": str(key),
                            "file_size": len(val),
                            "operation": operation_type,
                            "full_path": str(full_path),
                            "content_length": len(val),
                            "file_existed": file_existed,
                        },
                        tags={
                            "file_type": full_path.suffix or "no_extension",
                            "memory_path": str(self.path),
                            "operation": operation_type,
                            "file_size_category": "small"
                            if len(val) < 1000
                            else "medium"
                            if len(val) < 10000
                            else "large",
                        },
                    )
            except Exception:
                # Continue without observability if it fails
                pass

    def __delitem__(self, key: Union[str, Path]) -> None:
        """
        Delete a file or directory from the database corresponding to the given key.

        Parameters
        ----------
        key : str or Path
            The key (filename or directory name) to be deleted.

        Raises
        ------
        KeyError
            If the file or directory corresponding to the key does not exist in the database.

        """
        item_path = self.path / key
        if not item_path.exists():
            raise KeyError(f"Item '{key}' could not be found in '{self.path}'")

        if item_path.is_file():
            item_path.unlink()
        elif item_path.is_dir():
            shutil.rmtree(item_path)

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over the keys (filenames) in the database.

        Yields
        ------
        Iterator[str]
            An iterator over the sorted list of keys (filenames) in the database.

        """
        return iter(
            sorted(
                str(item.relative_to(self.path))
                for item in sorted(self.path.rglob("*"))
                if item.is_file()
            )
        )

    def __len__(self) -> int:
        """
        Get the number of files in the database.

        Returns
        -------
        int
            The number of files in the database.

        """
        return len(list(self.__iter__()))

    def _supported_files(self) -> str:
        valid_extensions = {
            ext for lang in SUPPORTED_LANGUAGES for ext in lang["extensions"]
        }
        file_paths = [
            str(item)
            for item in self
            if Path(item).is_file() and Path(item).suffix in valid_extensions
        ]
        result = "\n".join(file_paths)

        # Log file filtering query if observability is available
        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
                if observability.is_enabled():
                    # Removed log_retrieval call
                    pass
            except Exception:
                pass  # Don't fail operation if observability fails

        return result

    def _all_files(self) -> str:
        file_paths = [str(item) for item in self if Path(item).is_file()]
        result = "\n".join(file_paths)

        # Log file listing query if observability is available
        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
                if observability.is_enabled():
                    # Removed log_retrieval call
                    pass
            except Exception:
                pass  # Don't fail operation if observability fails

        return result

    def to_path_list_string(self, supported_code_files_only: bool = False) -> str:
        """
        Generate a string representation of the file paths in the database.

        Parameters
        ----------
        supported_code_files_only : bool, optional
            If True, filter the list to include only supported code file extensions.
            Default is False.

        Returns
        -------
        str
            A newline-separated string of file paths.

        """
        if supported_code_files_only:
            return self._supported_files()
        else:
            return self._all_files()

    def to_dict(self) -> Dict[Union[str, Path], str]:
        """
        Convert the database contents to a dictionary.

        Returns
        -------
        Dict[Union[str, Path], str]
            A dictionary with keys as filenames and values as file contents.

        """
        file_paths = list(self)
        result = {file_path: self[file_path] for file_path in file_paths}

        # Log bulk file read query if observability is available
        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
                if observability.is_enabled():
                    # Calculate total content size
                    sum(len(content) for content in result.values())

                    # Removed log_retrieval call
                    pass
            except Exception:
                pass  # Don't fail operation if observability fails

        return result

    def to_json(self) -> str:
        """
        Serialize the database contents to a JSON string.

        Returns
        -------
        str
            A JSON string representation of the database contents.

        """
        return json.dumps(self.to_dict())

    def log(self, key: Union[str, Path], val: str) -> None:
        """
        Append to a file or create and write to it if it doesn't exist.

        Parameters
        ----------
        key : str or Path
            The key (filename) where the content is to be appended.
        val : str
            The content to be appended to the file.

        """

        if str(key).startswith("../"):
            raise ValueError(f"File name {key} attempted to access parent path.")

        if not isinstance(val, str):
            raise TypeError("val must be str")

        full_path = self.path / "logs" / key
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Touch if it doesnt exist
        if not full_path.exists():
            full_path.touch()

        with open(full_path, "a", encoding="utf-8") as file:
            file.write(f"\n{datetime.now().isoformat()}\n")
            file.write(val + "\n")

    def archive_logs(self):
        """
        Moves all logs to archive directory based on current timestamp
        """
        if "logs" in self:
            archive_dir = (
                self.path / f"logs_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            )
            shutil.move(self.path / "logs", archive_dir)
