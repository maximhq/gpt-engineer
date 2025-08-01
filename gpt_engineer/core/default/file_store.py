import logging
import tempfile

from datetime import datetime
from pathlib import Path
from typing import Union

from gpt_engineer.core.files_dict import FilesDict
from gpt_engineer.core.linting import Linting

# Import observability
try:
    from gpt_engineer.core.maxim_observability import get_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_execution_env_path(path: Path) -> bool:
    """Check if the path is a gpt-engineer execution environment directory."""
    path_str = str(path)
    return path_str.startswith("/tmp/gpt-engineer-")


class FileStore:
    """
    Module for managing file storage in a temporary directory.

    This module provides a class that manages the storage of files in a temporary directory.
    It includes methods for uploading files to the directory and downloading them as a
    collection of files.

    Classes
    -------
    FileStore
        Manages file storage in a temporary directory, allowing for upload and download of files.

    Imports
    -------
    - tempfile: For creating temporary directories.
    - Path: For handling file system paths.
    - Union: For type annotations.
    - FilesDict: For handling collections of files.
    """

    def __init__(self, path: Union[str, Path, None] = None):
        if path is None:
            path = Path(tempfile.mkdtemp(prefix="gpt-engineer-"))

        self.working_dir = Path(path)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.id = self.working_dir.name.split("-")[-1]

    def push(self, files: FilesDict):
        # Only log if not in execution environment
        should_log = not is_execution_env_path(self.working_dir)

        # Initialize observability variables for file write span
        observability = None
        file_write_span_id = None

        if OBSERVABILITY_AVAILABLE and should_log:
            try:
                observability = get_observability()
                if observability.is_enabled():
                    from uuid import uuid4

                    # Start file write span
                    file_write_span_id = str(uuid4())
                    observability.start_span(
                        span_id=file_write_span_id,
                        name="File Write",
                        tags={
                            "operation": "file_write",
                            "file_count": str(len(files)),
                            "total_size": str(
                                sum(len(content) for content in files.values())
                            ),
                        },
                        metadata={
                            "file_names": list(files.keys()),
                            "file_types": list(
                                set(Path(name).suffix for name in files.keys())
                            ),
                            "working_directory": str(self.working_dir),
                        },
                    )

                    # Log the bulk file write operation as an event within the span
                    observability.log_event(
                        event_id=str(uuid4()),
                        event_type="bulk_file_write",
                        data={
                            "files_count": len(files),
                            "total_size": sum(
                                len(content) for content in files.values()
                            ),
                            "working_directory": str(self.working_dir),
                        },
                        tags={
                            "operation": "generated_files_write",
                            "file_count_category": "small"
                            if len(files) <= 5
                            else "medium"
                            if len(files) <= 20
                            else "large",
                        },
                        metadata={
                            "file_names": list(files.keys()),
                            "file_types": list(
                                set(Path(name).suffix for name in files.keys())
                            ),
                        },
                    )
            except Exception:
                # Continue without observability if it fails
                pass

        for name, content in files.items():
            # Skip directory entries and invalid file names
            if not name or name.endswith("/") or name.endswith("\\"):
                logger.warning(f"Skipping directory entry: {name}")
                continue

            # Skip entries with invalid content that look like directory descriptions
            if not content or content.strip().startswith("```") and "#### " in content:
                logger.warning(f"Skipping malformed file content for: {name}")
                continue

            path = self.working_dir / name

            # Create parent directories, but ensure we're not trying to write to a directory
            path.parent.mkdir(parents=True, exist_ok=True)

            # Double-check that this isn't a directory path
            if path.is_dir():
                logger.warning(f"Cannot write to directory path: {path}")
                continue

            # Check if file exists before writing
            file_existed = path.exists()

            try:
                with open(path, "w") as f:
                    f.write(content)
            except IsADirectoryError:
                logger.error(f"Attempted to write to directory: {path}")
                continue
            except Exception as e:
                logger.error(f"Failed to write file {path}: {e}")
                continue

            # Log individual file write and upload as attachment
            if OBSERVABILITY_AVAILABLE and should_log:
                try:
                    from uuid import uuid4

                    observability = get_observability()
                    if observability.is_enabled():
                        # Log as a tool call for file generation within the file write span
                        observability.log_tool_call(
                            tool_call_id=str(uuid4()),
                            name=f"File Generation: {path.name}",
                            description=f"Generate file content for {path.name}",
                            args={
                                "file_name": path.name,
                                "file_path": str(path),
                                "content_length": len(content),
                                "file_extension": path.suffix or "no_extension",
                                "generation_type": "ai_code_generation",
                                "file_content": content,  # Include content in args
                            },
                            result={
                                "file_generated": True,
                                "file_path": str(path),
                                "content_length": len(content),
                                "file_existed": file_existed,
                                "generation_successful": True,
                                "full_path": str(path),
                                "generation_timestamp": str(datetime.now()),
                                "generation_stage": "final_output",
                                "content_encoding": "utf-8",
                            },
                            tags={
                                "tool_type": "file_generation",
                                "operation": "file_generation",
                                "file_type": path.suffix or "no_extension",
                                "source": "ai_generation",
                                "content_size": str(len(content)),
                                "file_existed": str(file_existed),
                            },
                        )

                        if observability.enabled:
                            print(
                                f"✅ Maxim: Added generated file '{path.name}' to trace ({len(content)} bytes)"
                            )

                        # NOTE: We don't need to also add trace-level attachments
                        # as the retrieval already includes the file content attachment
                except Exception:
                    # Continue without observability if it fails
                    pass

        # End file write span
        if observability and observability.is_enabled() and file_write_span_id:
            try:
                observability.end_span(
                    span_id=file_write_span_id,
                    result={
                        "files_written": len(files),
                        "total_size": sum(len(content) for content in files.values()),
                    },
                )
            except Exception:
                # Continue even if span ending fails
                pass

        return self

    def linting(self, files: FilesDict) -> FilesDict:
        # lint the code
        linting = Linting()
        return linting.lint_files(files)

    def pull(self) -> FilesDict:
        files = {}
        for path in self.working_dir.glob("**/*"):
            if path.is_file():
                with open(path, "r") as f:
                    try:
                        content = f.read()
                    except UnicodeDecodeError:
                        content = "binary file"
                    files[str(path.relative_to(self.working_dir))] = content
        return FilesDict(files)
