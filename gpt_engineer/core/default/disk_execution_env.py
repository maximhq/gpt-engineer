"""
Module for managing the execution environment on the local disk.

This module provides a class that handles the execution of code stored on the local
file system. It includes methods for uploading files to the execution environment,
running commands, and capturing the output.

Classes
-------
DiskExecutionEnv
    An execution environment that runs code on the local file system and captures
    the output of the execution.

Imports
-------
- subprocess: For running shell commands.
- time: For timing the execution of commands.
- Path: For handling file system paths.
- Optional, Tuple, Union: For type annotations.
- BaseExecutionEnv: For inheriting the base execution environment interface.
- FileStore: For managing file storage.
- FilesDict: For handling collections of files.
"""

import subprocess
import time

from pathlib import Path
from typing import Optional, Tuple, Union

from gpt_engineer.core.base_execution_env import BaseExecutionEnv
from gpt_engineer.core.default.file_store import FileStore
from gpt_engineer.core.files_dict import FilesDict

# Import observability
try:
    from gpt_engineer.core.maxim_observability import get_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


class DiskExecutionEnv(BaseExecutionEnv):
    """
    An execution environment that runs code on the local file system and captures
    the output of the execution.

    This class is responsible for executing code that is stored on disk. It ensures that
    the necessary entrypoint file exists and then runs the code using a subprocess. If the
    execution is interrupted by the user, it handles the interruption gracefully.

    Attributes
    ----------
    store : FileStore
        An instance of FileStore that manages the storage of files in the execution
        environment.
    """

    def __init__(self, path: Union[str, Path, None] = None):
        self.files = FileStore(path)

    def upload(self, files: FilesDict) -> "DiskExecutionEnv":
        self.files.push(files)
        return self

    def download(self) -> FilesDict:
        return self.files.pull()

    def popen(self, command: str) -> subprocess.Popen:
        p = subprocess.Popen(
            command,
            shell=True,
            cwd=self.files.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return p

    def run(self, command: str, timeout: Optional[int] = None) -> Tuple[str, str, int]:
        start = time.time()

        # Log tool call if observability is available
        tool_call_id = None
        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
                if observability.is_enabled():
                    from uuid import uuid4

                    tool_call_id = str(uuid4())
                    observability.log_tool_call(
                        tool_call_id=tool_call_id,
                        name="Execute Shell Command",
                        description=f"Execute shell command: {command}",
                        args={
                            "command": command,
                            "timeout": timeout,
                            "working_dir": str(self.files.working_dir),
                        },
                        tags={
                            "tool_type": "shell_execution",
                            "command_type": command.split()[0]
                            if command.split()
                            else "unknown",
                            "has_timeout": str(timeout is not None),
                        },
                    )
            except Exception:
                # Continue without observability if it fails
                pass

        print("\n--- Start of run ---")
        print("$", command)
        stdout_full, stderr_full = "", ""
        execution_error = None
        return_code = None

        try:
            # Use communicate() to capture all output reliably
            p = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.files.working_dir,
                text=True,
                shell=True,
            )

            # Use communicate with timeout if specified
            if timeout:
                try:
                    stdout_full, stderr_full = p.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    p.kill()
                    stdout_full, stderr_full = p.communicate()
                    execution_error = TimeoutError("Command execution timed out")
                    raise execution_error
            else:
                stdout_full, stderr_full = p.communicate()

            return_code = p.returncode

            # Print output in real-time fashion for user experience
            if stdout_full:
                print(stdout_full, end="")
            if stderr_full:
                print(stderr_full, end="")

        except KeyboardInterrupt:
            print()
            print("Stopping execution.")
            print("Execution stopped.")
            if "p" in locals():
                p.kill()
                stdout_full, stderr_full = p.communicate()
            print()
            print("--- Finished run ---\n")
            execution_error = KeyboardInterrupt("Execution interrupted by user")
            return_code = -1

        # Update the single tool call with result or error
        if OBSERVABILITY_AVAILABLE and tool_call_id:
            try:
                observability = get_observability()
                if observability.is_enabled():
                    execution_time = time.time() - start

                    if execution_error:
                        observability.log_tool_call(
                            tool_call_id=tool_call_id,
                            name="Command Execution",
                            description=f"Execute shell command: {command}",
                            args={
                                "command": command,
                                "timeout": timeout,
                                "working_dir": str(self.files.working_dir),
                                "execution_time": execution_time,
                            },
                            error=execution_error,
                            tags={
                                "tool_type": "shell_execution",
                                "command_type": command.split()[0]
                                if command.split()
                                else "unknown",
                                "has_timeout": str(timeout is not None),
                                "error_type": type(execution_error).__name__,
                            },
                        )
                    else:
                        result_summary = {
                            "return_code": return_code,
                            "execution_time": execution_time,
                            "stdout_length": len(stdout_full),
                            "stderr_length": len(stderr_full),
                            "stdout": stdout_full,
                            "stderr": stderr_full,
                            "success": return_code == 0,
                        }

                        observability.log_tool_call(
                            tool_call_id=tool_call_id,
                            name="Command Execution",
                            description=f"Execute shell command: {command}",
                            args={
                                "command": command,
                                "timeout": timeout,
                                "working_dir": str(self.files.working_dir),
                                "execution_time": execution_time,
                            },
                            result=result_summary,
                            tags={
                                "tool_type": "shell_execution",
                                "command_type": command.split()[0]
                                if command.split()
                                else "unknown",
                                "has_timeout": str(timeout is not None),
                                "return_code": str(return_code),
                                "success": str(return_code == 0),
                            },
                        )
            except Exception:
                # Continue without observability if it fails
                pass

        return stdout_full, stderr_full, return_code
