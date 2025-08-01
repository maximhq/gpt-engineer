"""
Entrypoint for the CLI tool.

This module serves as the entry point for a command-line interface (CLI) tool.
It is designed to interact with OpenAI's language models.
The module provides functionality to:
- Load necessary environment variables,
- Configure various parameters for the AI interaction,
- Manage the generation or improvement of code projects.

Main Functionality
------------------
- Load environment variables required for OpenAI API interaction.
- Parse user-specified parameters for project configuration and AI behavior.
- Facilitate interaction with AI models, databases, and archival processes.

Parameters
----------
None

Notes
-----
- The `OPENAI_API_KEY` must be set in the environment or provided in a `.env` file within the working directory.
- The default project path is `projects/example`.
- When using the `azure_endpoint` parameter, provide the Azure OpenAI service endpoint URL.
"""

import difflib
import json
import logging
import os
import platform
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path

import openai
import typer

from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from termcolor import colored

from gpt_engineer.applications.cli.cli_agent import CliAgent
from gpt_engineer.applications.cli.collect import (
    collect_and_send_human_review,
    collect_learnings,
)
from gpt_engineer.applications.cli.file_selector import FileSelector
from gpt_engineer.applications.cli.learning import human_review_input
from gpt_engineer.applications.cli.multi_turn import run_multi_turn_mode
from gpt_engineer.core.ai import AI, ClipboardAI
from gpt_engineer.core.default.disk_execution_env import DiskExecutionEnv
from gpt_engineer.core.default.disk_memory import DiskMemory
from gpt_engineer.core.default.file_store import FileStore
from gpt_engineer.core.default.paths import PREPROMPTS_PATH, memory_path
from gpt_engineer.core.default.steps import (
    execute_entrypoint,
    gen_code,
    handle_improve_mode,
    improve_fn as improve_fn,
)
from gpt_engineer.core.files_dict import FilesDict
from gpt_engineer.core.git import stage_uncommitted_to_git
from gpt_engineer.core.observed_cache import ObservedSQLiteCache
from gpt_engineer.core.preprompts_holder import PrepromptsHolder
from gpt_engineer.core.prompt import Prompt
from gpt_engineer.tools.custom_steps import clarified_gen, lite_gen, self_heal

# Import observability
try:
    from gpt_engineer.core.maxim_observability import (
        get_observability,
        init_observability,
    )

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


@dataclass
class CliConfig:
    """Configuration class for CLI parameters."""

    project_path: str
    model: str
    temperature: float
    improve_mode: bool
    lite_mode: bool
    clarify_mode: bool
    self_heal_mode: bool
    multi_turn_mode: bool
    azure_endpoint: str
    use_custom_preprompts: bool
    llm_via_clipboard: bool
    verbose: bool
    debug: bool
    prompt_file: str
    entrypoint_prompt_file: str
    image_directory: str
    use_cache: bool
    skip_file_selection: bool
    no_execution: bool
    sysinfo: bool
    diff_timeout: int

    @property
    def mode(self) -> str:
        """Determine the operation mode based on flags."""
        if self.improve_mode:
            return "improve"
        elif self.clarify_mode:
            return "clarify"
        elif self.lite_mode:
            return "lite"
        elif self.self_heal_mode:
            return "self_heal"
        elif self.multi_turn_mode:
            return "multi_turn"
        else:
            return "generate"

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.improve_mode and (self.clarify_mode or self.lite_mode):
            raise typer.BadParameter(
                "Error: Clarify and lite mode are not compatible with improve mode."
            )

        # Multi-turn mode should not be combined with other specific modes
        if self.multi_turn_mode and (
            self.improve_mode
            or self.clarify_mode
            or self.lite_mode
            or self.self_heal_mode
        ):
            raise typer.BadParameter(
                "Error: Multi-turn mode is not compatible with other specific modes (improve, clarify, lite, self-heal)."
            )


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]}
)  # creates a CLI app

# logging.basicConfig(level=logging.DEBUG)


def load_env_if_needed():
    """
    Load environment variables if the OPENAI_API_KEY is not already set.

    This function checks if the OPENAI_API_KEY environment variable is set,
    and if not, it attempts to load it from a .env file in the current working
    directory. It then sets the openai.api_key for use in the application.
    """
    # We have all these checks for legacy reasons...
    if os.getenv("OPENAI_API_KEY") is None:
        load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None:
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

    openai.api_key = os.getenv("OPENAI_API_KEY")

    if os.getenv("ANTHROPIC_API_KEY") is None:
        load_dotenv()
    if os.getenv("ANTHROPIC_API_KEY") is None:
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))


def concatenate_paths(base_path, sub_path):
    # Compute the relative path from base_path to sub_path
    relative_path = os.path.relpath(sub_path, base_path)

    # If the relative path is not in the parent directory, use the original sub_path
    if not relative_path.startswith(".."):
        return sub_path

    # Otherwise, concatenate base_path and sub_path
    return os.path.normpath(os.path.join(base_path, sub_path))


def load_prompt(
    input_repo: DiskMemory,
    improve_mode: bool,
    prompt_file: str,
    image_directory: str,
    entrypoint_prompt_file: str = "",
) -> Prompt:
    """
    Load or request a prompt from the user based on the mode.

    Parameters
    ----------
    input_repo : DiskMemory
        The disk memory object where prompts and other data are stored.
    improve_mode : bool
        Flag indicating whether the application is in improve mode.

    Returns
    -------
    str
        The loaded or inputted prompt.
    """

    if os.path.isdir(prompt_file):
        raise ValueError(
            f"The path to the prompt, {prompt_file}, already exists as a directory. No prompt can be read from it. Please specify a prompt file using --prompt_file"
        )
    prompt_str = input_repo.get(prompt_file)
    if prompt_str:
        print(colored("Using prompt from file:", "green"), prompt_file)
        print(prompt_str)
    else:
        if not improve_mode:
            prompt_str = input(
                "\nWhat application do you want gpt-engineer to generate?\n"
            )
        else:
            prompt_str = input("\nHow do you want to improve the application?\n")

    if entrypoint_prompt_file == "":
        entrypoint_prompt = ""
    else:
        full_entrypoint_prompt_file = concatenate_paths(
            input_repo.path, entrypoint_prompt_file
        )
        if os.path.isfile(full_entrypoint_prompt_file):
            entrypoint_prompt = input_repo.get(full_entrypoint_prompt_file)

        else:
            raise ValueError("The provided file at --entrypoint-prompt does not exist")

    if image_directory == "":
        return Prompt(prompt_str, entrypoint_prompt=entrypoint_prompt)

    full_image_directory = concatenate_paths(input_repo.path, image_directory)
    if os.path.isdir(full_image_directory):
        if len(os.listdir(full_image_directory)) == 0:
            raise ValueError("The provided --image_directory is empty.")
        image_repo = DiskMemory(full_image_directory)
        image_dict = image_repo.get(".").to_dict()

        # Log image attachments if observability is available
        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
                if observability.is_enabled():
                    from uuid import uuid4

                    for image_name, image_data in image_dict.items():
                        # Log each image as an attachment/event
                        observability.log_event(
                            event_id=str(uuid4()),
                            event_type="image_attachment",
                            metadata={
                                "image_name": image_name,
                                "image_size": len(image_data),
                                "image_format": "base64_encoded",
                                "directory": image_directory,
                                "full_directory": full_image_directory,
                                "total_images": len(image_dict),
                            },
                            tags={
                                "attachment_type": "image",
                                "file_type": os.path.splitext(image_name)[1]
                                or "unknown",
                                "source": "image_directory",
                            },
                        )
            except Exception:
                # Continue without observability if it fails
                pass

        return Prompt(
            prompt_str,
            image_dict,
            entrypoint_prompt=entrypoint_prompt,
        )
    else:
        raise ValueError("The provided --image_directory is not a directory.")


def get_preprompts_path(use_custom_preprompts: bool, input_path: Path) -> Path:
    """
    Get the path to the preprompts, using custom ones if specified.

    Parameters
    ----------
    use_custom_preprompts : bool
        Flag indicating whether to use custom preprompts.
    input_path : Path
        The path to the project directory.

    Returns
    -------
    Path
        The path to the directory containing the preprompts.
    """
    original_preprompts_path = PREPROMPTS_PATH
    if not use_custom_preprompts:
        return original_preprompts_path

    custom_preprompts_path = input_path / "preprompts"
    if not custom_preprompts_path.exists():
        custom_preprompts_path.mkdir()

    for file in original_preprompts_path.glob("*"):
        if not (custom_preprompts_path / file.name).exists():
            (custom_preprompts_path / file.name).write_text(file.read_text())
    return custom_preprompts_path


def compare(f1: FilesDict, f2: FilesDict):
    def colored_diff(s1, s2):
        lines1 = s1.splitlines()
        lines2 = s2.splitlines()

        diff = difflib.unified_diff(lines1, lines2, lineterm="")

        RED = "\033[38;5;202m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        colored_lines = []
        for line in diff:
            if line.startswith("+"):
                colored_lines.append(GREEN + line + RESET)
            elif line.startswith("-"):
                colored_lines.append(RED + line + RESET)
            else:
                colored_lines.append(line)

        return "\n".join(colored_lines)

    for file in sorted(set(f1) | set(f2)):
        diff = colored_diff(f1.get(file, ""), f2.get(file, ""))
        if diff:
            print(f"Changes to {file}:")
            print(diff)


def prompt_yesno() -> bool:
    TERM_CHOICES = colored("y", "green") + "/" + colored("n", "red") + " "
    while True:
        response = input(TERM_CHOICES).strip().lower()
        if response in ["y", "yes"]:
            return True
        if response in ["n", "no"]:
            break
        print("Please respond with 'y' or 'n'")


def get_system_info():
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": sys.version,
        "packages": format_installed_packages(get_installed_packages()),
    }
    return system_info


def get_installed_packages():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
        )
        packages = json.loads(result.stdout)
        return {pkg["name"]: pkg["version"] for pkg in packages}
    except Exception as e:
        return {"error": str(e)}


def format_installed_packages(packages):
    if isinstance(packages, dict) and "error" not in packages:
        return "\n".join([f"{name}: {version}" for name, version in packages.items()])
    elif isinstance(packages, dict) and "error" in packages:
        return f"Error getting packages: {packages['error']}"
    else:
        return str(packages)


def execute_gpt_engineer(config: CliConfig) -> None:
    """
    Core execution logic for GPT-engineer CLI.

    This function contains all the main logic for running GPT-engineer,
    including AI initialization, project processing, and observability.

    Parameters
    ----------
    config : CliConfig
        Configuration object containing all CLI parameters.

    Returns
    -------
    None
    """
    if config.debug:
        import pdb

        sys.excepthook = lambda *_: pdb.pm()

    if config.sysinfo:
        sys_info = get_system_info()
        for key, value in sys_info.items():
            print(f"{key}: {value}")
        raise typer.Exit()

    # Validate arguments
    config.validate()

    # Set up logging
    # logging.basicConfig(level=logging.DEBUG if config.verbose else logging.INFO)
    if config.use_cache:
        set_llm_cache(ObservedSQLiteCache(database_path=".langchain.db"))
    if config.improve_mode:
        assert not (
            config.clarify_mode or config.lite_mode
        ), "Clarify and lite mode are not active for improve mode"

    # Initialize observability
    observability = None
    session_id = None
    trace_id = None

    if OBSERVABILITY_AVAILABLE:
        try:
            from uuid import uuid4

            observability = init_observability(enabled=True)

            if observability.is_enabled():
                # Create session for this CLI invocation
                session_id = str(uuid4())

                session_tags = {
                    "project_path": str(Path(config.project_path).absolute()),
                    "model": config.model,
                    "mode": config.mode,
                    "temperature": str(config.temperature),
                    "python_version": sys.version.split()[0],
                    "gpt_engineer_invocation": "cli",
                }

                session_metadata = {
                    "cli_args": {
                        "project_path": config.project_path,
                        "model": config.model,
                        "temperature": config.temperature,
                        "improve_mode": config.improve_mode,
                        "lite_mode": config.lite_mode,
                        "clarify_mode": config.clarify_mode,
                        "self_heal_mode": config.self_heal_mode,
                        "prompt_file": config.prompt_file,
                        "image_directory": config.image_directory,
                        "use_cache": config.use_cache,
                        "verbose": config.verbose,
                        "debug": config.debug,
                    },
                    "system_info": get_system_info(),
                }

                observability.start_session(
                    session_id=session_id, tags=session_tags, metadata=session_metadata
                )

                print(f"Started Maxim session: {session_id}")

        except Exception as e:
            logging.warning(f"Failed to initialize observability: {e}")

    load_env_if_needed()

    if config.llm_via_clipboard:
        ai = ClipboardAI()
    else:
        ai = AI(
            model_name=config.model,
            temperature=config.temperature,
            azure_endpoint=config.azure_endpoint,
        )

    path = Path(config.project_path)
    print("Running gpt-engineer in", path.absolute(), "\n")

    prompt = load_prompt(
        DiskMemory(path),
        config.improve_mode,
        config.prompt_file,
        config.image_directory,
        config.entrypoint_prompt_file,
    )

    # todo: if ai.vision is false and not llm_via_clipboard - ask if they would like to use gpt-4-vision-preview instead? If so recreate AI
    if not ai.vision:
        prompt.image_urls = None

    preprompts_holder = PrepromptsHolder(
        get_preprompts_path(config.use_custom_preprompts, Path(config.project_path))
    )

    # Handle multi-turn mode
    if config.multi_turn_mode:
        print(colored("🔄 Multi-turn mode activated", "green"))
        run_multi_turn_mode(ai, config.project_path, preprompts_holder, prompt)
        return

    # configure generation function
    if config.clarify_mode:
        code_gen_fn = clarified_gen
    elif config.lite_mode:
        code_gen_fn = lite_gen
    else:
        code_gen_fn = gen_code

    # configure execution function
    if config.self_heal_mode:
        execution_fn = self_heal
    else:
        execution_fn = execute_entrypoint

    memory = DiskMemory(memory_path(config.project_path))
    memory.archive_logs()

    execution_env = DiskExecutionEnv()
    agent = CliAgent.with_default_config(
        memory,
        execution_env,
        ai=ai,
        code_gen_fn=code_gen_fn,
        improve_fn=improve_fn,
        process_code_fn=execution_fn,
        preprompts_holder=preprompts_holder,
    )

    files = FileStore(config.project_path)

    try:
        # Start main trace for the operation
        if observability and observability.is_enabled():
            from uuid import uuid4

            trace_id = str(uuid4())

            operation_name = (
                "Code Improvement" if config.improve_mode else "Code Generation"
            )
            if config.clarify_mode:
                operation_name = "Clarify and Generate"
            elif config.lite_mode:
                operation_name = "Lite Generation"
            elif config.self_heal_mode:
                operation_name = "Self-Healing Generation"

            trace_tags = {
                "operation": operation_name.lower().replace(" ", "_"),
                "code_gen_function": code_gen_fn.__name__,
                "execution_function": execution_fn.__name__,
                "model": config.model,
                "project_path": str(Path(config.project_path).absolute()),
                "prompt_file": config.prompt_file,
            }

            trace_metadata = {
                "prompt_preview": prompt.text[:200] + "..."
                if len(prompt.text) > 200
                else prompt.text,
                "prompt_length": len(prompt.text),
                "has_images": bool(prompt.image_urls),
                "image_count": len(prompt.image_urls) if prompt.image_urls else 0,
            }

            observability.start_trace(
                trace_id=trace_id,
                name=operation_name,
                tags=trace_tags,
                metadata=trace_metadata,
                session_id=session_id,
            )

            # Set the prompt as trace input
            observability.set_trace_input(prompt.text)

        # Initialize files_dict to prevent UnboundLocalError in exception cases
        files_dict = {}

        if not config.no_execution:
            if config.improve_mode:
                # Start improve mode span
                improve_span_id = None
                if observability and observability.is_enabled():
                    improve_span_id = str(uuid4())
                    observability.start_span(
                        span_id=improve_span_id,
                        name="Code Improvement Process",
                        tags={
                            "operation": "improve_mode",
                            "skip_file_selection": str(config.skip_file_selection),
                        },
                        metadata={"diff_timeout": config.diff_timeout},
                    )

                try:
                    # File selection span
                    file_selection_span_id = None
                    if observability and observability.is_enabled():
                        file_selection_span_id = str(uuid4())
                        observability.start_span(
                            parent_span_id=improve_span_id,
                            span_id=file_selection_span_id,
                            name="File Selection",
                            tags={
                                "operation": "file_selection",
                                "skip_selection": str(config.skip_file_selection),
                            },
                        )

                    try:
                        files_dict_before, is_linting = FileSelector(
                            config.project_path
                        ).ask_for_files(skip_file_selection=config.skip_file_selection)

                        # Initialize files_dict with the before state to ensure it's always available
                        files_dict = files_dict_before

                    except Exception as file_selection_error:
                        print(f"⚠️ File selection failed: {file_selection_error}")
                        print("Using empty files dictionary as fallback.")
                        files_dict_before = {}
                        files_dict = {}
                        is_linting = False

                        # Log the file selection error
                        if observability and observability.is_enabled():
                            observability.log_error(
                                error=file_selection_error,
                                context={
                                    "operation": "file_selection",
                                    "project_path": config.project_path,
                                    "skip_file_selection": config.skip_file_selection,
                                },
                                tags={"error_type": "file_selection_error"},
                            )

                    if (
                        observability
                        and observability.is_enabled()
                        and file_selection_span_id
                    ):
                        observability.end_span(
                            span_id=file_selection_span_id,
                            result={
                                "files_selected": len(files_dict_before),
                                "linting_enabled": is_linting,
                            },
                        )

                    # lint the code
                    if is_linting:
                        linting_span_id = None
                        if observability and observability.is_enabled():
                            linting_span_id = str(uuid4())
                            observability.start_span(
                                parent_span_id=improve_span_id,
                                span_id=linting_span_id,
                                name="Code Linting",
                                tags={"operation": "linting"},
                            )
                            observability.log_event(
                                event_id=str(uuid4()),
                                event_type="linting_started",
                                metadata={
                                    "file_count": len(files_dict_before),
                                    "files": list(files_dict_before.keys()),
                                },
                                tags={"operation": "linting"},
                            )

                        files_dict_before = files.linting(files_dict_before)

                        if (
                            observability
                            and observability.is_enabled()
                            and linting_span_id
                        ):
                            observability.end_span(
                                span_id=linting_span_id,
                                result={"files_linted": len(files_dict_before)},
                            )

                    # Improvement process span
                    improvement_span_id = None
                    if observability and observability.is_enabled():
                        improvement_span_id = str(uuid4())
                        observability.start_span(
                            parent_span_id=improve_span_id,
                            span_id=improvement_span_id,
                            name="AI Code Improvement",
                            tags={
                                "operation": "ai_improvement",
                                "model": config.model,
                                "diff_timeout": str(config.diff_timeout),
                            },
                            metadata={
                                "files_before_count": len(files_dict_before),
                                "total_size_before": sum(
                                    len(content)
                                    for content in files_dict_before.values()
                                ),
                            },
                        )

                    try:
                        improved_files_dict = handle_improve_mode(
                            prompt,
                            agent,
                            memory,
                            files_dict_before,
                            diff_timeout=config.diff_timeout,
                            parent_span_id=improvement_span_id,
                        )
                        # Only update files_dict if improvement succeeded
                        if improved_files_dict is not None:
                            files_dict = improved_files_dict
                        else:
                            print(
                                "⚠️ No improvements were generated. Keeping original files."
                            )
                    except Exception as improvement_error:
                        print(f"⚠️ Improvement process failed: {improvement_error}")
                        print("Keeping original files.")
                        # files_dict remains as files_dict_before, which was set earlier

                        # Log the improvement error
                        if observability and observability.is_enabled():
                            observability.log_error(
                                error=improvement_error,
                                context={
                                    "operation": "ai_improvement",
                                    "model": config.model,
                                    "files_before_count": len(files_dict_before),
                                },
                                tags={"error_type": "improvement_error"},
                            )

                    if (
                        observability
                        and observability.is_enabled()
                        and improvement_span_id
                    ):
                        observability.end_span(
                            span_id=improvement_span_id,
                            result={
                                "files_after_count": len(files_dict)
                                if files_dict
                                else 0,
                                "changes_made": files_dict != files_dict_before
                                if files_dict
                                else False,
                                "improvement_success": files_dict != files_dict_before,
                            },
                        )

                    if not files_dict or files_dict_before == files_dict:
                        print(
                            f"No changes applied. Could you please upload the debug_log_file.txt in {memory.path}/logs folder in a github issue?"
                        )
                    else:
                        print("\nChanges to be made:")
                        compare(files_dict_before, files_dict)

                        print()
                        print(
                            colored(
                                "Do you want to apply these changes?", "light_green"
                            )
                        )
                        if not prompt_yesno():
                            files_dict = files_dict_before

                        # Log user feedback
                        if observability and observability.is_enabled():
                            feedback_result = (
                                "approved"
                                if files_dict != files_dict_before
                                else "rejected"
                            )
                            observability.log_event(
                                event_id=str(uuid4()),
                                event_type="user_approval",
                                metadata={
                                    "result": feedback_result,
                                    "changes_count": len(files_dict),
                                },
                                tags={
                                    "event_type": "user_decision",
                                    "result": feedback_result,
                                },
                            )

                finally:
                    # End improve mode span
                    if observability and observability.is_enabled() and improve_span_id:
                        observability.end_span(
                            span_id=improve_span_id,
                            result={
                                "improvement_completed": True,
                                "final_file_count": len(files_dict)
                                if files_dict
                                else 0,
                            },
                        )

            else:
                files_dict = agent.init(prompt)

            stage_uncommitted_to_git(path, files_dict, config.improve_mode)

            # Start finalization span to capture file operations and user feedback
            finalization_span_id = None
            if observability and observability.is_enabled():
                from uuid import uuid4

                finalization_span_id = str(uuid4())
                observability.start_span(
                    span_id=finalization_span_id,
                    name="Finalization",
                    tags={
                        "operation": "finalization",
                        "file_count": str(len(files_dict)),
                        "mode": "improve" if config.improve_mode else "generate",
                    },
                    metadata={
                        "total_file_size": sum(
                            len(content) for content in files_dict.values()
                        ),
                        "file_types": list(
                            set(Path(fname).suffix for fname in files_dict.keys())
                        ),
                    },
                )

            try:
                # File operations happen within finalization span
                files.push(files_dict)

                # User feedback collection happens within finalization span
                if not config.improve_mode:
                    # print(f"[MaximObservability] Adding feedback to trace {finalization_span_id}")
                    config_tuple = (code_gen_fn.__name__, execution_fn.__name__)
                    # print(f"[MaximObservability] config: {config_tuple}")
                    # if observability and observability.is_enabled():
                    #     print(f"[MaximObservability] Adding feedback to trace {trace_id}")
                    #     observability.add_feedback()
                    review = human_review_input(multi_turn=config.multi_turn_mode)
                    if review:
                        collect_learnings(
                            prompt,
                            config.model,
                            config.temperature,
                            config_tuple,
                            memory,
                            review,
                        )
                        if observability and observability.is_enabled():
                            observability.add_feedback(review)
                    else:
                        collect_and_send_human_review(
                            prompt,
                            config.model,
                            config.temperature,
                            config_tuple,
                            memory,
                        )

            finally:
                # End finalization span
                if (
                    observability
                    and observability.is_enabled()
                    and finalization_span_id
                ):
                    observability.end_span(
                        span_id=finalization_span_id,
                        result={
                            "files_written": len(files_dict),
                            "finalization_completed": True,
                        },
                    )

            # Set comprehensive final result as trace output
            if observability and observability.is_enabled():
                from datetime import datetime

                # Calculate comprehensive metrics
                total_size = sum(len(content) for content in files_dict.values())
                list(set(Path(fname).suffix for fname in files_dict.keys()))

                # Organize files by type
                files_by_type = {}
                for fname in files_dict.keys():
                    ext = Path(fname).suffix or "no_extension"
                    if ext not in files_by_type:
                        files_by_type[ext] = []
                    files_by_type[ext].append(fname)

                # Cost and token information
                cost_info = {}
                token_info = {}
                if ai.token_usage_log.is_openai_model():
                    cost_info = {
                        "total_cost": ai.token_usage_log.usage_cost(),
                        "currency": "USD",
                        "model": config.model,
                    }
                    token_info = {
                        "total_tokens": ai.token_usage_log.total_tokens(),
                        "cost_per_token": ai.token_usage_log.usage_cost()
                        / max(ai.token_usage_log.total_tokens(), 1),
                    }

                # Create markdown summary once and reuse
                markdown_summary = f"""# 🎯 {operation_name} Summary

## 📊 Overview
- **Operation**: {operation_name} ({'✅ Completed' if True else '❌ Failed'})
- **Files Generated**: {len(files_dict)} files ({total_size:,} characters)
- **Model Used**: {config.model}
- **Cost**: ${cost_info.get('total_cost', 0):.4f} ({token_info.get('total_tokens', 0):,} tokens)

## 📁 Files by Type
{chr(10).join([f'- **{ext if ext != "no_extension" else "other"}**: {", ".join(files)}' for ext, files in files_by_type.items()])}

## 🎯 Quality Assessment
- **Entry Point**: {'✅ Yes' if any(f in ['main.py', 'app.py', 'index.js', 'index.html'] for f in files_dict.keys()) else '⚠️ No'}
- **Dependencies**: {'✅ Yes' if any(f in ['requirements.txt', 'package.json', 'Pipfile'] for f in files_dict.keys()) else '⚠️ No'}
- **Tests**: {'✅ Yes' if any('test' in f.lower() for f in files_dict.keys()) else '❌ No'}
- **Run Script**: {'✅ Yes' if 'run.sh' in files_dict or any('run' in f for f in files_dict.keys()) else '⚠️ No'}
- **Documentation**: {'✅ Yes' if any(f.lower().endswith(('.md', '.rst')) for f in files_dict.keys()) else '❌ No'}

## 💰 Performance Metrics
- **Cost Efficiency**: ${cost_info.get('total_cost', 0) / max(total_size, 1) * 1000:.4f} per KB
- **Model**: {config.model}
- **Temperature**: {config.temperature}

## 📝 Project Structure
- **Has Proper Structure**: {'✅ Yes' if len([f for f in files_dict.keys() if '/' in f]) > 0 else '❌ No'}
- **Follows Conventions**: {'✅ Yes' if any(f.endswith('.py') for f in files_dict.keys()) and 'requirements.txt' in files_dict else '❌ No'}
- **Estimated Completeness**: {min(100, len(files_dict) * 20)}%

## 🚀 Context
- **Project Path**: {str(Path(config.project_path).absolute())}
- **Prompt Length**: {len(prompt.text)} characters
- **Complexity**: {'High' if len(prompt.text) > 1000 else 'Medium' if len(prompt.text) > 300 else 'Simple'}
- **Execution Time**: {datetime.now().isoformat()}

---
*Generated by GPT-Engineer v0.3*
"""

                observability.set_trace_output(markdown_summary)
                print(f"🔍 Trace Output Set: {markdown_summary}")

            # Display comprehensive task summary to user
            print("\n" + "=" * 70)
            print(colored("📋 TASK SUMMARY", "cyan", attrs=["bold"]))
            print("=" * 70)

            # What was accomplished
            operation_type = (
                "Code Improvement" if config.improve_mode else "Code Generation"
            )
            print(
                f"✅ {colored('Operation:', 'green')} {operation_type} completed successfully"
            )

            # Files generated/modified
            total_size = sum(len(content) for content in files_dict.values())
            file_count = len(files_dict)
            print(
                f"📁 {colored('Files:', 'blue')} {file_count} files ({total_size:,} characters total)"
            )

            # Organize and display files by type
            files_by_type = {}
            for fname in files_dict.keys():
                ext = Path(fname).suffix or "no_extension"
                if ext not in files_by_type:
                    files_by_type[ext] = []
                files_by_type[ext].append(fname)

            for file_type, files in files_by_type.items():
                type_display = file_type if file_type != "no_extension" else "other"
                print(f"   • {colored(type_display, 'yellow')}: {', '.join(files)}")

            # Project quality assessment
            print(f"\n🎯 {colored('Quality Assessment:', 'green')}")

            # Check for key project components
            has_main = any(
                f in ["main.py", "app.py", "index.js", "index.html"]
                for f in files_dict.keys()
            )
            has_tests = any("test" in f.lower() for f in files_dict.keys())
            has_deps = any(
                f in ["requirements.txt", "package.json", "Pipfile"]
                for f in files_dict.keys()
            )
            has_runner = "run.sh" in files_dict or any(
                "run" in f for f in files_dict.keys()
            )
            has_docs = any(
                f.lower().endswith((".md", ".rst")) for f in files_dict.keys()
            )

            components = [
                ("Entry Point", has_main, "✅" if has_main else "⚠️"),
                ("Dependencies", has_deps, "✅" if has_deps else "⚠️"),
                ("Tests", has_tests, "✅" if has_tests else "❌"),
                ("Run Script", has_runner, "✅" if has_runner else "⚠️"),
                ("Documentation", has_docs, "✅" if has_docs else "❌"),
            ]

            for component, present, icon in components:
                print(f"   {icon} {component}: {'Yes' if present else 'No'}")

            # Estimate completeness score
            score = sum([has_main, has_deps, has_tests, has_runner, has_docs])
            completeness = min(100, (score * 20) + (file_count * 5))

            if completeness >= 80:
                color = "green"
                status = "Excellent"
            elif completeness >= 60:
                color = "yellow"
                status = "Good"
            else:
                color = "red"
                status = "Basic"

            print(
                f"   📊 Estimated Completeness: {colored(f'{completeness}% ({status})', color)}"
            )

            # Performance metrics
            if ai.token_usage_log.is_openai_model():
                cost = ai.token_usage_log.usage_cost()
                tokens = ai.token_usage_log.total_tokens()
                efficiency = cost / max(total_size, 1) * 1000  # cost per KB

                print(f"\n💰 {colored('Performance Metrics:', 'blue')}")
                print(f"   • Cost: ${cost:.4f} ({tokens:,} tokens)")
                print(f"   • Efficiency: ${efficiency:.4f} per KB of code")
                print(f"   • Model: {config.model}")

            # Recommendations
            print(f"\n🎯 {colored('Recommendations:', 'magenta')}")
            recommendations = []

            if not has_tests:
                recommendations.append("Add unit tests to improve code reliability")
            if not has_docs and file_count > 3:
                recommendations.append("Consider adding a README.md for documentation")
            if not has_main and file_count > 1:
                recommendations.append("Consider adding a clear entry point (main.py)")
            if total_size < 500:
                recommendations.append(
                    "Project seems minimal - consider expanding functionality"
                )
            elif total_size > 10000:
                recommendations.append(
                    "Large project - consider splitting into modules"
                )

            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            else:
                print(
                    f"   ✨ {colored('Great work! The generated code follows best practices.', 'green')}"
                )

            # Next steps
            print(f"\n🚀 {colored('Next Steps:', 'cyan')}")
            if has_runner:
                print("   1. Run the code using the generated script")
                print("   2. Test the functionality with different inputs")
            else:
                print("   1. Test the generated code manually")

            if has_tests:
                print("   3. Run the test suite to verify functionality")
            else:
                print("   3. Consider writing tests for the code")

            print("   4. Review and customize the code as needed")

            print("=" * 70)

            # Start task summary span
            task_summary_span_id = None
            if observability and observability.is_enabled():
                from uuid import uuid4

                task_summary_span_id = str(uuid4())
                observability.start_span(
                    span_id=task_summary_span_id,
                    name="Task Summary",
                    tags={
                        "operation": "task_summary",
                        "mode": operation_name.lower().replace(" ", "_"),
                        "file_count": str(len(files_dict)),
                        "model": config.model,
                    },
                    metadata={
                        "summary_generation_timestamp": str(datetime.now()),
                        "operation_type": operation_type,
                        "project_path": str(Path(config.project_path).absolute()),
                        "prompt_length": len(prompt.text),
                        "prompt_complexity": "High"
                        if len(prompt.text) > 1000
                        else "Medium"
                        if len(prompt.text) > 300
                        else "Simple",
                        "quality_indicators": {
                            "has_entry_point": any(
                                f in ["main.py", "app.py", "index.js", "index.html"]
                                for f in files_dict.keys()
                            ),
                            "has_dependencies": any(
                                f in ["requirements.txt", "package.json", "Pipfile"]
                                for f in files_dict.keys()
                            ),
                            "has_tests": any(
                                "test" in f.lower() for f in files_dict.keys()
                            ),
                            "has_documentation": any(
                                f.lower().endswith((".md", ".rst"))
                                for f in files_dict.keys()
                            ),
                            "has_run_script": "run.sh" in files_dict
                            or any("run" in f for f in files_dict.keys()),
                        },
                        "performance_metrics": {
                            "total_cost": cost_info.get("total_cost", 0),
                            "total_tokens": token_info.get("total_tokens", 0),
                            "cost_efficiency": cost_info.get("total_cost", 0)
                            / max(total_size, 1)
                            * 1000,
                            "estimated_completeness": min(100, len(files_dict) * 20),
                        },
                    },
                )

            # --- Generate summary markdown ---
            try:
                # --- Observability: summary_markdown_generated ---
                if (
                    observability
                    and observability.is_enabled()
                    and task_summary_span_id
                ):
                    observability.log_event(
                        event_id=str(uuid4()),
                        event_type="summary_markdown_generated",
                        metadata={"summary_length": len(markdown_summary)},
                        tags={"event_type": "summary_markdown_generated"},
                    )

                # --- Set as trace output ---
                if observability and observability.is_enabled():
                    observability.set_trace_output(markdown_summary)
                    print(f"🔍 Trace Output Set: {markdown_summary}")

                # --- Display summary to user ---
                print("\n" + "=" * 70)
                print(colored("📋 TASK SUMMARY", "cyan", attrs=["bold"]))
                print("=" * 70)
                print(
                    f"✅ {colored('Operation:', 'green')} {operation_type} completed successfully"
                )
                print(
                    f"📁 {colored('Files:', 'blue')} {file_count} files ({total_size:,} characters total)"
                )
                for file_type, files in files_by_type.items():
                    type_display = file_type if file_type != "no_extension" else "other"
                    print(f"   • {colored(type_display, 'yellow')}: {', '.join(files)}")
                print(f"\n🎯 {colored('Quality Assessment:', 'green')}")
                for component, present, icon in components:
                    print(f"   {icon} {component}: {'Yes' if present else 'No'}")
                print(
                    f"   📊 Estimated Completeness: {colored(f'{completeness}% ({status})', color)}"
                )
                if ai.token_usage_log.is_openai_model():
                    print(f"\n💰 {colored('Performance Metrics:', 'blue')}")
                    print(f"   • Cost: ${cost:.4f} ({tokens:,} tokens)")
                    print(f"   • Efficiency: ${efficiency:.4f} per KB of code")
                    print(f"   • Model: {config.model}")
                print(f"\n🎯 {colored('Recommendations:', 'magenta')}")
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        print(f"   {i}. {rec}")
                else:
                    print(
                        f"   ✨ {colored('Great work! The generated code follows best practices.', 'green')}"
                    )
                print(f"\n🚀 {colored('Next Steps:', 'cyan')}")
                if has_runner:
                    print("   1. Run the code using the generated script")
                    print("   2. Test the functionality with different inputs")
                else:
                    print("   1. Test the generated code manually")
                if has_tests:
                    print("   3. Run the test suite to verify functionality")
                else:
                    print("   3. Consider writing tests for the code")
                print("   4. Review and customize the code as needed")
                print("=" * 70)

                # --- Observability: summary_displayed_to_user ---
                if (
                    observability
                    and observability.is_enabled()
                    and task_summary_span_id
                ):
                    observability.log_event(
                        event_id=str(uuid4()),
                        event_type="summary_displayed_to_user",
                        metadata={"displayed": True},
                        tags={"event_type": "summary_displayed_to_user"},
                    )
            except Exception as summary_error:
                # --- Observability: summary_error ---
                if (
                    observability
                    and observability.is_enabled()
                    and task_summary_span_id
                ):
                    observability.log_event(
                        event_id=str(uuid4()),
                        event_type="summary_error",
                        metadata={"error": str(summary_error)},
                        tags={"event_type": "summary_error"},
                    )
                raise

        if ai.token_usage_log.is_openai_model():
            print("Total api cost: $ ", ai.token_usage_log.usage_cost())
        elif os.getenv("LOCAL_MODEL"):
            print("Total api cost: $ 0.0 since we are using local LLM.")
        else:
            print("Total tokens used: ", ai.token_usage_log.total_tokens())

    except Exception as e:
        # Log any errors that occur
        if observability and observability.is_enabled():
            observability.log_error(
                error=e,
                context={
                    "operation": "main_execution",
                    "improve_mode": config.improve_mode,
                    "model": config.model,
                    "project_path": config.project_path,
                },
                tags={"error_type": "main_execution_error"},
            )
        raise

    finally:
        # Cleanup observability resources
        if observability and observability.is_enabled():
            try:
                # End trace
                if trace_id:
                    observability.end_trace(trace_id)

                # End session
                if session_id:
                    observability.end_session()

                # Cleanup SDK
                observability.cleanup()
                print("Maxim observability cleanup completed")

            except Exception as e:
                logging.warning(f"Failed to cleanup observability: {e}")


def create_config_from_args(
    project_path: str,
    model: str,
    temperature: float,
    improve_mode: bool,
    lite_mode: bool,
    clarify_mode: bool,
    self_heal_mode: bool,
    multi_turn_mode: bool,
    azure_endpoint: str,
    use_custom_preprompts: bool,
    llm_via_clipboard: bool,
    verbose: bool,
    debug: bool,
    prompt_file: str,
    entrypoint_prompt_file: str,
    image_directory: str,
    use_cache: bool,
    skip_file_selection: bool,
    no_execution: bool,
    sysinfo: bool,
    diff_timeout: int,
) -> CliConfig:
    """
    Create a CliConfig object from the command line arguments.

    Parameters
    ----------
    All parameters are the same as the main function parameters.

    Returns
    -------
    CliConfig
        Configuration object containing all the parameters.
    """
    return CliConfig(
        project_path=project_path,
        model=model,
        temperature=temperature,
        improve_mode=improve_mode,
        lite_mode=lite_mode,
        clarify_mode=clarify_mode,
        self_heal_mode=self_heal_mode,
        multi_turn_mode=multi_turn_mode,
        azure_endpoint=azure_endpoint,
        use_custom_preprompts=use_custom_preprompts,
        llm_via_clipboard=llm_via_clipboard,
        verbose=verbose,
        debug=debug,
        prompt_file=prompt_file,
        entrypoint_prompt_file=entrypoint_prompt_file,
        image_directory=image_directory,
        use_cache=use_cache,
        skip_file_selection=skip_file_selection,
        no_execution=no_execution,
        sysinfo=sysinfo,
        diff_timeout=diff_timeout,
    )


@app.command(
    help="""
        GPT-engineer lets you:

        \b
        - Specify a software in natural language
        - Sit back and watch as an AI writes and executes the code
        - Ask the AI to implement improvements
    """
)
def main(
    project_path: str = typer.Argument(".", help="path"),
    model: str = typer.Option(
        os.environ.get("MODEL_NAME", "gpt-4o"), "--model", "-m", help="model id string"
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature",
        "-t",
        help="Controls randomness: lower values for more focused, deterministic outputs",
    ),
    improve_mode: bool = typer.Option(
        False,
        "--improve",
        "-i",
        help="Improve an existing project by modifying the files.",
    ),
    lite_mode: bool = typer.Option(
        False,
        "--lite",
        "-l",
        help="Lite mode: run a generation using only the main prompt.",
    ),
    clarify_mode: bool = typer.Option(
        False,
        "--clarify",
        "-c",
        help="Clarify mode - discuss specification with AI before implementation.",
    ),
    self_heal_mode: bool = typer.Option(
        False,
        "--self-heal",
        "-sh",
        help="Self-heal mode - fix the code by itself when it fails.",
    ),
    multi_turn_mode: bool = typer.Option(
        False,
        "--multi-turn",
        "-mt",
        help="Multi-turn mode - engage in a conversation with the AI that dynamically determines the appropriate mode for each interaction.",
    ),
    azure_endpoint: str = typer.Option(
        "",
        "--azure",
        "-a",
        help="""Endpoint for your Azure OpenAI Service (https://xx.openai.azure.com).
            In that case, the given model is the deployment name chosen in the Azure AI Studio.""",
    ),
    use_custom_preprompts: bool = typer.Option(
        False,
        "--use-custom-preprompts",
        help="""Use your project's custom preprompts instead of the default ones.
          Copies all original preprompts to the project's workspace if they don't exist there.""",
    ),
    llm_via_clipboard: bool = typer.Option(
        False,
        "--llm-via-clipboard",
        help="Use the clipboard to communicate with the AI.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging for debugging."
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug mode for debugging."
    ),
    prompt_file: str = typer.Option(
        "prompt",
        "--prompt_file",
        help="Relative path to a text file containing a prompt.",
    ),
    entrypoint_prompt_file: str = typer.Option(
        "",
        "--entrypoint_prompt",
        help="Relative path to a text file containing a file that specifies requirements for you entrypoint.",
    ),
    image_directory: str = typer.Option(
        "",
        "--image_directory",
        help="Relative path to a folder containing images.",
    ),
    use_cache: bool = typer.Option(
        False,
        "--use_cache",
        help="Speeds up computations and saves tokens when running the same prompt multiple times by caching the LLM response.",
    ),
    skip_file_selection: bool = typer.Option(
        False,
        "--skip-file-selection",
        "-s",
        help="Skip interactive file selection in improve mode and use the generated TOML file directly.",
    ),
    no_execution: bool = typer.Option(
        False,
        "--no_execution",
        help="Run setup but to not call LLM or write any code. For testing purposes.",
    ),
    sysinfo: bool = typer.Option(
        False,
        "--sysinfo",
        help="Output system information for debugging",
    ),
    diff_timeout: int = typer.Option(
        3,
        "--diff_timeout",
        help="Diff regexp timeout. Default: 3. Increase if regexp search timeouts.",
    ),
):
    """
    The main entry point for the CLI tool that generates or improves a project.

    This function sets up the CLI tool, loads environment variables, initializes
    the AI, and processes the user's request to generate or improve a project
    based on the provided arguments.

    Parameters
    ----------
    project_path : str
        The file path to the project directory.
    model : str
        The model ID string for the AI.
    temperature : float
        The temperature setting for the AI's responses.
    improve_mode : bool
        Flag indicating whether to improve an existing project.
    lite_mode : bool
        Flag indicating whether to run in lite mode.
    clarify_mode : bool
        Flag indicating whether to discuss specifications with AI before implementation.
    self_heal_mode : bool
        Flag indicating whether to enable self-healing mode.
    multi_turn_mode : bool
        Flag indicating whether to enable multi-turn conversation mode.
    azure_endpoint : str
        The endpoint for Azure OpenAI services.
    use_custom_preprompts : bool
        Flag indicating whether to use custom preprompts.
    prompt_file : str
        Relative path to a text file containing a prompt.
    entrypoint_prompt_file: str
        Relative path to a text file containing a file that specifies requirements for you entrypoint.
    image_directory: str
        Relative path to a folder containing images.
    use_cache: bool
        Speeds up computations and saves tokens when running the same prompt multiple times by caching the LLM response.
    verbose : bool
        Flag indicating whether to enable verbose logging.
    skip_file_selection: bool
        Skip interactive file selection in improve mode and use the generated TOML file directly
    no_execution: bool
        Run setup but to not call LLM or write any code. For testing purposes.
    sysinfo: bool
        Flag indicating whether to output system information for debugging.

    Returns
    -------
    None
    """
    config = create_config_from_args(
        project_path=project_path,
        model=model,
        temperature=temperature,
        improve_mode=improve_mode,
        lite_mode=lite_mode,
        clarify_mode=clarify_mode,
        self_heal_mode=self_heal_mode,
        multi_turn_mode=multi_turn_mode,
        azure_endpoint=azure_endpoint,
        use_custom_preprompts=use_custom_preprompts,
        llm_via_clipboard=llm_via_clipboard,
        verbose=verbose,
        debug=debug,
        prompt_file=prompt_file,
        entrypoint_prompt_file=entrypoint_prompt_file,
        image_directory=image_directory,
        use_cache=use_cache,
        skip_file_selection=skip_file_selection,
        no_execution=no_execution,
        sysinfo=sysinfo,
        diff_timeout=diff_timeout,
    )
    execute_gpt_engineer(config)


if __name__ == "__main__":
    app()
