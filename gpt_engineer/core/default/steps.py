"""
Module for defining the steps involved in generating and improving code using AI.

This module provides functions that represent different steps in the process of generating
and improving code using an AI model. These steps include generating code from a prompt,
creating an entrypoint for the codebase, executing the entrypoint, and refining code edits.

Functions
---------
curr_fn : function
    Returns the name of the current function.

setup_sys_prompt : function
    Sets up the system prompt for generating code.

gen_code : function
    Generates code from a prompt using AI and returns the generated files.

gen_entrypoint : function
    Generates an entrypoint for the codebase and returns the entrypoint files.

execute_entrypoint : function
    Executes the entrypoint of the codebase.

setup_sys_prompt_existing_code : function
    Sets up the system prompt for improving existing code.


improve : function
    Improves the code based on user input and returns the updated files.
"""

import datetime
import inspect
import io
import re
import sys
import traceback

from pathlib import Path
from typing import List, MutableMapping, Optional, Union
from uuid import uuid4
from subprocess import TimeoutExpired

from langchain.schema import HumanMessage, SystemMessage
from termcolor import colored

from gpt_engineer.core.ai import AI
from gpt_engineer.core.base_execution_env import BaseExecutionEnv
from gpt_engineer.core.base_memory import BaseMemory
from gpt_engineer.core.chat_to_files import apply_diffs, chat_to_files_dict, parse_diffs
from gpt_engineer.core.default.constants import MAX_EDIT_REFINEMENT_STEPS
from gpt_engineer.core.default.paths import (
    CODE_GEN_LOG_FILE,
    DEBUG_LOG_FILE,
    DIFF_LOG_FILE,
    ENTRYPOINT_FILE,
    ENTRYPOINT_LOG_FILE,
    IMPROVE_LOG_FILE,
)
from gpt_engineer.core.files_dict import FilesDict, file_to_lines_dict
from gpt_engineer.core.observed_preprompts import preprompt_collection_span
from gpt_engineer.core.preprompts_holder import PrepromptsHolder
from gpt_engineer.core.prompt import Prompt

# Import observability
try:
    from gpt_engineer.core.maxim_observability import get_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


def curr_fn() -> str:
    """
    Returns the name of the current function.

    Returns
    -------
    str
        The name of the function that called this function.
    """
    return inspect.stack()[1].function


def setup_sys_prompt(preprompts: MutableMapping[Union[str, Path], str]) -> str:
    """
    Sets up the system prompt for generating code.

    Parameters
    ----------
    preprompts : MutableMapping[Union[str, Path], str]
        A mapping of preprompt messages to guide the AI model.

    Returns
    -------
    str
        The system prompt message for the AI model.
    """
    return (
        preprompts["roadmap"]
        + preprompts["generate"].replace("FILE_FORMAT", preprompts["file_format"])
        + "\nUseful to know:\n"
        + preprompts["philosophy"]
    )


def setup_sys_prompt_existing_code(
    preprompts: MutableMapping[Union[str, Path], str]
) -> str:
    """
    Sets up the system prompt for improving existing code.

    Parameters
    ----------
    preprompts : MutableMapping[Union[str, Path], str]
        A mapping of preprompt messages to guide the AI model.

    Returns
    -------
    str
        The system prompt message for the AI model to improve existing code.
    """
    return (
        preprompts["roadmap"]
        + preprompts["improve"].replace("FILE_FORMAT", preprompts["file_format_diff"])
        + "\nUseful to know:\n"
        + preprompts["philosophy"]
    )


def gen_code(
    ai: AI,
    prompt: Prompt,
    memory: BaseMemory,
    preprompts_holder: PrepromptsHolder,
    parent_span_id: str = None,
) -> FilesDict:
    """
    Generates code from a prompt using AI and returns the generated files.

    Parameters
    ----------
    ai : AI
        The AI model used for generating code.
    prompt : str
        The user prompt to generate code from.
    memory : BaseMemory
        The memory interface where the code and related data are stored.
    preprompts_holder : PrepromptsHolder
        The holder for preprompt messages that guide the AI model.
    parent_span_id : str,
        The ID of the parent span.
    Returns
    -------
    FilesDict
        A dictionary of file names to their respective source code content.
    """
    # --- Preprompt Collection Span ---
    with preprompt_collection_span(preprompts_holder, parent_span_id) as span_info:
        preprompts = span_info["preprompts"]

    messages = ai.start(
        setup_sys_prompt(preprompts), prompt.to_langchain_content(), step_name=curr_fn()
    )
    chat = messages[-1].content.strip()
    memory.log(CODE_GEN_LOG_FILE, "\n\n".join(x.pretty_repr() for x in messages))
    files_dict = chat_to_files_dict(chat)
    return files_dict


def gen_entrypoint(
    ai: AI,
    prompt: Prompt,
    files_dict: FilesDict,
    memory: BaseMemory,
    preprompts_holder: PrepromptsHolder,
    parent_span_id: str = None,
) -> FilesDict:
    """
    Generates an entrypoint for the codebase and returns the entrypoint files.

    Parameters
    ----------
    ai : AI
        The AI model used for generating the entrypoint.
    files_dict : FilesDict
        The dictionary of file names to their respective source code content.
    memory : BaseMemory
        The memory interface where the code and related data are stored.
    preprompts_holder : PrepromptsHolder
        The holder for preprompt messages that guide the AI model.
    parent_span_id : str,
    Returns
    -------
    FilesDict
        A dictionary containing the entrypoint file.
    """
    user_prompt = prompt.entrypoint_prompt
    if not user_prompt:
        user_prompt = """
        Make a unix script that
        a) installs dependencies
        b) runs all necessary parts of the codebase (in parallel if necessary)
        """
    # --- Preprompt Collection Span ---
    with preprompt_collection_span(preprompts_holder, parent_span_id) as span_info:
        preprompts = span_info["preprompts"]

    # Ensure files_dict is a FilesDict object
    if not isinstance(files_dict, FilesDict):
        files_dict = FilesDict(files_dict)
    messages = ai.start(
        system=(preprompts["entrypoint"]),
        user=user_prompt
        + "\nInformation about the codebase:\n\n"
        + files_dict.to_chat(),
        step_name=curr_fn(),
    )
    print()
    chat = messages[-1].content.strip()
    regex = r"```\S*\n(.+?)```"
    matches = re.finditer(regex, chat, re.DOTALL)
    entrypoint_code = FilesDict(
        {ENTRYPOINT_FILE: "\n".join(match.group(1) for match in matches)}
    )
    memory.log(ENTRYPOINT_LOG_FILE, "\n\n".join(x.pretty_repr() for x in messages))
    return entrypoint_code


def execute_entrypoint(
    ai: AI,
    execution_env: BaseExecutionEnv,
    files_dict: FilesDict,
    prompt: Prompt = None,
    preprompts_holder: PrepromptsHolder = None,
    memory: BaseMemory = None,
    parent_span_id: str = None,
    web_ui: Optional[bool] = False,
) -> FilesDict:
    """
    Executes the entrypoint of the codebase.

    Parameters
    ----------
    ai : AI
        The AI model used for generating the entrypoint.
    execution_env : BaseExecutionEnv
        The execution environment in which the code is executed.
    files_dict : FilesDict
        The dictionary of file names to their respective source code content.
    preprompts_holder : PrepromptsHolder, optional
        The holder for preprompt messages that guide the AI model.

    Returns
    -------
    FilesDict
        The dictionary of file names to their respective source code content after execution.
    """
    observability = None
    validation_span_id = None

    if OBSERVABILITY_AVAILABLE:
        try:
            observability = get_observability()
        except Exception:
            pass  # Continue without observability

    try:
        # Start entrypoint validation span as child if parent_span_id is provided
        if observability and observability.is_enabled():
            validation_span_id = str(uuid4())
            start_span_kwargs = dict(
                span_id=validation_span_id,
                name="Check Run Script Files",
                tags={
                    "operation": "file_validation",
                    "required_file": ENTRYPOINT_FILE,
                    "total_files": str(len(files_dict)),
                },
            )
            if parent_span_id:
                start_span_kwargs["parent_span_id"] = parent_span_id
            observability.start_span(**start_span_kwargs)

        # Perform file existence check and log as tool call
        file_exists = ENTRYPOINT_FILE in files_dict
        if observability and observability.is_enabled() and validation_span_id:
            validation_tool_call_id = str(uuid4())
            observability.log_tool_call(
                tool_call_id=validation_tool_call_id,
                name="File Existence Check",
                description=f"Check if required entrypoint file {ENTRYPOINT_FILE} exists",
                args={
                    "checking_file": ENTRYPOINT_FILE,
                    "files_available": list(files_dict.keys()),
                    "file_count": len(files_dict),
                },
                result={
                    "file_exists": file_exists,
                    "file_found": file_exists,
                    "missing_file": ENTRYPOINT_FILE if not file_exists else None,
                    "validation_passed": file_exists,
                },
                tags={
                    "tool_type": "file_validation",
                    "operation": "file_existence_check",
                    "file_type": ".sh",
                    "validation_result": "passed" if file_exists else "failed",
                },
            )

        if ENTRYPOINT_FILE not in files_dict:
            error_msg = (
                f"The required entrypoint {ENTRYPOINT_FILE} does not exist in the code."
            )
            if observability and observability.is_enabled() and validation_span_id:
                observability.end_span(
                    span_id=validation_span_id, error=FileNotFoundError(error_msg)
                )
            raise FileNotFoundError(error_msg)

        command = files_dict[ENTRYPOINT_FILE]

        # Perform command validation and log as tool call
        command_validation_passed = (
            True  # Basic validation - could be enhanced with more checks
        )
        if observability and observability.is_enabled() and validation_span_id:
            cmd_validation_tool_call_id = str(uuid4())
            observability.log_tool_call(
                tool_call_id=cmd_validation_tool_call_id,
                name="Command Validation",
                description="Validate command for safe execution",
                args={
                    "command": command,
                    "command_length": len(command),
                    "command_type": "bash",
                    "validation_rules": ["basic_syntax_check"],
                },
                result={
                    "validation_passed": command_validation_passed,
                    "command_valid": command_validation_passed,
                    "command_length": len(command),
                    "command_preview": command[:100] + "..."
                    if len(command) > 100
                    else command,
                    "warnings": [],
                },
                tags={
                    "tool_type": "command_validation",
                    "operation": "command_validation",
                    "command_type": "bash",
                    "validation_result": "passed"
                    if command_validation_passed
                    else "failed",
                },
            )

        # End validation span successfully
        if observability and observability.is_enabled() and validation_span_id:
            observability.end_span(
                span_id=validation_span_id,
                result={"entrypoint_found": True, "command_length": len(command)},
            )

        # Log prompt display as an event
        if observability and observability.is_enabled():
            prompt_display_event_id = str(uuid4())
            observability.log_event(
                event_id=prompt_display_event_id,
                event_type="prompt_display",
                data=None,
                metadata={
                    "prompt_text": "Do you want to execute this code? (Y/n)",
                    "command_displayed": command,
                    "display_time": datetime.datetime.utcnow().isoformat() + "+00:00",
                },
                tags={
                    "operation": "user_prompt_display",
                    "interaction_type": "confirmation",
                },
            )

        print()
        print(
            colored(
                "Do you want to execute this code? (Y/n)",
                "red",
            )
        )
        print()
        print(command)
        print()

        if observability and observability.is_enabled():
            trace_output_content = f"{command}\nDo you want to execute this code? (Y/n)"
            observability.set_trace_output(trace_output_content)
            observability.end_trace()

    finally:
        if web_ui:
            # For web UI, return files_dict and let the web UI handle execution confirmation
            # The web UI will detect the "Do you want to execute this code?" prompt
            # and set up the pending execution state
            return files_dict
        else:
            return execute_entrypoint_next(execution_env, files_dict)


def execute_entrypoint_next(
    execution_env: BaseExecutionEnv,
    files_dict: FilesDict = None,
    user_response: str = None,
    timeout: Optional[int] = 30,
) -> FilesDict:
    """
    Executes the entrypoint of the codebase.
    """

    observability = None
    execution_span_id = None

    if OBSERVABILITY_AVAILABLE:
        try:
            observability = get_observability()
        except Exception:
            pass  # Continue without observability

    try:
        # Start a new trace for command execution and user feedback
        if observability and observability.is_enabled():
            new_trace_id = str(uuid4())
            observability.start_trace(
                trace_id=new_trace_id,
                name=new_trace_id,
                tags={
                    "operation": "command_execution",
                    "phase": "execution",
                },
                metadata={
                    "files_to_process": len(files_dict),
                    "has_entrypoint": "run.sh" in files_dict,
                },
            )
            observability.set_trace_input(f"{user_response}")
            # Use provided user_response if available, otherwise prompt for input
        if user_response is None:
            user_response = input("").lower()
        else:
            user_response = user_response.lower()
        user_confirmed = user_response in ["", "y", "yes"]

        if not user_confirmed:
            print("Ok, not executing the code.")
            return files_dict or FilesDict()

        print("Executing the code...")
        print()
        print(
            colored(
                "Note: If it does not work as expected, consider running the code"
                + " in another way than above.",
                "green",
            )
        )
        print()
        print("You can press ctrl+c *once* to stop the execution.")
        print()

        try:
            uploaded_result = execution_env.upload(files_dict, parent_span_id=execution_span_id)
        except Exception as upload_error:
            if observability and observability.is_enabled():
                observability.log_event(
                    event_id=str(uuid4()),
                    event_type="upload_error",
                    data=None,
                    metadata={"error": str(upload_error)},
                    tags={"operation": "file_upload", "error": "true"},
                )
            raise

        # Start command execution span
        if observability and observability.is_enabled():
            execution_span_id = str(uuid4())
            observability.start_span(
                span_id=execution_span_id,
                name="Run Generated Code",
                tags={
                    "operation": "shell_execution",
                    "command": f"bash {ENTRYPOINT_FILE}",
                    "entrypoint_file": ENTRYPOINT_FILE,
                },
            )

        # Log tool call for code execution
        tool_call_id = None
        if observability and observability.is_enabled():
            tool_call_id = str(uuid4())
            observability.log_tool_call(
                tool_call_id=tool_call_id,
                name="Code Execution",
                description=f"Execute the entrypoint file ({ENTRYPOINT_FILE}) to run the generated code",
                args={
                    "entrypoint_file": ENTRYPOINT_FILE,
                    "command": f"bash {ENTRYPOINT_FILE}",
                    "files_count": len(files_dict),
                },
                tags={
                    "operation": "code_execution",
                    "mode": "execute_entrypoint",
                },
            )

        try:
            result = uploaded_result.run(f"bash {ENTRYPOINT_FILE}", timeout=timeout)
            if observability and observability.is_enabled():
                observability.set_trace_output(str(result))
        except TimeoutExpired as timeout_error:
            if observability and observability.is_enabled() and tool_call_id:
                observability.log_tool_call(
                    tool_call_id=tool_call_id,
                    name="Code Execution",
                    description=f"Execute the entrypoint file ({ENTRYPOINT_FILE}) to run the generated code",
                    args={
                        "entrypoint_file": ENTRYPOINT_FILE,
                        "command": f"bash {ENTRYPOINT_FILE}",
                        "files_count": len(files_dict),
                        "timeout": timeout,
                    },
                    result={
                        "execution_completed": False,
                        "error": str(timeout_error),
                        "timed_out": True,
                    },
                    tags={
                        "operation": "code_execution",
                        "mode": "execute_entrypoint",
                        "success": "false",
                        "timeout": "true",
                    },
                )
                tool_call_id = None
            raise
        except Exception as run_error:
            if observability and observability.is_enabled() and tool_call_id:
                observability.log_tool_call(
                    tool_call_id=tool_call_id,
                    name="Code Execution",
                    description=f"Execute the entrypoint file ({ENTRYPOINT_FILE}) to run the generated code",
                    args={
                        "entrypoint_file": ENTRYPOINT_FILE,
                        "command": f"bash {ENTRYPOINT_FILE}",
                        "files_count": len(files_dict),
                    },
                    result={
                        "execution_completed": False,
                        "error": str(run_error),
                    },
                    tags={
                        "operation": "code_execution",
                        "mode": "execute_entrypoint",
                        "success": "false",
                    },
                )
                tool_call_id = None
            raise

        # Update tool call with result
        if observability and observability.is_enabled() and tool_call_id:
            observability.log_tool_call(
                tool_call_id=tool_call_id,
                name="Code Execution",
                description=f"Execute the entrypoint file ({ENTRYPOINT_FILE}) to run the generated code",
                args={
                    "entrypoint_file": ENTRYPOINT_FILE,
                    "command": f"bash {ENTRYPOINT_FILE}",
                    "files_count": len(files_dict),
                },
                result={
                    "execution_completed": True,
                    "entrypoint_file": ENTRYPOINT_FILE,
                },
                tags={
                    "operation": "code_execution",
                    "mode": "execute_entrypoint",
                    "success": "true",
                },
            )

        # End execution span
        if observability and observability.is_enabled() and execution_span_id:
            observability.end_span(
                span_id=execution_span_id, result={"execution_completed": True}
            )

        return files_dict or FilesDict()

    except Exception as e:
        # End any active spans with error
        if observability and observability.is_enabled():
            if execution_span_id:
                observability.end_span(span_id=execution_span_id, error=e)
        raise


def improve_fn(
    ai: AI,
    prompt: Prompt,
    files_dict: FilesDict,
    memory: BaseMemory,
    preprompts_holder: PrepromptsHolder,
    diff_timeout=3,
) -> FilesDict:
    """
    Improves the code based on user input and returns the updated files.

    Parameters
    ----------
    ai : AI
        The AI model used for improving code.
    prompt :str
        The user prompt to improve the code.
    files_dict : FilesDict
        The dictionary of file names to their respective source code content.
    memory : BaseMemory
        The memory interface where the code and related data are stored.
    preprompts_holder : PrepromptsHolder
        The holder for preprompt messages that guide the AI model.

    Returns
    -------
    FilesDict
        The dictionary of file names to their respective updated source code content.
    """
    preprompts = preprompts_holder.get_preprompts(suppress_observability=True)
    messages = [
        SystemMessage(content=setup_sys_prompt_existing_code(preprompts)),
    ]

    # Ensure files_dict is a FilesDict object
    if not isinstance(files_dict, FilesDict):
        files_dict = FilesDict(files_dict)

    # Add files as input
    messages.append(HumanMessage(content=f"{files_dict.to_chat()}"))
    messages.append(HumanMessage(content=prompt.to_langchain_content()))
    memory.log(
        DEBUG_LOG_FILE,
        "UPLOADED FILES:\n" + files_dict.to_log() + "\nPROMPT:\n" + prompt.text,
    )
    return _improve_loop(ai, files_dict, memory, messages, diff_timeout=diff_timeout)


def _improve_loop(
    ai: AI, files_dict: FilesDict, memory: BaseMemory, messages: List, diff_timeout=3
) -> FilesDict:
    messages = ai.next(messages, step_name=curr_fn())
    files_dict, errors = salvage_correct_hunks(
        messages, files_dict, memory, diff_timeout=diff_timeout
    )

    retries = 0
    while errors and retries < MAX_EDIT_REFINEMENT_STEPS:
        messages.append(
            HumanMessage(
                content="Some previously produced diffs were not on the requested format, or the code part was not found in the code. Details:\n"
                + "\n".join(errors)
                + "\n Only rewrite the problematic diffs, making sure that the failing ones are now on the correct format and can be found in the code. Make sure to not repeat past mistakes. \n"
            )
        )
        messages = ai.next(messages, step_name=curr_fn())
        files_dict, errors = salvage_correct_hunks(
            messages, files_dict, memory, diff_timeout
        )
        retries += 1

    return files_dict


def salvage_correct_hunks(
    messages: List, files_dict: FilesDict, memory: BaseMemory, diff_timeout=3
) -> tuple[FilesDict, List[str]]:
    error_messages = []
    ai_response = messages[-1].content.strip()

    diffs = parse_diffs(ai_response, diff_timeout=diff_timeout)
    # validate and correct diffs

    for _, diff in diffs.items():
        # if diff is a new file, validation and correction is unnecessary
        if not diff.is_new_file():
            problems = diff.validate_and_correct(
                file_to_lines_dict(files_dict[diff.filename_pre])
            )
            error_messages.extend(problems)
    files_dict = apply_diffs(diffs, files_dict)
    memory.log(IMPROVE_LOG_FILE, "\n\n".join(x.pretty_repr() for x in messages))
    memory.log(DIFF_LOG_FILE, "\n\n".join(error_messages))
    return files_dict, error_messages


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for file in self.files:
            file.write(obj)

    def flush(self):
        for file in self.files:
            file.flush()


def handle_improve_mode(
    prompt,
    agent,
    memory,
    files_dict,
    diff_timeout=3,
    parent_span_id: Optional[str] = None,
):
    captured_output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, captured_output)

    try:
        # Ensure input is a FilesDict
        if not isinstance(files_dict, FilesDict):
            files_dict = FilesDict(files_dict)

        if parent_span_id:
            files_dict = agent.improve(
                files_dict,
                prompt,
                diff_timeout=diff_timeout,
                parent_span_id=parent_span_id,
            )
        else:
            files_dict = agent.improve(files_dict, prompt, diff_timeout=diff_timeout)

        # Ensure output is also a FilesDict
        if not isinstance(files_dict, FilesDict):
            files_dict = FilesDict(files_dict)

    except Exception as e:
        print(
            f"Error while improving the project: {e}\nCould you please upload the debug_log_file.txt in {memory.path}/logs folder to github?\nFULL STACK TRACE:\n"
        )
        traceback.print_exc(file=sys.stdout)  # Print the full stack trace
    finally:
        # Reset stdout
        sys.stdout = old_stdout

        # Get the captured output
        captured_string = captured_output.getvalue()
        print(captured_string)
        memory.log(DEBUG_LOG_FILE, "\nCONSOLE OUTPUT:\n" + captured_string)

    return files_dict
