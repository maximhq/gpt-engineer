import subprocess

from platform import platform
from sys import version_info
from typing import List, Optional, Union

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from gpt_engineer.core.ai import AI
from gpt_engineer.core.base_execution_env import BaseExecutionEnv
from gpt_engineer.core.base_memory import BaseMemory
from gpt_engineer.core.chat_to_files import chat_to_files_dict
from gpt_engineer.core.default.paths import CODE_GEN_LOG_FILE, ENTRYPOINT_FILE
from gpt_engineer.core.default.steps import curr_fn, improve_fn, setup_sys_prompt
from gpt_engineer.core.files_dict import FilesDict
from gpt_engineer.core.preprompts_holder import PrepromptsHolder
from gpt_engineer.core.prompt import Prompt

# Import observability
try:
    from gpt_engineer.core.maxim_observability import get_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

# Type hint for chat messages
Message = Union[AIMessage, HumanMessage, SystemMessage]
MAX_SELF_HEAL_ATTEMPTS = 10


def get_platform_info() -> str:
    """
    Returns a string containing the OS and Python version information.

    This function is used for self-healing by providing information about the current
    operating system and Python version. It assumes that the Python version in the
    virtual environment is the one being used.

    Returns:
        str: A string containing the OS and Python version information.
    """

    v = version_info
    a = f"Python Version: {v.major}.{v.minor}.{v.micro}"
    b = f"\nOS: {platform()}\n"
    return a + b


def self_heal(
    ai: AI,
    execution_env: BaseExecutionEnv,
    files_dict: FilesDict,
    prompt: Prompt = None,
    preprompts_holder: PrepromptsHolder = None,
    memory: BaseMemory = None,
    diff_timeout=3,
    parent_span_id: str = None,
    execution_timeout: Optional[int] = 30,
) -> FilesDict:
    """
    Attempts to execute the code from the entrypoint and if it fails, sends the error output back to the AI with instructions to fix.

    Parameters
    ----------
    ai : AI
        An instance of the AI model.
    execution_env : BaseExecutionEnv
        The execution environment where the code is run.
    files_dict : FilesDict
        A dictionary of file names to their contents.
    prompt : Prompt, optional
        The original prompt that was used to generate the code.
    preprompts_holder : PrepromptsHolder, optional
        A holder for preprompt messages.
    memory : BaseMemory, optional
        The memory instance for logging.
    diff_timeout : int, optional
        Timeout for diff operations.
    parent_span_id : str, optional
        Parent span ID for observability tracing.

    Returns
    -------
    FilesDict
        The updated files dictionary after self-healing attempts.

    Raises
    ------
    FileNotFoundError
        If the required entrypoint file does not exist in the code.
    AssertionError
        If the preprompts_holder is None.

    Notes
    -----
    This code will make `MAX_SELF_HEAL_ATTEMPTS` to try and fix the code
    before giving up.
    This makes the assuption that the previous step was `gen_entrypoint`,
    this code could work with `simple_gen`, or `gen_clarified_code` as well.
    """

    # Initialize observability
    observability = None
    validation_span_id = None
    execution_span_id = None
    improvement_span_id = None

    if OBSERVABILITY_AVAILABLE:
        try:
            observability = get_observability()
        except Exception:
            pass  # Continue without observability

    try:
        # Start validation span
        if observability and observability.is_enabled():
            from uuid import uuid4

            validation_span_id = str(uuid4())
            start_span_kwargs = dict(
                span_id=validation_span_id,
                name="Self-Heal Validation",
                tags={
                    "operation": "self_heal_validation",
                    "required_file": ENTRYPOINT_FILE,
                    "total_files": str(len(files_dict)),
                    "max_attempts": str(MAX_SELF_HEAL_ATTEMPTS),
                },
                metadata={
                    "file_count": len(files_dict),
                    "files_available": list(files_dict.keys()),
                    "diff_timeout": diff_timeout,
                },
            )
            if parent_span_id:
                start_span_kwargs["parent_span_id"] = parent_span_id
            observability.start_span(**start_span_kwargs)

        # step 1. execute the entrypoint
        # log_path = dbs.workspace.path / "log.txt"
        if ENTRYPOINT_FILE not in files_dict:
            error_msg = (
                "The required entrypoint "
                + ENTRYPOINT_FILE
                + " does not exist in the code."
            )
            if observability and observability.is_enabled() and validation_span_id:
                observability.end_span(
                    span_id=validation_span_id, error=FileNotFoundError(error_msg)
                )
            raise FileNotFoundError(error_msg)

        # End validation span successfully
        if observability and observability.is_enabled() and validation_span_id:
            observability.end_span(
                span_id=validation_span_id,
                result={"entrypoint_found": True, "validation_passed": True},
            )

        attempts = 0
        if preprompts_holder is None:
            raise AssertionError("Prepromptsholder required for self-heal")

        # Start main self-heal execution span
        if observability and observability.is_enabled():
            execution_span_id = str(uuid4())
            observability.start_span(
                span_id=execution_span_id,
                name="Self-Heal Execution Loop",
                tags={
                    "operation": "self_heal_execution",
                    "max_attempts": str(MAX_SELF_HEAL_ATTEMPTS),
                    "diff_timeout": str(diff_timeout),
                },
                metadata={
                    "initial_file_count": len(files_dict),
                    "entrypoint_content_length": len(files_dict[ENTRYPOINT_FILE]),
                },
            )

        print(
            f"Starting self-healing process (max {MAX_SELF_HEAL_ATTEMPTS} attempts)..."
        )

        while attempts < MAX_SELF_HEAL_ATTEMPTS:
            attempts += 1
            timed_out = False

            print(f"Attempt {attempts}/{MAX_SELF_HEAL_ATTEMPTS}: Executing code...")

            # Start the process
            execution_env.upload(files_dict, parent_span_id=execution_span_id)
            p = execution_env.popen(files_dict[ENTRYPOINT_FILE])

            # Log tool call for code execution
            tool_call_id = None
            if observability and observability.is_enabled():
                tool_call_id = str(uuid4())
                observability.log_tool_call(
                    tool_call_id=tool_call_id,
                    name="Code Execution",
                    description=f"Execute the entrypoint file ({ENTRYPOINT_FILE}) to test if the code runs successfully",
                    args={
                        "entrypoint_file": ENTRYPOINT_FILE,
                        "attempt_number": attempts,
                        "max_attempts": MAX_SELF_HEAL_ATTEMPTS,
                        "files_count": len(files_dict),
                        "entrypoint_content_length": len(files_dict[ENTRYPOINT_FILE]),
                        "execution_timeout": execution_timeout,
                    },
                    tags={
                        "operation": "code_execution",
                        "attempt": str(attempts),
                        "mode": "self_heal",
                    },
                )

            # Wait for the process to complete and get output
            try:
                stdout_full, stderr_full = p.communicate(timeout=execution_timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                stdout_full, stderr_full = p.communicate()
                timed_out = True
                print(f"Execution timed out after {execution_timeout} seconds")

            # Update tool call with result
            if observability and observability.is_enabled() and tool_call_id:
                success = p.returncode == 0 or p.returncode == 2
                observability.log_tool_call(
                    tool_call_id=tool_call_id,
                    name="Code Execution",
                    description=f"Execute the entrypoint file ({ENTRYPOINT_FILE}) to test if the code runs successfully",
                    args={
                        "entrypoint_file": ENTRYPOINT_FILE,
                        "attempt_number": attempts,
                        "max_attempts": MAX_SELF_HEAL_ATTEMPTS,
                        "files_count": len(files_dict),
                        "entrypoint_content_length": len(files_dict[ENTRYPOINT_FILE]),
                        "execution_timeout": execution_timeout,
                    },
                    result={
                        "success": success,
                        "return_code": p.returncode,
                        "stdout_length": len(stdout_full),
                        "stderr_length": len(stderr_full),
                        "stdout_preview": stdout_full.decode("utf-8")[:500]
                        if stdout_full
                        else "",
                        "stderr_preview": stderr_full.decode("utf-8")[:500]
                        if stderr_full
                        else "",
                        "timed_out": timed_out,
                    },
                    tags={
                        "operation": "code_execution",
                        "attempt": str(attempts),
                        "mode": "self_heal",
                        "success": str(success),
                    },
                )

            # Log execution result event
            if observability and observability.is_enabled() and execution_span_id:
                execution_result_event_id = str(uuid4())
                observability.log_event(
                    event_id=execution_result_event_id,
                    event_type="self_heal_execution_result",
                    metadata={
                        "attempt_number": attempts,
                        "return_code": p.returncode,
                        "stdout_length": len(stdout_full),
                        "stderr_length": len(stderr_full),
                        "timed_out": timed_out,
                        "success": p.returncode == 0 or p.returncode == 2,
                    },
                    tags={
                        "operation": "self_heal_execution_result",
                        "attempt": str(attempts),
                        "success": str(p.returncode == 0 or p.returncode == 2),
                    },
                )

            if (p.returncode != 0 and p.returncode != 2) and not timed_out:
                print(
                    f"Attempt {attempts} failed (return code: {p.returncode}). Attempting to fix..."
                )
                print("Error output:")
                print(stdout_full.decode("utf-8"))
                print(stderr_full.decode("utf-8"))

                # Log failure event
                if observability and observability.is_enabled() and execution_span_id:
                    failure_event_id = str(uuid4())
                    observability.log_event(
                        event_id=failure_event_id,
                        event_type="self_heal_execution_failed",
                        metadata={
                            "attempt_number": attempts,
                            "return_code": p.returncode,
                            "stdout": stdout_full.decode("utf-8"),
                            "stderr": stderr_full.decode("utf-8"),
                            "remaining_attempts": MAX_SELF_HEAL_ATTEMPTS - attempts,
                        },
                        tags={
                            "operation": "self_heal_failure",
                            "attempt": str(attempts),
                            "return_code": str(p.returncode),
                        },
                    )

                # Start improvement span
                if observability and observability.is_enabled():
                    improvement_span_id = str(uuid4())
                    observability.start_span(
                        parent_span_id=execution_span_id,
                        span_id=improvement_span_id,
                        name="Self-Heal Code Improvement",
                        tags={
                            "operation": "self_heal_improvement",
                            "attempt": str(attempts),
                            "model": ai.model_name
                            if hasattr(ai, "model_name")
                            else "unknown",
                            "diff_timeout": str(diff_timeout),
                        },
                        metadata={
                            "attempt_number": attempts,
                            "return_code": p.returncode,
                            "stdout_length": len(stdout_full),
                            "stderr_length": len(stderr_full),
                            "files_before_count": len(files_dict),
                        },
                    )

                # Log improvement start event
                if observability and observability.is_enabled() and improvement_span_id:
                    improvement_start_event_id = str(uuid4())
                    observability.log_event(
                        event_id=improvement_start_event_id,
                        event_type="self_heal_improvement_started",
                        metadata={
                            "attempt_number": attempts,
                            "error_output": stderr_full.decode("utf-8"),
                            "stdout_output": stdout_full.decode("utf-8"),
                        },
                        tags={
                            "operation": "self_heal_improvement",
                            "attempt": str(attempts),
                            "phase": "started",
                        },
                    )

                new_prompt = Prompt(
                    f"A program with this specification was requested:\n{prompt}\n, but running it produced the following output:\n{stdout_full}\n and the following errors:\n{stderr_full}. Please change it so that it fulfills the requirements."
                )

                files_dict_before = files_dict.copy()

                # Set the AI step name for proper LLM call tracking in self-heal mode
                if hasattr(ai, "set_current_step"):
                    ai.set_current_step(f"self_heal_improvement_attempt_{attempts}")

                print(f"Attempt {attempts}: Requesting AI to fix the code...")
                files_dict = improve_fn(
                    ai, new_prompt, files_dict, memory, preprompts_holder, diff_timeout
                )

                # Log improvement completion event
                if observability and observability.is_enabled() and improvement_span_id:
                    improvement_complete_event_id = str(uuid4())
                    observability.log_event(
                        event_id=improvement_complete_event_id,
                        event_type="self_heal_improvement_completed",
                        metadata={
                            "attempt_number": attempts,
                            "files_before_count": len(files_dict_before),
                            "files_after_count": len(files_dict),
                            "changes_made": files_dict != files_dict_before,
                        },
                        tags={
                            "operation": "self_heal_improvement",
                            "attempt": str(attempts),
                            "phase": "completed",
                            "changes_made": str(files_dict != files_dict_before),
                        },
                    )

                # End improvement span
                if observability and observability.is_enabled() and improvement_span_id:
                    observability.end_span(
                        span_id=improvement_span_id,
                        result={
                            "improvement_completed": True,
                            "files_after_count": len(files_dict),
                            "changes_made": files_dict != files_dict_before,
                        },
                    )

            else:
                print(f"Success! Code executed successfully on attempt {attempts}.")

                # Log success event
                if observability and observability.is_enabled() and execution_span_id:
                    success_event_id = str(uuid4())
                    observability.log_event(
                        event_id=success_event_id,
                        event_type="self_heal_execution_succeeded",
                        metadata={
                            "attempt_number": attempts,
                            "return_code": p.returncode,
                            "stdout_length": len(stdout_full),
                            "stderr_length": len(stderr_full),
                            "total_attempts_needed": attempts,
                        },
                        tags={
                            "operation": "self_heal_success",
                            "attempt": str(attempts),
                            "return_code": str(p.returncode),
                        },
                    )

                break

        # End execution span
        if observability and observability.is_enabled() and execution_span_id:
            observability.end_span(
                span_id=execution_span_id,
                result={
                    "total_attempts": attempts,
                    "final_file_count": len(files_dict),
                    "execution_successful": attempts < MAX_SELF_HEAL_ATTEMPTS,
                    "max_attempts_reached": attempts >= MAX_SELF_HEAL_ATTEMPTS,
                },
            )

        # Print final summary
        if attempts >= MAX_SELF_HEAL_ATTEMPTS:
            print(f"Self-healing failed after {MAX_SELF_HEAL_ATTEMPTS} attempts.")
        else:
            print(f"Self-healing completed successfully in {attempts} attempt(s).")

        # Log self-heal summary event
        if observability and observability.is_enabled():
            from uuid import uuid4

            summary_event_id = str(uuid4())
            observability.log_event(
                event_id=summary_event_id,
                event_type="self_heal_process_summary",
                metadata={
                    "total_attempts": attempts,
                    "execution_successful": attempts < MAX_SELF_HEAL_ATTEMPTS,
                    "max_attempts_reached": attempts >= MAX_SELF_HEAL_ATTEMPTS,
                    "final_file_count": len(files_dict),
                    "llm_calls_made": attempts
                    if attempts < MAX_SELF_HEAL_ATTEMPTS
                    else MAX_SELF_HEAL_ATTEMPTS,
                    "improvements_applied": attempts - 1
                    if attempts > 1 and attempts < MAX_SELF_HEAL_ATTEMPTS
                    else 0,
                },
                tags={
                    "operation": "self_heal_summary",
                    "execution_successful": str(attempts < MAX_SELF_HEAL_ATTEMPTS),
                    "total_attempts": str(attempts),
                },
            )

        return files_dict

    except Exception as e:
        # End any active spans with error
        if observability and observability.is_enabled():
            if validation_span_id:
                observability.end_span(span_id=validation_span_id, error=e)
            if execution_span_id:
                observability.end_span(span_id=execution_span_id, error=e)
            if improvement_span_id:
                observability.end_span(span_id=improvement_span_id, error=e)
        raise


def clarified_gen(
    ai: AI, prompt: Prompt, memory: BaseMemory, preprompts_holder: PrepromptsHolder
) -> FilesDict:
    """
    Generates code based on clarifications obtained from the user and saves it to a specified workspace.

    Parameters
    ----------
    ai : AI
        An instance of the AI model, responsible for processing and generating the code.
    prompt : str
        The user's clarification prompt.
    memory : BaseMemory
        The memory instance where the generated code log is saved.
    preprompts_holder : PrepromptsHolder
        A holder for preprompt messages.

    Returns
    -------
    FilesDict
        A dictionary of file names to their contents generated by the AI.
    """

    preprompts = preprompts_holder.get_preprompts()
    messages: List[Message] = [SystemMessage(content=preprompts["clarify"])]
    user_input = prompt.text  # clarify does not work with vision right now
    while True:
        messages = ai.next(messages, user_input, step_name=curr_fn())
        msg = messages[-1].content.strip()

        if "nothing to clarify" in msg.lower():
            break

        if msg.lower().startswith("no"):
            print("Nothing to clarify.")
            break

        print('(answer in text, or "c" to move on)\n')
        user_input = input("")
        print()

        if not user_input or user_input == "c":
            print("(letting gpt-engineer make its own assumptions)")
            print()
            messages = ai.next(
                messages,
                "Make your own assumptions and state them explicitly before starting",
                step_name=curr_fn(),
            )
            print()

        user_input += """
            \n\n
            Is anything else unclear? If yes, ask another question.\n
            Otherwise state: "Nothing to clarify"
            """

    print()

    messages = [
        SystemMessage(content=setup_sys_prompt(preprompts)),
    ] + messages[
        1:
    ]  # skip the first clarify message, which was the original clarify priming prompt
    messages = ai.next(
        messages,
        preprompts["generate"].replace("FILE_FORMAT", preprompts["file_format"]),
        step_name=curr_fn(),
    )
    print()
    chat = messages[-1].content.strip()
    memory.log(CODE_GEN_LOG_FILE, "\n\n".join(x.pretty_repr() for x in messages))
    files_dict = chat_to_files_dict(chat)
    return files_dict


def lite_gen(
    ai: AI, prompt: Prompt, memory: BaseMemory, preprompts_holder: PrepromptsHolder
) -> FilesDict:
    """
    Executes the AI model using the main prompt and saves the generated results to the specified workspace.

    Parameters
    ----------
    ai : AI
        An instance of the AI model.
    prompt : str
        The main prompt to feed to the AI model.
    memory : BaseMemory
        The memory instance where the generated code log is saved.
    preprompts_holder : PrepromptsHolder
        A holder for preprompt messages.

    Returns
    -------
    FilesDict
        A dictionary of file names to their contents generated by the AI.

    Notes
    -----
    The function assumes the `ai.start` method and the `to_files` utility to be correctly
    set up and functional. Ensure these prerequisites before invoking `lite_gen`.
    """

    preprompts = preprompts_holder.get_preprompts()
    messages = ai.start(
        prompt.to_langchain_content(), preprompts["file_format"], step_name=curr_fn()
    )
    chat = messages[-1].content.strip()
    memory.log(CODE_GEN_LOG_FILE, "\n\n".join(x.pretty_repr() for x in messages))
    files_dict = chat_to_files_dict(chat)
    return files_dict
