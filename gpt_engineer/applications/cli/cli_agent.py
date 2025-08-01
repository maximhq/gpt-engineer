"""
This module provides the CliAgent class which manages the lifecycle of code generation and improvement
using an AI model. It includes functionalities to initialize code generation, improve existing code,
and process the code through various steps defined in the step bundle.
"""

from typing import Callable, Optional, TypeVar

# from gpt_engineer.core.default.git_version_manager import GitVersionManager
from gpt_engineer.core.ai import AI
from gpt_engineer.core.base_agent import BaseAgent
from gpt_engineer.core.base_execution_env import BaseExecutionEnv
from gpt_engineer.core.base_memory import BaseMemory
from gpt_engineer.core.default.disk_execution_env import DiskExecutionEnv
from gpt_engineer.core.default.disk_memory import DiskMemory
from gpt_engineer.core.default.paths import PREPROMPTS_PATH
from gpt_engineer.core.default.steps import (
    execute_entrypoint,
    gen_code,
    gen_entrypoint,
    improve_fn,
)
from gpt_engineer.core.files_dict import FilesDict
from gpt_engineer.core.preprompts_holder import PrepromptsHolder
from gpt_engineer.core.prompt import Prompt

# Import observability
try:
    from gpt_engineer.core.maxim_observability import get_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

CodeGenType = TypeVar("CodeGenType", bound=Callable[[AI, str, BaseMemory], FilesDict])
CodeProcessor = TypeVar(
    "CodeProcessor", bound=Callable[[AI, BaseExecutionEnv, FilesDict], FilesDict]
)
ImproveType = TypeVar(
    "ImproveType", bound=Callable[[AI, str, FilesDict, BaseMemory], FilesDict]
)


class CliAgent(BaseAgent):
    """
    The `CliAgent` class is responsible for managing the lifecycle of code generation and improvement
    using an AI model. It orchestrates the generation of new code and the improvement of existing code
    based on given prompts and utilizes a memory system and execution environment for processing.

    Parameters
    ----------
    memory : BaseMemory
        An instance of a class that adheres to the BaseMemory interface, used for storing and retrieving
        information during the code generation process.
    execution_env : BaseExecutionEnv
        An instance of a class that adheres to the BaseExecutionEnv interface, used for executing code
        and managing the execution environment.
    ai : AI, optional
        An instance of the AI class that manages calls to the language model. If not provided, a default
        instance is created.
    code_gen_fn : CodeGenType, optional
        A callable that takes an AI instance, a prompt, and a memory instance to generate code. Defaults
        to the `gen_code` function.
    improve_fn : ImproveType, optional
        A callable that takes an AI instance, a prompt, a FilesDict instance, and a memory instance to
        improve code. Defaults to the `improve` function.
    process_code_fn : CodeProcessor, optional
        A callable that takes an AI instance, an execution environment, and a FilesDict instance to
        process code. Defaults to the `execute_entrypoint` function.
    preprompts_holder : PrepromptsHolder, optional
        An instance of PrepromptsHolder that manages preprompt templates. If not provided, a default
        instance is created using the PREPROMPTS_PATH.

    Attributes
    ----------
    memory : BaseMemory
        The memory instance where the agent stores and retrieves information.
    execution_env : BaseExecutionEnv
        The execution environment instance where the agent executes and manages code.
    ai : AI
        The AI instance used for interacting with the language model.
    code_gen_fn : CodeGenType
        The function used for generating code.
    improve_fn : ImproveType
        The function used for improving code.
    process_code_fn : CodeProcessor
        The function used for processing code.
    preprompts_holder : PrepromptsHolder
        The holder for preprompt templates.
    """

    def __init__(
        self,
        memory: BaseMemory,
        execution_env: BaseExecutionEnv,
        ai: AI = None,
        code_gen_fn: CodeGenType = gen_code,
        improve_fn: ImproveType = improve_fn,
        process_code_fn: CodeProcessor = execute_entrypoint,
        preprompts_holder: PrepromptsHolder = None,
    ):
        self.memory = memory
        self.execution_env = execution_env
        self.ai = ai or AI()
        self.code_gen_fn = code_gen_fn
        self.process_code_fn = process_code_fn
        self.improve_fn = improve_fn
        self.preprompts_holder = preprompts_holder or PrepromptsHolder(PREPROMPTS_PATH)

    @classmethod
    def with_default_config(
        cls,
        memory: DiskMemory,
        execution_env: DiskExecutionEnv,
        ai: AI = None,
        code_gen_fn: CodeGenType = gen_code,
        improve_fn: ImproveType = improve_fn,
        process_code_fn: CodeProcessor = execute_entrypoint,
        preprompts_holder: PrepromptsHolder = None,
        diff_timeout=3,
    ):
        """
        Creates a new instance of CliAgent with default configurations for memory, execution environment,
        AI, and other functional parameters.

        Parameters
        ----------
        memory : DiskMemory
            An instance of DiskMemory for storing and retrieving information.
        execution_env : DiskExecutionEnv
            An instance of DiskExecutionEnv for executing code.
        ai : AI, optional
            An instance of AI for interacting with the language model. Defaults to None, which will create
            a new AI instance.
        code_gen_fn : CodeGenType, optional
            A function for generating code. Defaults to `gen_code`.
        improve_fn : ImproveType, optional
            A function for improving code. Defaults to `improve`.
        process_code_fn : CodeProcessor, optional
            A function for processing code. Defaults to `execute_entrypoint`.
        preprompts_holder : PrepromptsHolder, optional
            An instance of PrepromptsHolder for managing preprompt templates. Defaults to None, which will
            create a new PrepromptsHolder instance using PREPROMPTS_PATH.

        Returns
        -------
        CliAgent
            An instance of CliAgent configured with the provided or default parameters.
        """
        return cls(
            memory=memory,
            execution_env=execution_env,
            ai=ai,
            code_gen_fn=code_gen_fn,
            process_code_fn=process_code_fn,
            improve_fn=improve_fn,
            preprompts_holder=preprompts_holder or PrepromptsHolder(PREPROMPTS_PATH),
        )

    def init(self, prompt: Prompt) -> FilesDict:
        """
        Generates a new piece of code using the AI and step bundle based on the provided prompt.

        Parameters
        ----------
        prompt : str
            A string prompt that guides the code generation process.

        Returns
        -------
        FilesDict
            An instance of the `FilesDict` class containing the generated code.
        """
        observability = None
        code_gen_span_id = None
        entrypoint_span_id = None
        execution_span_id = None

        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
            except Exception:
                pass  # Continue without observability

        try:
            # Start code generation span
            if observability and observability.is_enabled():
                from uuid import uuid4

                code_gen_span_id = str(uuid4())
                observability.start_span(
                    span_id=code_gen_span_id,
                    name="Code Generation",
                    tags={
                        "operation": "code_generation",
                        "function": self.code_gen_fn.__name__,
                        "prompt_length": str(len(prompt.text)),
                    },
                    metadata={
                        "prompt_preview": prompt.text[:100] + "..."
                        if len(prompt.text) > 100
                        else prompt.text
                    },
                )

            files_dict = self.code_gen_fn(
                self.ai,
                prompt,
                self.memory,
                self.preprompts_holder,
                parent_span_id=code_gen_span_id,
            )

            # End code generation span
            if observability and observability.is_enabled() and code_gen_span_id:
                observability.end_span(
                    span_id=code_gen_span_id,
                    result={"files_generated": len(files_dict)},
                )

            # Start entrypoint generation span
            if observability and observability.is_enabled():
                entrypoint_span_id = str(uuid4())
                observability.start_span(
                    span_id=entrypoint_span_id,
                    name="Run Script Generation(Entrypoint)",
                    tags={
                        "operation": "entrypoint_generation",
                        "file_count": str(len(files_dict)),
                    },
                )

            entrypoint = gen_entrypoint(
                self.ai,
                prompt,
                files_dict,
                self.memory,
                self.preprompts_holder,
                parent_span_id=entrypoint_span_id,
            )
            combined_dict = {**files_dict, **entrypoint}
            files_dict = FilesDict(combined_dict)

            # End entrypoint generation span
            if observability and observability.is_enabled() and entrypoint_span_id:
                observability.end_span(
                    span_id=entrypoint_span_id,
                    result={"entrypoint_generated": len(entrypoint)},
                )

            # Start code processing span
            if observability and observability.is_enabled():
                execution_span_id = str(uuid4())
                observability.start_span(
                    span_id=execution_span_id,
                    name="Code Processing",
                    tags={
                        "operation": "code_processing",
                        "function": self.process_code_fn.__name__,
                        "total_files": str(len(files_dict)),
                    },
                )
            files_dict = self.process_code_fn(
                self.ai,
                self.execution_env,
                files_dict,
                preprompts_holder=self.preprompts_holder,
                prompt=prompt,
                memory=self.memory,
                parent_span_id=execution_span_id,
            )

            # Log processing summary event
            if observability and observability.is_enabled() and execution_span_id:
                observability.log_event(
                    event_id=str(uuid4()),
                    event_type="processing_summary",
                    data=None,
                    tags={"operation": "code_processing", "phase": "summary"},
                    metadata={
                        "final_file_count": len(files_dict),
                        "final_file_names": list(files_dict.keys()),
                        "execution_attempted": True,  # Always true in this flow
                        "execution_success": True,  # Assume success unless exception is raised
                    },
                )

            # End code processing span
            if observability and observability.is_enabled() and execution_span_id:
                observability.end_span(
                    span_id=execution_span_id,
                    result={"final_file_count": len(files_dict)},
                )

            return files_dict

        except Exception as e:
            # End any active spans with error
            if observability and observability.is_enabled():
                if code_gen_span_id:
                    observability.end_span(span_id=code_gen_span_id, error=e)
                if entrypoint_span_id:
                    observability.end_span(span_id=entrypoint_span_id, error=e)
                if execution_span_id:
                    observability.end_span(span_id=execution_span_id, error=e)
            raise

    def improve(
        self,
        files_dict: FilesDict,
        prompt: Prompt,
        execution_command: Optional[str] = None,
        diff_timeout=3,
        parent_span_id: Optional[str] = None,
    ) -> FilesDict:
        """
        Improves an existing piece of code using the AI and step bundle based on the provided prompt.

        Parameters
        ----------
        files_dict : FilesDict
            An instance of `FilesDict` containing the code to be improved.
        prompt : str
            A string prompt that guides the code improvement process.
        execution_command : str, optional
            An optional command to execute the code. If not provided, the default execution command is used.

        Returns
        -------
        FilesDict
            An instance of the `FilesDict` class containing the improved code.
        """
        observability = None
        improve_span_id = None

        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
            except Exception:
                pass  # Continue without observability

        try:
            # Start code improvement span
            if observability and observability.is_enabled():
                from uuid import uuid4

                improve_span_id = str(uuid4())
                observability.start_span(
                    parent_span_id=parent_span_id if parent_span_id else None,
                    span_id=improve_span_id,
                    name="Code Improvement",
                    tags={
                        "operation": "code_improvement",
                        "function": self.improve_fn.__name__,
                        "initial_file_count": str(len(files_dict)),
                        "prompt_length": str(len(prompt.text)),
                        "diff_timeout": str(diff_timeout),
                    },
                    metadata={
                        "prompt_preview": prompt.text[:100] + "..."
                        if len(prompt.text) > 100
                        else prompt.text,
                        "file_list": list(files_dict.keys())[:10],  # Log first 10 files
                    },
                )

            files_dict = self.improve_fn(
                self.ai,
                prompt,
                files_dict,
                self.memory,
                self.preprompts_holder,
                diff_timeout=diff_timeout,
            )

            # End code improvement span
            if observability and observability.is_enabled() and improve_span_id:
                observability.end_span(
                    span_id=improve_span_id,
                    result={
                        "final_file_count": len(files_dict),
                        "files_modified": list(files_dict.keys())[
                            :10
                        ],  # Log first 10 modified files
                    },
                )

            # entrypoint = gen_entrypoint(
            #     self.ai, prompt, files_dict, self.memory, self.preprompts_holder
            # )
            # combined_dict = {**files_dict, **entrypoint}
            # files_dict = FilesDict(combined_dict)
            # files_dict = self.process_code_fn(
            #     self.ai,
            #     self.execution_env,
            #     files_dict,
            #     preprompts_holder=self.preprompts_holder,
            #     prompt=prompt,
            #     memory=self.memory,
            # )

            return files_dict

        except Exception as e:
            # End span with error
            if observability and observability.is_enabled() and improve_span_id:
                observability.end_span(span_id=improve_span_id, error=e)
            raise
