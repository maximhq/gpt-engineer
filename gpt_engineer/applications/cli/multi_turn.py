"""
Multi-turn conversation module for GPT-Engineer.

This module provides functionality for engaging in multi-turn conversations
with the AI, dynamically determining the appropriate mode for each interaction.

The multi-turn mode allows users to have continuous conversations with GPT-Engineer,
where each user input is automatically analyzed to determine the most appropriate
action mode. This creates a more natural and interactive development experience.

Available Modes:
- clarify: Discuss and clarify requirements before implementation
- debug: Fix errors and issues in existing code
- improve: Enhance existing functionality
- generate: Create new code from scratch
- misc: Provide information and explanations

Usage:
    gpt-engineer --multi-turn project_path
    gpt-engineer -mt project_path

Features:
- Dynamic mode selection based on user input
- Continuous conversation loop
- Conversation history tracking
- Automatic file management
- Git integration
- User feedback collection

Example Conversation Flow:
1. User: "I want to build a web app"
   → Mode: generate (creates initial code)
2. User: "Add user authentication"
   → Mode: improve (enhances existing code)
3. User: "The login is not working"
   → Mode: debug (fixes issues)
4. User: "Explain how the authentication works"
   → Mode: misc (provides explanation)
5. User: "Let's discuss the database design"
   → Mode: clarify (discusses requirements)
"""

from pathlib import Path
from typing import Optional

from termcolor import colored

from gpt_engineer.applications.cli.cli_agent import CliAgent
from gpt_engineer.applications.cli.collect import collect_learnings
from gpt_engineer.applications.cli.file_selector import FileSelector
from gpt_engineer.applications.cli.learning import human_review_input
from gpt_engineer.core.ai import AI
from gpt_engineer.core.default.disk_execution_env import DiskExecutionEnv
from gpt_engineer.core.default.disk_memory import DiskMemory
from gpt_engineer.core.default.file_store import FileStore
from gpt_engineer.core.default.paths import memory_path
from gpt_engineer.core.default.steps import (
    execute_entrypoint,
    gen_code,
    gen_entrypoint,
    handle_improve_mode,
)
from gpt_engineer.core.files_dict import FilesDict
from gpt_engineer.core.git import stage_uncommitted_to_git
from gpt_engineer.core.preprompts_holder import PrepromptsHolder
from gpt_engineer.core.prompt import Prompt
from gpt_engineer.tools.custom_steps import clarified_gen, self_heal

# Import observability
try:
    from gpt_engineer.core.maxim_observability import get_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


class MultiTurnEngine:
    """
    Handles multi-turn conversations with the AI, dynamically determining
    the appropriate mode for each interaction.
    """

    def __init__(self, ai: AI, project_path: str, preprompts_holder: PrepromptsHolder):
        """
        Initialize the multi-turn engine.

        Parameters
        ----------
        ai : AI
            The AI instance to use for conversations.
        project_path : str
            The path to the project directory.
        preprompts_holder : PrepromptsHolder
            The preprompts holder for managing templates.
        """
        self.ai = ai
        self.project_path = project_path
        self.preprompts_holder = preprompts_holder
        self.memory = DiskMemory(memory_path(project_path))
        self.execution_env = DiskExecutionEnv()
        self.files = FileStore(project_path)

        # Initialize the agent
        self.agent = CliAgent.with_default_config(
            self.memory,
            self.execution_env,
            ai=ai,
            preprompts_holder=preprompts_holder,
        )

        # Track conversation state
        self.conversation_history = []

        # Load existing files from the project directory
        self.current_files = self._load_existing_files()

        # Initialize observability (use existing session from main.py)
        self.observability = None

        if OBSERVABILITY_AVAILABLE:
            try:
                self.observability = get_observability()
                if self.observability.is_enabled():
                    print("Using existing Maxim session for multi-turn mode")

            except Exception as e:
                print(f"⚠️  Failed to get observability: {e}")
                self.observability = None

    def _load_existing_files(self) -> FilesDict:
        """
        Load existing files from the project directory.

        Returns
        -------
        FilesDict
            Dictionary containing existing files and their contents.
        """
        try:
            print(f"🔍 Loading existing files from: {self.project_path}")

            # Use FileSelector to get existing files
            file_selector = FileSelector(self.project_path)
            files_dict, is_linting = file_selector.ask_for_files(
                skip_file_selection=True
            )

            if files_dict:
                print(
                    f"📁 Loaded {len(files_dict)} existing files from project directory"
                )
                for file_path in list(files_dict.keys())[:5]:  # Show first 5 files
                    print(f"   - {file_path}")
                if len(files_dict) > 5:
                    print(f"   ... and {len(files_dict) - 5} more files")
                return files_dict
            else:
                print("📁 No existing files found in project directory")
                return FilesDict()

        except Exception as e:
            print(f"⚠️  Error loading existing files: {e}")
            print("Starting with empty file context")
            return FilesDict()

    def _reload_files_from_disk(self) -> None:
        """
        Reload files from disk to ensure we have the most up-to-date context.
        This is useful if files were created manually outside of the multi-turn mode.
        """
        try:
            # Use FileSelector to get current files from disk
            file_selector = FileSelector(self.project_path)
            files_dict, is_linting = file_selector.ask_for_files(
                skip_file_selection=True
            )

            if files_dict:
                # Update current files with what's on disk
                self.current_files = files_dict
                print(f"🔄 Reloaded {len(files_dict)} files from disk")

        except Exception as e:
            print(f"⚠️  Error reloading files from disk: {e}")
            # Keep existing current_files if reload fails

    def _determine_mode(self, prompt: Prompt) -> str:
        """
        Determines which mode to run based on the user's prompt using an LLM call.

        Parameters
        ----------
        prompt : Prompt
            The user's prompt to analyze.

        Returns
        -------
        str
            The mode to run: 'clarify', 'debug', 'improve', 'generate', or 'misc'.
        """
        # Start mode determination span
        mode_span_id = None
        if self.observability and self.observability.is_enabled():
            from uuid import uuid4

            mode_span_id = str(uuid4())
            self.observability.start_span(
                span_id=mode_span_id,
                name="Mode Determination",
                tags={
                    "operation": "mode_determination",
                    "prompt_length": str(len(prompt.text)),
                },
                metadata={
                    "prompt_preview": prompt.text[:200] + "..."
                    if len(prompt.text) > 200
                    else prompt.text,
                    "prompt_length": len(prompt.text),
                },
            )
        system_prompt = """You are a mode selector for GPT-Engineer. Based on the user's prompt, determine which mode would be most appropriate.

Available modes:
- debug: When the user mentions errors, bugs, crashes, or problems that need fixing
- improve: When the user wants to add features, enhance existing code, or modify functionality
- generate: When the user wants to create new components, files, or functionality from scratch
- misc: When the user asks for explanations, documentation, or informational requests that don't involve coding

Answer in single word: debug, improve, generate, or misc.

Mode Selection Guidelines:
- debug: Use when user mentions errors, bugs, crashes, or problems that need fixing
- improve: Use when user wants to add features, enhance existing code, or modify functionality
- generate: Use when user wants to create, generate, or add new components, files, or functionality from scratch
- misc: Use when user asks for explanations, documentation, or informational requests that don't involve coding

Examples:
- "Fix the error in the code" -> debug
- "The code has bugs" -> debug
- "There's an issue with the function" -> debug
- "The app is crashing" -> debug
- "Add a new feature" -> improve
- "Make it faster" -> improve
- "Improve the performance" -> improve
- "Enhance the UI" -> improve
- "Create a new component" -> generate
- "Build a new app" -> generate
- "Generate a new file" -> generate
- "Create a new module" -> generate
- "Build a new feature from scratch" -> generate
- "Explain the code you have generated" -> misc
- "What does this function do?" -> generate
- "Show me the structure of the code" -> misc
- "How does this work?" -> misc
- "Tell me about the architecture" -> misc"""

        try:
            messages = self.ai.start(
                system_prompt, prompt.text, step_name="mode_selection"
            )
            mode = messages[-1].content.strip().lower()

            # Validate the mode
            valid_modes = ["clarify", "debug", "improve", "generate", "misc"]
            if mode not in valid_modes:
                print(f"⚠️  Invalid mode '{mode}' detected, defaulting to 'improve'")
                mode = "improve"

            print(f"🎯 Selected mode: {mode}")

            return mode

        except Exception as e:
            print(f"⚠️  Error determining mode: {e}, defaulting to 'improve'")

            # Log mode determination error
            if self.observability and self.observability.is_enabled():
                self.observability.log_error(
                    error=e,
                    context={
                        "operation": "mode_determination",
                        "prompt_length": len(prompt.text),
                    },
                    tags={"error_type": "mode_determination_error"},
                )

            return "improve"

        finally:
            # End mode determination span
            if self.observability and self.observability.is_enabled() and mode_span_id:
                self.observability.end_span(
                    span_id=mode_span_id,
                    result={"selected_mode": mode},
                )

    def _execute_mode(self, mode: str, prompt: Prompt) -> FilesDict:
        """
        Execute the determined mode with the given prompt.

        Parameters
        ----------
        mode : str
            The mode to execute.
        prompt : Prompt
            The user's prompt.

        Returns
        -------
        FilesDict
            The resulting files dictionary.
        """
        print(f"\n🚀 Executing {mode} mode...")

        try:
            if mode == "clarify":
                files_dict = self._execute_clarify_mode(prompt)
            elif mode == "debug":
                files_dict = self._execute_debug_mode(prompt)
            elif mode == "improve":
                files_dict = self._execute_improve_mode(prompt)
            elif mode == "generate":
                files_dict = self._execute_generate_mode(prompt)
            elif mode == "misc":
                files_dict = self._execute_misc_mode(prompt)
            else:
                print(f"⚠️  Unknown mode '{mode}', defaulting to improve")
                files_dict = self._execute_improve_mode(prompt)

            return files_dict

        except Exception as e:
            print(f"⚠️  Error executing {mode} mode: {e}")

            # Log mode execution error
            if self.observability and self.observability.is_enabled():
                self.observability.log_error(
                    error=e,
                    context={
                        "operation": "mode_execution",
                        "mode": mode,
                        "prompt_length": len(prompt.text),
                    },
                    tags={"error_type": "mode_execution_error", "mode": mode},
                )

            # Return current files as fallback
            return self.current_files

    def _execute_clarify_mode(self, prompt: Prompt) -> FilesDict:
        """Execute clarify mode - discuss requirements before implementation."""
        print("💬 Clarify mode: Engaging in discussion about requirements...")

        # Initialize observability variables
        observability = None
        clarify_span_id = None

        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
            except Exception:
                pass  # Continue without observability

        try:
            # Start clarify mode span
            if observability and observability.is_enabled():
                from uuid import uuid4

                clarify_span_id = str(uuid4())
                observability.start_span(
                    span_id=clarify_span_id,
                    name="Clarify Mode",
                    tags={
                        "operation": "clarify_mode",
                        "prompt_length": str(len(prompt.text)),
                    },
                    metadata={
                        "prompt_preview": prompt.text[:200] + "..."
                        if len(prompt.text) > 200
                        else prompt.text,
                    },
                )

            # Check if we have existing files to provide context
            if self.current_files:
                print(
                    f"📁 Clarifying requirements in context of {len(self.current_files)} existing files..."
                )

                # Create a context-aware prompt that includes existing files
                context_prompt = Prompt(
                    f"{prompt.text}\n\n"
                    f"Note: This project already has the following files:\n"
                    f"{chr(10).join([f'- {file_path}' for file_path in self.current_files.keys()])}\n"
                    f"Please discuss requirements and clarifications in the context of these existing files."
                )

                # Use the clarified_gen function with context-aware prompt
                files_dict = clarified_gen(
                    self.ai, context_prompt, self.memory, self.preprompts_holder
                )

                # Merge with existing files instead of overwriting
                merged_files = FilesDict({**self.current_files, **files_dict})

                # Apply the merged files
                self.files.push(merged_files)
                self.current_files = merged_files

                print("✅ Clarification completed, merged with existing files")

            else:
                print("🆕 Clarifying requirements for new project...")

                # Use the original clarified_gen function for new projects
                files_dict = clarified_gen(
                    self.ai, prompt, self.memory, self.preprompts_holder
                )

                # Apply the files
                self.files.push(files_dict)
                self.current_files = files_dict

            # End clarify mode span
            if observability and observability.is_enabled() and clarify_span_id:
                observability.end_span(
                    span_id=clarify_span_id,
                    result={"files_generated": len(files_dict)},
                )

            return files_dict

        except Exception as e:
            print(f"⚠️  Error in clarify mode: {e}")

            # End span with error
            if observability and observability.is_enabled() and clarify_span_id:
                observability.end_span(span_id=clarify_span_id, error=e)

            # Log clarify mode error
            if self.observability and self.observability.is_enabled():
                self.observability.log_error(
                    error=e,
                    context={
                        "operation": "clarify_mode",
                        "prompt_length": len(prompt.text),
                    },
                    tags={"error_type": "clarify_mode_error"},
                )

            # Return current files as fallback
            return self.current_files

    def _execute_debug_mode(self, prompt: Prompt) -> FilesDict:
        """Execute debug mode - fix errors and issues."""
        print("🔧 Debug mode: Attempting to fix issues...")

        # Initialize observability variables
        observability = None
        debug_span_id = None

        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
            except Exception:
                pass  # Continue without observability

        try:
            # Start debug mode span
            if observability and observability.is_enabled():
                from uuid import uuid4

                debug_span_id = str(uuid4())
                observability.start_span(
                    span_id=debug_span_id,
                    name="Debug Mode",
                    tags={
                        "operation": "debug_mode",
                        "prompt_length": str(len(prompt.text)),
                    },
                    metadata={
                        "prompt_preview": prompt.text[:200] + "..."
                        if len(prompt.text) > 200
                        else prompt.text,
                    },
                )

            if observability and observability.is_enabled():
                trace_output_content = (
                    "Debug mode successfully completed, no modification needed"
                )
                observability.set_trace_output(trace_output_content)
                print(f"🔍 Trace Output Set: {trace_output_content}")

            # Use self-heal functionality
            files_dict = self_heal(
                self.ai,
                self.execution_env,
                self.current_files,
                prompt=prompt,
                preprompts_holder=self.preprompts_holder,
                memory=self.memory,
            )

            # Apply the files
            self.files.push(files_dict)
            self.current_files = files_dict

            # End debug mode span
            if observability and observability.is_enabled() and debug_span_id:
                observability.end_span(
                    span_id=debug_span_id,
                    result={"files_modified": len(files_dict)},
                )

            return files_dict

        except Exception as e:
            print(f"⚠️  Error in debug mode: {e}")

            # End span with error
            if observability and observability.is_enabled() and debug_span_id:
                observability.end_span(span_id=debug_span_id, error=e)

            # Log debug mode error
            if self.observability and self.observability.is_enabled():
                self.observability.log_error(
                    error=e,
                    context={
                        "operation": "debug_mode",
                        "prompt_length": len(prompt.text),
                    },
                    tags={"error_type": "debug_mode_error"},
                )

            # Return current files as fallback
            return self.current_files

    def _execute_improve_mode(self, prompt: Prompt) -> FilesDict:
        """Execute improve mode - enhance existing functionality."""
        print("✨ Improve mode: Enhancing existing code...")

        # Initialize observability variables
        observability = None
        improve_span_id = None

        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
            except Exception:
                pass  # Continue without observability

        try:
            # Use the improve functionality
            files_dict = handle_improve_mode(
                prompt,
                self.agent,
                self.memory,
                self.current_files,
                diff_timeout=3,
            )

            if files_dict is not None:
                # Apply the files
                self.files.push(files_dict)
                self.current_files = files_dict

            # End improve mode span

            return files_dict or self.current_files

        except Exception as e:
            print(f"⚠️  Error in improve mode: {e}")

            # End span with error
            if observability and observability.is_enabled() and improve_span_id:
                observability.end_span(span_id=improve_span_id, error=e)

            # Log improve mode error
            if self.observability and self.observability.is_enabled():
                self.observability.log_error(
                    error=e,
                    context={
                        "operation": "improve_mode",
                        "prompt_length": len(prompt.text),
                    },
                    tags={"error_type": "improve_mode_error"},
                )

            # Return current files as fallback
            return self.current_files

    def _execute_generate_mode(self, prompt: Prompt) -> FilesDict:
        """Execute generate mode - create new code from scratch."""
        print("🆕 Generate mode: Creating new code...")

        # Initialize observability variables
        observability = None
        generate_span_id = None
        entrypoint_span_id = None
        execution_span_id = None

        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
            except Exception:
                pass  # Continue without observability

        try:
            # Start generate mode span
            if observability and observability.is_enabled():
                from uuid import uuid4

                generate_span_id = str(uuid4())
                observability.start_span(
                    span_id=generate_span_id,
                    name="Generate Mode",
                    tags={
                        "operation": "generate_mode",
                        "prompt_length": str(len(prompt.text)),
                    },
                    metadata={
                        "prompt_preview": prompt.text[:200] + "..."
                        if len(prompt.text) > 200
                        else prompt.text,
                    },
                )

            # Check if we have existing files
            if self.current_files:
                print(f"📁 Merging with {len(self.current_files)} existing files...")

                # Create a prompt that includes context about existing files
                context_prompt = Prompt(
                    f"{prompt.text}\n\n"
                    f"Note: This project already has the following files:\n"
                    f"{chr(10).join([f'- {file_path}' for file_path in self.current_files.keys()])}\n"
                    f"Please create new files or modify existing ones as needed. "
                    f"Do not overwrite existing files unless explicitly requested."
                )

                print("🔧 Calling gen_code with context prompt...")

                # Generate new files using gen_code
                new_files_dict = gen_code(
                    self.ai,
                    context_prompt,
                    self.memory,
                    self.preprompts_holder,
                    parent_span_id=generate_span_id,  # ← Pass parent_span_id to generate_span_id instead
                )
                print(f"✅ gen_code completed, generated {len(new_files_dict)} files")

                # Generate entrypoint that considers all files (existing + new)
                combined_files = FilesDict({**self.current_files, **new_files_dict})
                print(
                    f"🔧 Calling gen_entrypoint with {len(combined_files)} combined files..."
                )

                # Start entrypoint generation span
                if observability and observability.is_enabled():
                    entrypoint_span_id = str(uuid4())
                    observability.start_span(
                        span_id=entrypoint_span_id,
                        name="Run Script Generation(Entrypoint)",
                        parent_span_id=generate_span_id,
                        tags={
                            "operation": "entrypoint_generation",
                            "file_count": str(len(combined_files)),
                        },
                    )

                entrypoint_files = gen_entrypoint(
                    self.ai,
                    context_prompt,
                    combined_files,
                    self.memory,
                    self.preprompts_holder,
                    parent_span_id=entrypoint_span_id,  # ← Pass parent_span_id
                )
                print(
                    f"✅ gen_entrypoint completed, generated {len(entrypoint_files)} files"
                )

                # End entrypoint generation span
                if observability and observability.is_enabled() and entrypoint_span_id:
                    observability.end_span(
                        span_id=entrypoint_span_id,
                        result={"entrypoint_generated": len(entrypoint_files)},
                    )

                # Start code processing span
                if observability and observability.is_enabled():
                    execution_span_id = str(uuid4())
                    observability.start_span(
                        span_id=execution_span_id,
                        name="Code Processing",
                        parent_span_id=generate_span_id,
                        tags={
                            "operation": "code_processing",
                            "function": "execute_entrypoint",
                            "total_files": str(
                                len(combined_files) + len(entrypoint_files)
                            ),
                        },
                    )

                # Merge all files: existing + new + entrypoint
                merged_files = FilesDict(
                    {**self.current_files, **new_files_dict, **entrypoint_files}
                )
                print(f"📦 Final merge: {len(merged_files)} total files")

                # Execute the entrypoint (code processing)
                merged_files = execute_entrypoint(
                    self.ai,
                    self.execution_env,
                    merged_files,
                    prompt=context_prompt,
                    preprompts_holder=self.preprompts_holder,
                    memory=self.memory,
                    parent_span_id=execution_span_id,  # ← Pass parent_span_id
                )

                # End code processing span
                if observability and observability.is_enabled() and execution_span_id:
                    observability.end_span(
                        span_id=execution_span_id,
                        result={"final_file_count": len(merged_files)},
                    )

                print(
                    f"✅ Generated {len(new_files_dict)} new files + {len(entrypoint_files)} entrypoint files, merged with {len(self.current_files)} existing files"
                )

            else:
                # No existing files, generate from scratch using direct function calls
                print("🆕 Generating new project from scratch...")

                # Generate code using gen_code
                new_files_dict = gen_code(
                    self.ai,
                    prompt,
                    self.memory,
                    self.preprompts_holder,
                    parent_span_id=generate_span_id,
                )
                print(f"✅ gen_code completed, generated {len(new_files_dict)} files")

                # Generate entrypoint
                entrypoint_files = gen_entrypoint(
                    self.ai,
                    prompt,
                    new_files_dict,
                    self.memory,
                    self.preprompts_holder,
                    parent_span_id=generate_span_id,
                )
                print(
                    f"✅ gen_entrypoint completed, generated {len(entrypoint_files)} files"
                )

                # Merge all files
                merged_files = FilesDict({**new_files_dict, **entrypoint_files})
                print(f"✅ Generated {len(merged_files)} total files")

                # Execute the entrypoint (code processing)
                merged_files = execute_entrypoint(
                    self.ai,
                    self.execution_env,
                    merged_files,
                    prompt=prompt,
                    preprompts_holder=self.preprompts_holder,
                    memory=self.memory,
                    parent_span_id=generate_span_id,
                )
                print("✅ Code execution completed")

            print(f"💾 Writing {len(merged_files)} files to disk...")
            # Apply the files
            self.files.push(merged_files)
            self.current_files = merged_files
            print("✅ Files written to disk successfully")

            # End generate mode span
            if observability and observability.is_enabled() and generate_span_id:
                observability.end_span(
                    span_id=generate_span_id,
                    result={"final_file_count": len(merged_files)},
                )

            return merged_files

        except Exception as e:
            print(f"⚠️  Error in generate mode: {e}")
            import traceback

            traceback.print_exc()

            # End any active spans with error
            if observability and observability.is_enabled():
                if generate_span_id:
                    observability.end_span(span_id=generate_span_id, error=e)
                if entrypoint_span_id:
                    observability.end_span(span_id=entrypoint_span_id, error=e)
                if execution_span_id:
                    observability.end_span(span_id=execution_span_id, error=e)

            # Log generate mode error
            if self.observability and self.observability.is_enabled():
                self.observability.log_error(
                    error=e,
                    context={
                        "operation": "generate_mode",
                        "prompt_length": len(prompt.text),
                    },
                    tags={"error_type": "generate_mode_error"},
                )

            # Return current files as fallback
            return self.current_files

    def _execute_misc_mode(self, prompt: Prompt) -> FilesDict:
        """Execute misc mode - provide information and explanations."""
        print("ℹ️  Misc mode: Providing information...")

        # Initialize observability variables
        observability = None
        misc_span_id = None

        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
            except Exception:
                pass  # Continue without observability

        try:
            # Start misc mode span
            if observability and observability.is_enabled():
                from uuid import uuid4

                misc_span_id = str(uuid4())
                observability.start_span(
                    span_id=misc_span_id,
                    name="Misc Mode",
                    tags={
                        "operation": "misc_mode",
                        "prompt_length": str(len(prompt.text)),
                    },
                    metadata={
                        "prompt_preview": prompt.text[:200] + "..."
                        if len(prompt.text) > 200
                        else prompt.text,
                    },
                )

            # For misc mode, we might just provide information without changing files
            # or we could generate documentation/explanation files
            if "explain" in prompt.text.lower() or "what" in prompt.text.lower():
                # Generate explanation files
                explanation_prompt = Prompt(
                    f"Please explain the current codebase and provide documentation. "
                    f"User request: {prompt.text}"
                )
                files_dict = gen_code(
                    self.ai,
                    explanation_prompt,
                    self.memory,
                    self.preprompts_holder,
                )

                # Apply the files
                self.files.push(files_dict)
                self.current_files = files_dict

                # End misc mode span
                if observability and observability.is_enabled() and misc_span_id:
                    observability.end_span(
                        span_id=misc_span_id,
                        result={"files_generated": len(files_dict)},
                    )

                return files_dict
            else:
                # Just return current files without changes
                # End misc mode span
                if observability and observability.is_enabled() and misc_span_id:
                    observability.end_span(
                        span_id=misc_span_id,
                        result={"files_generated": 0},
                    )

                return self.current_files

        except Exception as e:
            print(f"⚠️  Error in misc mode: {e}")

            # End span with error
            if observability and observability.is_enabled() and misc_span_id:
                observability.end_span(span_id=misc_span_id, error=e)

            # Log misc mode error
            if self.observability and self.observability.is_enabled():
                self.observability.log_error(
                    error=e,
                    context={
                        "operation": "misc_mode",
                        "prompt_length": len(prompt.text),
                    },
                    tags={"error_type": "misc_mode_error"},
                )

            # Return current files as fallback
            return self.current_files

    def _get_user_input(self) -> Optional[str]:
        """
        Get user input for the next turn.

        Returns
        -------
        Optional[str]
            The user's input, or None if they want to exit.
        """
        print("\n" + "=" * 50)
        print(colored("💬 Multi-turn conversation active", "cyan"))
        print("=" * 50)
        print("Type your next request, or type 'exit' to end the conversation.")
        print("You can ask me to:")
        print("  • Fix bugs or issues")
        print("  • Improve existing code")
        print("  • Generate new features")
        print("=" * 50)

        user_input = input("\n💭 Your request: ").strip()

        if user_input.lower() in ["exit", "quit", "end", "stop"]:
            return None

        return user_input

    def run_conversation(self, initial_prompt: Prompt) -> None:
        """
        Run the multi-turn conversation loop.

        Parameters
        ----------
        initial_prompt : Prompt
            The initial prompt to start the conversation.
        """
        print(colored("🔄 Starting multi-turn conversation mode...", "green"))

        current_prompt = initial_prompt
        turn_number = 0

        while True:
            turn_number += 1

            # Reload files from disk to ensure we have the most up-to-date context
            if turn_number > 1:  # Skip for the first turn since we just loaded files
                self._reload_files_from_disk()

            # Start trace for this conversation turn
            trace_id = None
            if self.observability and self.observability.is_enabled():
                from uuid import uuid4

                trace_id = str(uuid4())

                trace_tags = {
                    "operation": "multi_turn_conversation",
                    "turn_number": str(turn_number),
                    "model": self.ai.model_name,
                    "project_path": str(Path(self.project_path).absolute()),
                }

                trace_metadata = {
                    "turn_number": turn_number,
                    "prompt_preview": current_prompt.text[:200] + "..."
                    if len(current_prompt.text) > 200
                    else current_prompt.text,
                    "prompt_length": len(current_prompt.text),
                    "conversation_history_length": len(self.conversation_history),
                    "current_files_count": len(self.current_files),
                }

                self.observability.start_trace(
                    trace_id=trace_id,
                    name=f"Conversation Turn {turn_number}",
                    tags=trace_tags,
                    metadata=trace_metadata,
                    # Use existing session from main.py
                )

                # Set the prompt as trace input
                self.observability.set_trace_input(current_prompt.text)

            # Determine the mode for this turn
            mode = self._determine_mode(current_prompt)

            # Execute the mode
            files_dict = None  # Initialize files_dict
            try:
                files_dict = self._execute_mode(mode, current_prompt)

                # Attach generated files to the current trace
                if OBSERVABILITY_AVAILABLE and files_dict and trace_id:
                    try:
                        observability = get_observability()
                        if observability and observability.is_enabled():
                            # Attach each generated file to the current trace
                            for filename, content in files_dict.items():
                                try:
                                    # Determine MIME type based on file extension
                                    mime_type = None
                                    if filename.endswith(".py"):
                                        mime_type = "text/x-python"
                                    elif filename.endswith(".js"):
                                        mime_type = "text/javascript"
                                    elif filename.endswith(".html"):
                                        mime_type = "text/html"
                                    elif filename.endswith(".css"):
                                        mime_type = "text/css"
                                    elif filename.endswith(".json"):
                                        mime_type = "application/json"
                                    elif filename.endswith(".md"):
                                        mime_type = "text/markdown"
                                    elif filename.endswith(".txt"):
                                        mime_type = "text/plain"
                                    elif filename.endswith(".toml"):
                                        mime_type = "text/x-toml"
                                    elif filename.endswith(".yml") or filename.endswith(
                                        ".yaml"
                                    ):
                                        mime_type = "text/x-yaml"
                                    else:
                                        mime_type = "text/plain"

                                    # Add file attachment to the current trace
                                    observability.add_file_attachment(
                                        filename=filename,
                                        content=content,
                                        mime_type=mime_type,
                                        target="trace",
                                        target_id=trace_id,
                                    )
                                    print(f"📎 Attached {filename} to trace {trace_id}")
                                except Exception as e:
                                    print(
                                        f"⚠️  Failed to attach {filename} to trace: {e}"
                                    )
                    except Exception as e:
                        print(f"⚠️  Failed to attach files to trace: {e}")

                # Stage changes to git
                stage_uncommitted_to_git(
                    self.project_path, files_dict, mode == "improve"
                )

                # Add to conversation history
                self.conversation_history.append(
                    {
                        "prompt": current_prompt.text,
                        "mode": mode,
                        "files_count": len(files_dict),
                    }
                )

                # Log turn completion
                if self.observability and self.observability.is_enabled():
                    self.observability.log_event(
                        event_id=str(uuid4()),
                        event_type="turn_completed",
                        metadata={
                            "turn_number": turn_number,
                            "mode": mode,
                            "files_generated": len(files_dict),
                            "prompt_length": len(current_prompt.text),
                        },
                        tags={
                            "operation": "turn_completion",
                            "mode": mode,
                            "turn_number": str(turn_number),
                        },
                    )

            except Exception as e:
                print(f"⚠️  Error in {mode} mode: {e}")
                print("Continuing with conversation...")

                # Log turn error
                if self.observability and self.observability.is_enabled():
                    self.observability.log_error(
                        error=e,
                        context={
                            "operation": "turn_execution",
                            "turn_number": turn_number,
                            "mode": mode,
                        },
                        tags={"error_type": "turn_execution_error", "mode": mode},
                    )

                # Ensure files_dict is defined even on error
                if files_dict is None:
                    files_dict = self.current_files

            # Get next user input
            next_input = self._get_user_input()

            if next_input is None:
                print(colored("👋 Ending multi-turn conversation. Goodbye!", "green"))
                
                # Set trace output and end trace when user exits
                if self.observability and self.observability.is_enabled() and trace_id:
                    # Calculate comprehensive metrics for trace output
                    total_size = sum(len(content) for content in files_dict.values())
                    file_types = list(
                        set(Path(fname).suffix for fname in files_dict.keys())
                    )

                    # Cost and token information
                    cost_info = {}
                    token_info = {}
                    if self.ai.token_usage_log.is_openai_model():
                        cost_info = {
                            "total_cost": self.ai.token_usage_log.usage_cost(),
                            "currency": "USD",
                            "model": self.ai.model_name,
                        }
                        token_info = {
                            "total_tokens": self.ai.token_usage_log.total_tokens(),
                            "cost_per_token": self.ai.token_usage_log.usage_cost()
                            / max(self.ai.token_usage_log.total_tokens(), 1),
                        }

                    # Format as markdown
                    trace_output = f"""# 🎯 Turn {turn_number} Summary

## 📊 Overview
- **Mode**: {mode.title()}
- **Files Generated**: {len(files_dict)} files ({total_size:,} characters)
- **Model Used**: {self.ai.model_name}
- **Cost**: ${cost_info.get('total_cost', 0):.4f} ({token_info.get('total_tokens', 0):,} tokens)

## 📁 Files by Type
{chr(10).join([f'- **{ext if ext != "no_extension" else "other"}**: {", ".join(files)}' for ext, files in {ext: [f for f in files_dict.keys() if Path(f).suffix == ext] for ext in file_types}.items()])}

## 🚀 Context
- **Project Path**: {str(Path(self.project_path).absolute())}
- **Prompt Length**: {len(current_prompt.text)} characters
- **Turn Number**: {turn_number}
- **Total Turns**: {len(self.conversation_history)}

---
*Generated by GPT-Engineer Multi-turn Mode*
"""

                    self.observability.set_trace_output(trace_output)
                    print(f"🔍 Trace Output Set: {trace_output}")
                    self.observability.end_trace(trace_id)
                
                # Collect user feedback before ending
                print("\n" + "=" * 50)
                print(colored("📝 Session Feedback", "cyan"))
                print("=" * 50)
                print("Please provide feedback about your multi-turn session experience:")
                
                review = human_review_input(multi_turn=False)  # Enable feedback collection for exit
                if review:
                    if self.observability and self.observability.is_enabled():
                        self.observability.add_session_feedback(review)
                
                break

            # Set trace output and end trace AFTER confirming user wants to continue
            if self.observability and self.observability.is_enabled() and trace_id:
                # Calculate comprehensive metrics for trace output
                total_size = sum(len(content) for content in files_dict.values())
                file_types = list(
                    set(Path(fname).suffix for fname in files_dict.keys())
                )

                # Cost and token information
                cost_info = {}
                token_info = {}
                if self.ai.token_usage_log.is_openai_model():
                    cost_info = {
                        "total_cost": self.ai.token_usage_log.usage_cost(),
                        "currency": "USD",
                        "model": self.ai.model_name,
                    }
                    token_info = {
                        "total_tokens": self.ai.token_usage_log.total_tokens(),
                        "cost_per_token": self.ai.token_usage_log.usage_cost()
                        / max(self.ai.token_usage_log.total_tokens(), 1),
                    }

                # Format as markdown
                trace_output = f"""# 🎯 Turn {turn_number} Summary

## 📊 Overview
- **Mode**: {mode.title()}
- **Files Generated**: {len(files_dict)} files ({total_size:,} characters)
- **Model Used**: {self.ai.model_name}
- **Cost**: ${cost_info.get('total_cost', 0):.4f} ({token_info.get('total_tokens', 0):,} tokens)

## 📁 Files by Type
{chr(10).join([f'- **{ext if ext != "no_extension" else "other"}**: {", ".join(files)}' for ext, files in {ext: [f for f in files_dict.keys() if Path(f).suffix == ext] for ext in file_types}.items()])}

## 🚀 Context
- **Project Path**: {str(Path(self.project_path).absolute())}
- **Prompt Length**: {len(current_prompt.text)} characters
- **Turn Number**: {turn_number}
- **Total Turns**: {len(self.conversation_history)}

---
*Generated by GPT-Engineer Multi-turn Mode*
"""

                self.observability.set_trace_output(trace_output)
                print(f"🔍 Trace Output Set: {trace_output}")
                self.observability.end_trace(trace_id)

            # Create new prompt for next turn
            current_prompt = Prompt(next_input)

        # Print conversation summary
        self._print_conversation_summary()

        # Note: Session cleanup is handled by main.py, not here
        if self.observability and self.observability.is_enabled():
            print(
                "Multi-turn conversation completed - session cleanup handled by main.py"
            )

    def _print_conversation_summary(self) -> None:
        """Print a summary of the conversation."""
        print("\n" + "=" * 50)
        print(colored("📊 Conversation Summary", "cyan"))
        print("=" * 50)

        for i, turn in enumerate(self.conversation_history, 1):
            print(f"Turn {i}:")
            print(f"  Mode: {turn['mode']}")
            print(f"  Files: {turn['files_count']}")
            print(
                f"  Request: {turn['prompt'][:100]}{'...' if len(turn['prompt']) > 100 else ''}"
            )
            print()

        print(f"Total turns: {len(self.conversation_history)}")
        print("=" * 50)


def run_multi_turn_mode(
    ai: AI,
    project_path: str,
    preprompts_holder: PrepromptsHolder,
    initial_prompt: Prompt,
) -> None:
    """
    Run the multi-turn mode with the given parameters.

    Parameters
    ----------
    ai : AI
        The AI instance to use.
    project_path : str
        The project path.
    preprompts_holder : PrepromptsHolder
        The preprompts holder.
    initial_prompt : Prompt
        The initial prompt to start the conversation.
    """
    engine = MultiTurnEngine(ai, project_path, preprompts_holder)
    engine.run_conversation(initial_prompt)
