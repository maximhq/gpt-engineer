"""
GPT Engineer Web UI - Flask Application

This module provides a web interface for the GPT Engineer multi-turn agent.
"""

import io
import json
import logging
import os
import sys
import time
import uuid

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, List

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

# Type hints for function signatures
from gpt_engineer.core.ai import AI
from gpt_engineer.core.preprompts_holder import PrepromptsHolder
from gpt_engineer.core.prompt import Prompt

# Import database-backed state management
from .state_manager import cleanup_state_manager, get_state_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database-backed state manager
state_manager = get_state_manager()

# Initialize observability globally
observability = None
try:
    from gpt_engineer.core.maxim_observability import init_observability

    observability = init_observability(enabled=True)
    logger.info("Observability initialized for web UI")
except ImportError:
    logger.warning("Observability not available")
    observability = None


# Note: Trace and Session classes have been replaced by database-backed versions
# in state_manager.py - DatabaseBackedTrace and DatabaseBackedSession


class WebMultiTurnEngine:
    """
    Web UI wrapper for the MultiTurnEngine that uses database-backed state management.
    """

    def __init__(
        self, ai, project_path: str, preprompts_holder, session_id: str = None
    ):
        """
        Initialize the WebMultiTurnEngine.

        Parameters
        ----------
        ai : AI
            The AI instance to use.
        project_path : str
            The project path.
        preprompts_holder : PrepromptsHolder
            The preprompts holder.
        session_id : str, optional
            The session ID.
        """
        self.ai = ai
        self.project_path = project_path
        self.preprompts_holder = preprompts_holder
        self.session_id = session_id
        self.turn_number = 0  # Track conversation turns

        # Initialize observability for this session
        if observability and observability.is_enabled():
            try:
                # Get system information for comprehensive session metadata
                import platform
                import sys

                system_info = {
                    "os": platform.system(),
                    "os_version": platform.version(),
                    "architecture": platform.machine(),
                    "python_version": sys.version,
                }

                session_tags = {
                    "operation": "web_ui_session",
                    "session_id": session_id,
                    "project_path": str(Path(project_path).absolute()),
                    "model": ai.model_name,
                    "temperature": str(ai.temperature),
                    "python_version": sys.version.split()[0],
                    "gpt_engineer_invocation": "web_ui",
                }

                session_metadata = {
                    "web_ui_session": {
                        "session_id": session_id,
                        "project_path": project_path,
                        "model": ai.model_name,
                        "temperature": ai.temperature,
                    },
                    "system_info": system_info,
                }

                observability.start_session(
                    session_id=session_id, tags=session_tags, metadata=session_metadata
                )

                self.observability = observability
                self.add_stream_output(f"Started Maxim session: {session_id}")

            except Exception as e:
                logger.warning(
                    f"Failed to initialize observability for session {session_id}: {e}"
                )

        # Initialize the real MultiTurnEngine
        from gpt_engineer.applications.cli.multi_turn import MultiTurnEngine

        self.engine = MultiTurnEngine(ai, project_path, preprompts_holder, web_ui=True)

    def add_stream_output(self, message: str, trace_id: str = None):
        """Add output to the stream for this session."""
        if self.session_id:
            state_manager.add_stream_output(self.session_id, message, trace_id)

    def clear_stream_output(self):
        """Clear stream output for this session."""
        if self.session_id:
            state_manager.clear_stream_output(self.session_id)
            logger.info(f"Cleared stream output for session: {self.session_id}")

    @contextmanager
    def capture_output(self, trace_id: str = None):
        """Context manager to capture stdout and stderr without interfering with observability."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # Create a custom stream that writes to both original and our buffer
        class DualStream:
            def __init__(self, original_stream, buffer_stream, is_error=False):
                self.original_stream = original_stream
                self.buffer_stream = buffer_stream
                self.is_error = is_error

            def write(self, text):
                # Write to original stream first (for observability)
                self.original_stream.write(text)
                self.original_stream.flush()

                # Also write to our buffer
                self.buffer_stream.write(text)

            def flush(self):
                self.original_stream.flush()
                self.buffer_stream.flush()

            def __getattr__(self, attr):
                return getattr(self.original_stream, attr)

        # Create string buffers
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Create dual streams that write to both original and our buffer
            dual_stdout = DualStream(old_stdout, stdout_buffer)
            dual_stderr = DualStream(old_stderr, stderr_buffer, is_error=True)

            # Redirect stdout and stderr
            sys.stdout = dual_stdout
            sys.stderr = dual_stderr
            yield
        finally:
            # Restore original streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Get captured output
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()

            # Process the captured output
            if stdout_content:
                lines = stdout_content.strip().split("\n")
                execution_confirmation_detected = False
                command_to_execute = None

                for i, line in enumerate(lines):
                    if line.strip() and not self._should_filter_line(line.strip()):
                        # Check if this is the execution confirmation prompt
                        if "Do you want to execute this code? (Y/n)" in line:
                            execution_confirmation_detected = True
                            # Look for the command in the next few lines
                            for j in range(i + 1, min(i + 10, len(lines))):
                                if lines[j].strip() and not self._should_filter_line(
                                    lines[j].strip()
                                ):
                                    if not lines[j].startswith(
                                        "Do you want to execute this code?"
                                    ):
                                        command_to_execute = lines[j]
                                        break
                            break
                        else:
                            self.add_stream_output(line.strip(), trace_id)

                # If execution confirmation was detected, handle it
                if execution_confirmation_detected and command_to_execute:
                    self.add_stream_output(
                        "Do you want to execute this code? (Y/n)", trace_id
                    )
                    self.add_stream_output("", trace_id)
                    self.add_stream_output(command_to_execute, trace_id)
                    self.add_stream_output("", trace_id)
                    # Store the command for later execution in database
                    state_manager.save_pending_execution(
                        self.session_id, command_to_execute, True
                    )

            if stderr_content:
                for line in stderr_content.strip().split("\n"):
                    if line.strip() and not self._should_filter_line(line.strip()):
                        self.add_stream_output(f"ERROR: {line.strip()}", trace_id)

    def _should_filter_line(self, line: str) -> bool:
        """Filter out debug noise from output while keeping observability info."""
        # Only filter out debug noise, keep observability success messages
        filter_patterns = [
            "WARNING:werkzeug",
            "INFO:werkzeug",
            "DEBUG:werkzeug",
            "INFO:httpx",
            "DEBUG:httpx",
            "WARNING:gpt_engineer.core.maxim_observability: No current trace",
            "WARNING:gpt_engineer.core.maxim_observability: No current span",
            "WARNING:gpt_engineer.core.maxim_observability: Parent span id",
            "WARNING:gpt_engineer.core.maxim_observability: not found in stack",
        ]

        return any(pattern in line for pattern in filter_patterns)

    def process_prompt(self, prompt_text: str) -> Dict:
        """
        Process a single prompt and return the response.

        Parameters
        ----------
        prompt_text : str
            The user's prompt text.

        Returns
        -------
        Dict
            Response containing mode, files, trace_id, and other information.
        """
        # Clear stream output for new prompt
        self.clear_stream_output()

        # Increment turn number
        self.turn_number += 1

        # Create a new trace for this prompt
        trace = state_manager.create_trace(self.session_id, prompt_text)
        trace_id = trace.trace_id

        # Initialize trace for this prompt
        if self.observability and self.observability.is_enabled():
            try:
                trace_tags = {
                    "operation": "web_ui_prompt",
                    "session_id": self.session_id,
                    "model": self.ai.model_name,
                    "project_path": str(Path(self.project_path).absolute()),
                    "turn_number": str(self.turn_number),
                }

                trace_metadata = {
                    "prompt_preview": prompt_text[:200] + "..."
                    if len(prompt_text) > 200
                    else prompt_text,
                    "prompt_length": len(prompt_text),
                    "turn_number": self.turn_number,
                    "conversation_history_length": len(self.engine.conversation_history)
                    if hasattr(self.engine, "conversation_history")
                    else 0,
                    "current_files_count": len(self.engine.current_files)
                    if hasattr(self.engine, "current_files")
                    else 0,
                }

                self.observability.start_trace(
                    trace_id=trace_id,
                    name=trace_id,
                    tags=trace_tags,
                    metadata=trace_metadata,
                )

                # Set the prompt as trace input
                self.observability.set_trace_input(prompt_text)

            except Exception as e:
                logger.warning(f"Failed to start trace: {e}")

        try:
            # Create prompt object
            prompt = Prompt(prompt_text)

            # Add initial message to stream
            self.add_stream_output(
                f"Processing prompt: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}",
                trace_id,
            )

            # Use the real engine to determine mode and execute with output capture
            with self.capture_output(trace_id):
                mode = self.engine._determine_mode(prompt)
                self.add_stream_output(f"Determined mode: {mode}", trace_id)
                files_dict = self.engine._execute_mode(mode, prompt)

            # Generate AI summary with CLI output
            self.add_stream_output("Generating AI summary...", trace_id)
            cli_output = state_manager.get_trace_stream_output(trace_id) if trace_id else []
            ai_summary = self.generate_ai_summary(prompt_text, files_dict, mode, cli_output)
            self.add_stream_output(f"AI Summary: {ai_summary}", trace_id)

            # Update trace with results including AI summary
            trace.mode = mode
            trace.files = files_dict
            trace.execution_result = {
                "success": True,
                "mode": mode,
                "files_count": len(files_dict),
                "ai_summary": ai_summary,
            }

            # Store generated files in session for later execution
            session = state_manager.get_or_create_session(self.session_id)
            session.current_files = files_dict

            # Also update the engine's current_files
            self.engine.current_files = files_dict

            # Add completion message
            self.add_stream_output(
                f"Completed {mode} mode - generated {len(files_dict)} files", trace_id
            )

            # Store trace info for later ending (after execution completes)
            self.current_trace_id = trace_id
            self.current_trace_mode = mode
            self.current_trace_files = files_dict
            self.current_trace_prompt = prompt_text
            self.current_trace_ai_summary = ai_summary

            # Don't end trace here - wait until after execution completes
            # The trace will be ended in the execute endpoint after execute_entrypoint_next

            return {
                "success": True,
                "mode": mode,
                "files": files_dict,
                "files_count": len(files_dict),
                "message": f"Successfully executed {mode} mode",
                "trace_id": trace_id,
                "ai_summary": ai_summary,
            }

        except Exception as e:
            error_msg = f"Error processing prompt: {e}"
            logger.error(error_msg)
            self.add_stream_output(error_msg, trace_id)

            # Update trace with error
            trace.execution_result = {"success": False, "error": str(e)}

            # End trace with error
            if self.observability and self.observability.is_enabled() and trace_id:
                try:
                    self.observability.log_error(
                        error=e,
                        context={
                            "operation": "web_ui_prompt",
                            "session_id": self.session_id,
                            "prompt_length": len(prompt_text),
                            "turn_number": self.turn_number,
                        },
                        tags={
                            "error_type": "web_ui_prompt_error",
                            "session_id": self.session_id,
                            "turn_number": str(self.turn_number),
                        },
                    )
                    self.observability.end_trace(trace_id)
                except Exception as trace_error:
                    logger.warning(f"Failed to end trace with error: {trace_error}")

            return {
                "success": False,
                "error": str(e),
                "message": error_msg,
                "trace_id": trace_id,
            }

    def has_pending_execution(self) -> bool:
        """Check if there's pending execution for this session."""
        return state_manager.has_pending_execution(self.session_id)

    @property
    def pending_execution(self) -> Optional[Dict]:
        """Get pending execution data."""
        return state_manager.get_pending_execution(self.session_id)

    def generate_ai_summary(self, prompt: str, files_dict, mode: str, cli_output: list = None) -> str:
        """
        Use AI to generate a natural language summary of the generated files.
        
        Parameters
        ----------
        prompt : str
            The original user prompt
        files_dict : FilesDict
            The files generated by the agent
        mode : str
            The mode used for generation
        cli_output : list, optional
            The CLI output messages from the generation process
            
        Returns
        -------
        str
            AI-generated summary of the work completed
        """
        try:
            # Create a summary prompt for the AI
            files_list = list(files_dict.keys())
            
            # Get a preview of file contents for better context
            files_preview = []
            for filename, content in files_dict.items():
                # Get first few meaningful lines (skip empty lines and comments)
                lines = content.split('\n')
                meaningful_lines = []
                for line in lines[:10]:  # Look at first 10 lines
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#') and not stripped.startswith('//'):
                        meaningful_lines.append(line)
                    if len(meaningful_lines) >= 3:  # Get 3 meaningful lines
                        break
                
                if meaningful_lines:
                    files_preview.append(f"{filename}:\n{chr(10).join(meaningful_lines[:3])}")
                else:
                    files_preview.append(f"{filename}: (configuration/data file)")
            
            # Prepare CLI output summary
            cli_output_summary = ""
            if cli_output and len(cli_output) > 0:
                # Filter out noise and get meaningful CLI messages
                meaningful_output = []
                for msg in cli_output:
                    msg_clean = msg.strip()
                    if (msg_clean and 
                        not msg_clean.startswith("Processing prompt:") and
                        not msg_clean.startswith("Generating AI summary") and
                        not msg_clean.startswith("AI Summary:") and
                        "Determined mode:" not in msg_clean):
                        meaningful_output.append(msg_clean)
                
                if meaningful_output:
                    cli_output_summary = "\n".join(meaningful_output[-10:])  # Last 10 meaningful messages

            summary_prompt = f"""Based on the user's request: "{prompt}"

The coding agent just completed work in {mode} mode and {mode} {len(files_dict)} files:
{chr(10).join(files_list)}

Here's a preview of the key files:
{chr(10).join(files_preview)}

CLI Output from the process:
{cli_output_summary if cli_output_summary else "Standard generation completed successfully."}

Please provide a helpful 2-3 sentence summary explaining:
1. What was built/created/debugged for the user
2. Summarize the output: {cli_output_summary if cli_output_summary else "Files generated successfully"}.
3. Next steps or how to use it (if applicable)

Be concise and try not to make ideal conversation points"""

            # Use the existing AI instance to generate the summary
            from gpt_engineer.core.prompt import Prompt
            summary_prompt_obj = Prompt(summary_prompt)
            
            # Use a simplified approach - direct AI call for summary
            messages = self.ai.start(
                system="You are a helpful coding assistant. Provide clear, concise summaries of code generation results. Focus on what was built and how the user can use it.",
                user=summary_prompt_obj.to_langchain_content(),
                step_name="generate_summary"
            )
            
            summary = messages[-1].content.strip()
            logger.info(f"Generated AI summary: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate AI summary: {e}")
            # Fallback to basic summary
            file_types = set()
            for filename in files_dict.keys():
                if '.' in filename:
                    ext = filename.split('.')[-1]
                    file_types.add(ext)
            
            types_str = ", ".join(sorted(file_types)) if file_types else "various"
            return f"Created {len(files_dict)} files ({types_str}) in {mode} mode. The generated code is ready for review and use."

    def format_trace_output(self, mode: str, files_dict, ai_summary: str, prompt_text: str) -> str:
        """
        Format trace output similar to formatAgentResponse.
        Includes full file contents for each generated file to support evaluators.

        Parameters
        ----------
        mode : str
            The execution mode
        files_dict : FilesDict
            The generated files
        ai_summary : str
            The AI-generated summary
        prompt_text : str
            The original prompt text

        Returns
        -------
        str
            Formatted trace output
        """
        output = ''

        if files_dict and len(files_dict) > 0:
            for filename, content in files_dict.items():
                ext = filename.rsplit('.', 1)[-1] if '.' in filename else 'txt'
                output += f"=== {filename} ===\n"
                output += f"```{ext}\n{content}\n```\n\n"

        return output

    def cleanup(self):
        """Clean up observability session when done."""
        if self.observability and self.observability.is_enabled():
            try:
                # End any open trace before ending session
                if hasattr(self, "current_trace_id") and self.current_trace_id:
                    try:
                        self.observability.end_trace(self.current_trace_id)
                    except Exception as trace_err:
                        logger.warning(f"Failed to end trace during cleanup: {trace_err}")
                        try:
                            self.observability.end_trace(self.current_trace_id)
                        except Exception:
                            pass
                    finally:
                        self.current_trace_id = None
                        self.current_trace_mode = None
                        self.current_trace_files = None
                        self.current_trace_prompt = None
                        self.current_trace_ai_summary = None

                # Flush data before ending session
                self.observability.flush_data()

                # End session with validation
                if self.session_id:
                    if hasattr(self.observability, "validate_session_state"):
                        if not self.observability.validate_session_state():
                            logger.warning(
                                "Session state validation failed during cleanup, attempting recovery"
                            )
                            if hasattr(self.observability, "recover_session_state"):
                                self.observability.recover_session_state()

                    self.observability.end_session()

                logger.info(f"Cleaned up observability session: {self.session_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup observability session {self.session_id}: {e}"
                )
                # Try to force cleanup even if there are errors
                try:
                    if hasattr(self.observability, "flush_data"):
                        self.observability.flush_data()
                    if hasattr(self.observability, "cleanup"):
                        self.observability.cleanup()
                except Exception as force_cleanup_error:
                    logger.error(f"Force cleanup also failed: {force_cleanup_error}")


def create_ai_instance() -> AI:
    """
    Create an AI instance with proper configuration.

    Returns
    -------
    AI
        Configured AI instance.
    """
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")
    temperature = float(os.environ.get("TEMPERATURE", "0.1"))
    azure_endpoint = os.environ.get("AZURE_ENDPOINT", "")

    return AI(
        model_name=model_name,
        temperature=temperature,
        azure_endpoint=azure_endpoint if azure_endpoint else None,
    )


def create_preprompts_holder(project_path: str) -> PrepromptsHolder:
    """
    Create a preprompts holder for the project.

    Parameters
    ----------
    project_path : str
        The project path.

    Returns
    -------
    PrepromptsHolder
        Configured preprompts holder.
    """
    from gpt_engineer.core.default.paths import PREPROMPTS_PATH

    return PrepromptsHolder(PREPROMPTS_PATH)


def recreate_engine_with_files(
    session_id: str, 
    project_path: str, 
    conversation_history: Optional[List[Dict]] = None,
    file_data: Optional[Dict[str, str]] = None
) -> "WebMultiTurnEngine":
    """
    Recreate an engine and load current files from the session or use provided data.
    
    Parameters
    ----------
    session_id : str
        The session ID.
    project_path : str
        The project path.
    conversation_history : Optional[List[Dict]], optional
        Conversation history to use instead of loading files. If provided,
        file loading will be skipped and this history will be set on the engine.
        Format: [{"prompt": str, "mode": str, "files_count": int}, ...]
    file_data : Optional[Dict[str, str]], optional
        File data to use directly. Keys are file paths, values are file contents.
        If provided along with conversation_history, will be used instead of loading from session.

    Returns
    -------
    WebMultiTurnEngine
        The recreated engine with current files loaded or provided data set.
    """
    ai = create_ai_instance()
    preprompts_holder = create_preprompts_holder(project_path)
    engine = WebMultiTurnEngine(ai, project_path, preprompts_holder, session_id)

    if conversation_history or file_data:
        # Use provided data instead of loading from session
        from gpt_engineer.core.files_dict import FilesDict
        
        if file_data:
            # Use provided file_data directly
            engine.engine.current_files = FilesDict(file_data)
            logger.info(
                f"Set {len(file_data)} files from provided file_data on engine for session {session_id}"
            )
        else:
            # No file_data provided, start with empty files
            engine.engine.current_files = FilesDict()
            logger.info(
                f"Starting with empty files for session {session_id} (conversation_history provided)"
            )
        
        if conversation_history:
            engine.engine.conversation_history = conversation_history
            logger.info(
                f"Set conversation history ({len(conversation_history)} turns) on engine for session {session_id}"
            )
    else:
        # Normal flow: Load current files from session database
        current_files = state_manager.session_file_repo.get_session_files(session_id)
        if current_files:
            engine.engine.current_files = current_files
            logger.info(
                f"Loaded {len(current_files)} files into recreated engine for session {session_id}"
            )

    return engine


# Create Flask app
app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    """Serve the main web interface."""
    return render_template("index.html")


@app.route("/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return jsonify({"status": "healthy", "service": "gpt-engineer-web"}), 200


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Handle chat requests from the web interface.

    Expected JSON payload:
    {
        "prompt": "user prompt text",
        "session_id": "optional session id"
    }

    Returns
    -------
    JSON response with conversation results.
    """
    try:
        data = request.get_json()
        prompt_text = data.get("prompt", "").strip()
        session_id = data.get("session_id")

        if not prompt_text:
            return jsonify({"success": False, "error": "No prompt provided"}), 400

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info(f"Processing prompt for session: {session_id}")

        # Get or create session using state manager
        active_session = state_manager.get_active_session()
        if not active_session or active_session.session_id != session_id:
            # Create new session
            logger.info(f"Creating new session: {session_id}")
            project_path = f"projects/web_session_{session_id}"

            # Create session in database
            active_session = state_manager.get_or_create_session(
                session_id, project_path
            )
            state_manager.set_active_session(session_id)

            # Create engine and store reference
            active_session.engine = recreate_engine_with_files(session_id, project_path)

            logger.info(f"Session created successfully: {active_session.session_id}")
        else:
            # Reuse existing session for multi-turn conversation
            logger.info(f"Reusing existing session: {session_id}")

            # Ensure engine exists for the session (it's transient and not persisted in DB)
            if not hasattr(active_session, "engine") or not active_session.engine:
                logger.info(f"Recreating engine for existing session: {session_id}")
                active_session.engine = recreate_engine_with_files(
                    session_id, active_session.project_path
                )

            # Do NOT clear stream output here; if execution is pending we return 409
            # and must preserve the existing CLI output and execution prompt dialog.
            # Update last activity
            active_session.update_last_activity()

        # Ensure we have an engine
        if not hasattr(active_session, "engine") or not active_session.engine:
            logger.error("Session engine is not properly initialized")
            return (
                jsonify({"success": False, "error": "Session engine not initialized"}),
                500,
            )

        engine = active_session.engine

        # Check if there's pending execution and block new input
        if engine.has_pending_execution():
            logger.info("Blocking new input - pending execution exists")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Cannot process new input while execution is pending. Please respond to the execution prompt first.",
                        "blocked": True,
                        "pending_execution": True,
                    }
                ),
                409,
            )  # Conflict status code

        # No pending execution; safe to clear stream inside process_prompt
        result = engine.process_prompt(prompt_text)

        trace_output = engine.format_trace_output(
            mode=engine.current_trace_mode,
            files_dict=engine.current_trace_files,
            ai_summary=getattr(engine, 'current_trace_ai_summary', None),
            prompt_text=prompt_text
        )

        result["trace_output"] = trace_output

        trace_id = result.get("trace_id")

        if trace_id:
            cli_output = state_manager.get_trace_stream_output(trace_id)
            result["cli_output"] = cli_output
        else:
            result["cli_output"] = []

        # Log turn completion event
        if observability and observability.is_enabled():
            try:
                from uuid import uuid4


                observability.set_trace_output(trace_output)
                observability.attach_files_to_trace(engine.current_trace_files)
                observability.log_event(
                    event_id=str(uuid4()),
                    event_type="turn_completed",
                    metadata={
                        "turn_number": engine.turn_number,
                        "mode": result.get("mode"),
                        "files_generated": result.get("files_count", 0),
                        "prompt_length": len(prompt_text),
                        "session_id": session_id,
                    },
                    tags={
                        "operation": "turn_completion",
                        "mode": result.get("mode"),
                        "turn_number": str(engine.turn_number),
                        "session_id": session_id,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log turn completion event: {e}")

        # Trace is already created and managed by the database
        logger.info(f"Trace processed successfully: {result.get('trace_id')}")

        # End trace for non-generate modes (improve, chat, misc) - no execution will follow
        if trace_id:
            try:
                if observability and observability.is_enabled():
                    observability.end_trace(trace_id)
            except Exception as e:
                logger.warning(f"Failed to end trace for non-generate mode: {e}")

        # Add session ID to response
        result["session_id"] = session_id

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/single-turn", methods=["POST"])
def single_turn():
    """
    Handle single-turn chat requests with provided file_data and conversation_history.
    This endpoint is for test mode and skips file loading from session.

    Expected JSON payload:
    {
        "prompt": "user prompt text",
        "conversation_history": "optional list of conversation history entries",
        "file_data": "optional dict of file paths to file contents"
    }

    Returns
    -------
    JSON response with conversation results.
    """
    try:
        data = request.get_json()
        prompt_text = data.get("prompt", "").strip()
        conversation_history = data.get("conversation_history")
        file_data = data.get("file_data")

        if not prompt_text:
            return jsonify({"success": False, "error": "No prompt provided"}), 400

        # Generate a temporary session ID for this single turn
        session_id = str(uuid.uuid4())
        project_path = f"projects/web_session_{session_id}"

        logger.info(f"Processing single-turn prompt for session: {session_id}")

        # Create a temporary session (won't be persisted)
        active_session = state_manager.get_or_create_session(
            session_id, project_path
        )
        state_manager.set_active_session(session_id)

        # Create engine with provided conversation_history and file_data
        active_session.engine = recreate_engine_with_files(
            session_id, 
            project_path, 
            conversation_history=conversation_history,
            file_data=file_data
        )

        engine = active_session.engine

        # Process the prompt
        result = engine.process_prompt(prompt_text)

        trace_output = engine.format_trace_output(
            mode=engine.current_trace_mode,
            files_dict=engine.current_trace_files,
            ai_summary=getattr(engine, 'current_trace_ai_summary', None),
            prompt_text=prompt_text
        )

        result["trace_output"] = trace_output

        # Get CLI output (stream messages) for this trace to include in response
        trace_id = result.get("trace_id")
        if trace_id:
            cli_output = state_manager.get_trace_stream_output(trace_id)
            result["cli_output"] = cli_output
        else:
            result["cli_output"] = []

        # Log turn completion event
        if observability and observability.is_enabled():
            try:
                from uuid import uuid4

                observability.set_trace_output(trace_output)
                observability.attach_files_to_trace(engine.current_trace_files)
                observability.log_event(
                    event_id=str(uuid4()),
                    event_type="turn_completed",
                    metadata={
                        "turn_number": engine.turn_number,
                        "mode": result.get("mode"),
                        "files_generated": result.get("files_count", 0),
                        "prompt_length": len(prompt_text),
                        "session_id": session_id,
                        "single_turn": True,
                    },
                    tags={
                        "operation": "turn_completion",
                        "mode": result.get("mode"),
                        "turn_number": str(engine.turn_number),
                        "session_id": session_id,
                        "single_turn": "true",
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log turn completion event: {e}")

        # Trace is already created and managed by the database
        logger.info(f"Single-turn trace processed successfully: {result.get('trace_id')}")

        # Single-turn has no execution - end trace immediately
        if trace_id:
            try:
                if observability and observability.is_enabled():
                    observability.end_trace(trace_id)
            except Exception as e:
                logger.warning(f"Failed to end single-turn trace: {e}")

        # Add session ID to response
        result["session_id"] = session_id

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in single-turn endpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/execute", methods=["POST"])
def execute_code():
    """
    Handle code execution confirmation from the web interface.
    """
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        confirm = data.get("confirm")
        user_response = data.get("user_response")

        if not session_id:
            return jsonify({"success": False, "error": "Session ID is required"}), 400

        # Handle both confirm (boolean) and user_response (string) fields
        if confirm is not None:
            # Convert boolean confirm to string user_response
            user_response = "y" if confirm else "n"
        elif user_response is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Either confirm or user_response is required",
                    }
                ),
                400,
            )

        logger.info(
            f"Processing execution confirmation for session: {session_id}, user_response: {user_response}"
        )

        # Get session from state manager
        active_session = state_manager.get_active_session()
        if not active_session or active_session.session_id != session_id:
            logger.error(f"Session {session_id} not found")
            return jsonify({"success": False, "error": "Session not found"}), 404

        # Check if user declined execution
        user_confirmed = user_response.lower() in ["", "y", "yes"]
        if not user_confirmed:
            logger.info("User declined execution")

            # Ensure we have an engine for stream output - recreate if missing
            if not hasattr(active_session, "engine") or not active_session.engine:
                logger.info(
                    f"Recreating engine for decline message in session: {session_id}"
                )
                active_session.engine = recreate_engine_with_files(
                    session_id, active_session.project_path
                )

            active_session.engine.add_stream_output("Ok, not executing the code.")
            # Clear pending execution when user declines
            state_manager.clear_pending_execution(session_id)

            # End trace when user declines - trace is complete even without execution
            if (
                hasattr(active_session.engine, "current_trace_id")
                and active_session.engine.current_trace_id
            ):
                try:
                    if observability and observability.is_enabled():
                        trace_output = active_session.engine.format_trace_output(
                            mode=active_session.engine.current_trace_mode,
                            files_dict=active_session.engine.current_trace_files,
                            ai_summary=getattr(
                                active_session.engine, "current_trace_ai_summary", None
                            ),
                            prompt_text=active_session.engine.current_trace_prompt,
                        )
                        observability.set_trace_output(trace_output)
                        observability.attach_files_to_trace(active_session.engine.current_trace_files)
                        observability.end_trace(active_session.engine.current_trace_id)
                except Exception as e:
                    logger.warning(f"Failed to end trace after user declined: {e}")

            # Do NOT clear the stream here; keep the previous CLI output and dialog visible

            return jsonify(
                {"success": True, "message": "Execution declined", "executed": False}
            )

        # User confirmed execution
        logger.info("User confirmed execution")

        # Ensure we have an engine before proceeding - recreate if missing
        if not hasattr(active_session, "engine") or not active_session.engine:
            logger.info(f"Recreating engine for execution in session: {session_id}")
            active_session.engine = recreate_engine_with_files(
                session_id, active_session.project_path
            )

        active_session.engine.add_stream_output("Executing the code...")
        active_session.engine.add_stream_output("")
        active_session.engine.add_stream_output(
            "Note: If it does not work as expected, consider running the code in another way than above."
        )
        active_session.engine.add_stream_output("")
        active_session.engine.add_stream_output(
            "You can press ctrl+c *once* to stop the execution."
        )
        active_session.engine.add_stream_output("")

        # Log user confirmation event
        if observability and observability.is_enabled():
            try:
                from uuid import uuid4

                observability.log_event(
                    event_id=str(uuid4()),
                    event_type="user_execution_confirmation",
                    metadata={
                        "user_response": user_response,
                        "session_id": session_id,
                        "confirmed": user_confirmed,
                    },
                    tags={
                        "operation": "user_decision",
                        "result": "confirmed" if user_confirmed else "declined",
                        "session_id": session_id,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to log user confirmation event: {e}")

        # Execute the pending command
        pending_execution = state_manager.get_pending_execution(session_id)
        if pending_execution:
            pending_execution["command"]

            try:
                # Import the execute_entrypoint_next function
                from gpt_engineer.core.default.steps import execute_entrypoint_next

                # Execute using the execute_entrypoint_next function with user response and timeout
                execute_entrypoint_next(
                    execution_env=active_session.engine.engine.execution_env,
                    files_dict=active_session.engine.engine.current_files,
                    user_response=user_response,
                    timeout=30,  # 30 second timeout for web UI execution
                )

                # Clear pending execution
                state_manager.clear_pending_execution(session_id)

                # Clear stream output after successful execution
                active_session.engine.clear_stream_output()

                # End the current trace with proper output (similar to CLI multi-turn)
                if (
                    hasattr(active_session.engine, "current_trace_id")
                    and active_session.engine.current_trace_id
                ):
                    try:
                        if observability and observability.is_enabled():
                            # Calculate comprehensive metrics for trace output
                            files_dict = active_session.engine.current_trace_files
                            total_size = sum(
                                len(content) for content in files_dict.values()
                            )
                            file_types = list(
                                set(Path(fname).suffix for fname in files_dict.keys())
                            )

                            # Cost and token information
                            cost_info = {}
                            token_info = {}
                            if (
                                active_session.engine.ai.token_usage_log.is_openai_model()
                            ):
                                cost_info = {
                                    "total_cost": active_session.engine.ai.token_usage_log.usage_cost(),
                                    "currency": "USD",
                                    "model": active_session.engine.ai.model_name,
                                }
                                token_info = {
                                    "total_tokens": active_session.engine.ai.token_usage_log.total_tokens(),
                                    "cost_per_token": active_session.engine.ai.token_usage_log.usage_cost()
                                    / max(
                                        active_session.engine.ai.token_usage_log.total_tokens(),
                                        1,
                                    ),
                                }

                            # Format trace output similar to formatAgentResponse
                            trace_output = active_session.engine.format_trace_output(
                                mode=active_session.engine.current_trace_mode,
                                files_dict=files_dict,
                                ai_summary=getattr(active_session.engine, 'current_trace_ai_summary', None),
                                prompt_text=active_session.engine.current_trace_prompt
                            )

                            # Set trace output and end trace AFTER execution completes
                            observability.set_trace_output(trace_output)
                            observability.attach_files_to_trace(files_dict)
                            observability.end_trace(
                                active_session.engine.current_trace_id
                            )

                            # Clear the current trace info
                            active_session.engine.current_trace_id = None
                            active_session.engine.current_trace_mode = None
                            active_session.engine.current_trace_files = None
                            active_session.engine.current_trace_prompt = None
                            active_session.engine.current_trace_ai_summary = None

                    except Exception as e:
                        logger.warning(f"Failed to end trace after execution: {e}")
                        # Try to end trace without output if setting output fails
                        try:
                            if observability and observability.is_enabled():
                                observability.end_trace(
                                    active_session.engine.current_trace_id
                                )
                        except Exception as end_error:
                            logger.error(
                                f"Failed to end trace after error: {end_error}"
                            )

                logger.info("Code execution completed successfully")
                return jsonify(
                    {
                        "success": True,
                        "message": "Code executed successfully",
                        "executed": True,
                    }
                )
            except Exception as e:
                error_msg = f"Error executing code: {e}"
                logger.error(error_msg)
                if hasattr(active_session, "engine") and active_session.engine:
                    active_session.engine.add_stream_output(error_msg)
                    # Do NOT clear the stream on error; keep dialog/context visible
                # Clear pending execution on error
                state_manager.clear_pending_execution(session_id)

                # End the current trace with error output
                if (
                    hasattr(active_session, "engine")
                    and active_session.engine
                    and hasattr(active_session.engine, "current_trace_id")
                    and active_session.engine.current_trace_id
                ):
                    try:
                        if observability and observability.is_enabled():
                            # Create error summary using the same format
                            error_summary = f"Mode: {active_session.engine.current_trace_mode if active_session.engine.current_trace_mode else 'Unknown'}\n"
                            error_summary += f"Files generated: 0\n\n"
                            error_summary += f"❌ Execution Error:\n{str(e)}\n"

                            # Set error trace output and end trace
                            observability.set_trace_output(error_summary)
                            observability.end_trace(
                                active_session.engine.current_trace_id
                            )

                            # Clear the current trace info
                            active_session.engine.current_trace_id = None
                            active_session.engine.current_trace_mode = None
                            active_session.engine.current_trace_files = None
                            active_session.engine.current_trace_prompt = None
                            active_session.engine.current_trace_ai_summary = None

                    except Exception as trace_error:
                        logger.warning(
                            f"Failed to end trace after execution error: {trace_error}"
                        )
                        # Try to end trace without output if setting output fails
                        try:
                            if observability and observability.is_enabled():
                                observability.end_trace(
                                    active_session.engine.current_trace_id
                                )
                        except Exception as end_error:
                            logger.error(
                                f"Failed to end trace after error: {end_error}"
                            )

                return jsonify(
                    {
                        "success": False,
                        "error": str(e),
                        "message": error_msg,
                        "executed": False,
                    }
                )
        else:
            return jsonify(
                {
                    "success": False,
                    "error": "No pending execution found",
                    "executed": False,
                }
            )

    except Exception as e:
        logger.error(f"Error in execute endpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/session/<session_id>/status", methods=["GET"])
def get_session_status(session_id: str):
    """
    Get the current status of a session.

    Parameters
    ----------
    session_id : str
        The session ID to check status for.

    Returns
    -------
    JSON response with session status.
    """
    try:
        # Check if session exists in database
        session_data = state_manager.session_repo.get_session(session_id)
        if not session_data:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Session not found",
                        "session_exists": False,
                    }
                ),
                404,
            )

        has_pending_execution = state_manager.has_pending_execution(session_id)
        pending_execution = (
            state_manager.get_pending_execution(session_id)
            if has_pending_execution
            else None
        )

        return jsonify(
            {
                "success": True,
                "session_exists": True,
                "has_pending_execution": has_pending_execution,
                "pending_execution": pending_execution,
            }
        )

    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stream/<session_id>")
def stream_output(session_id: str):
    """
    Stream real-time output for a session using Server-Sent Events.

    Parameters
    ----------
    session_id : str
        The session ID to stream output for.

    Returns
    -------
    Response
        Server-Sent Events stream.
    """

    def generate():
        """Generate SSE events."""
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Stream connected'})}\n\n"

        # Get existing messages from database
        current_messages = state_manager.get_stream_output(session_id)
        for message in current_messages:
            # Handle special stream cleared message
            if message == "[STREAM_CLEARED]":
                yield f"data: {json.dumps({'type': 'stream_cleared', 'message': 'Stream cleared'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'message', 'message': message})}\n\n"

        # Keep track of last message index
        last_index = len(current_messages)

        # Keep connection alive and check for new messages
        try:
            while True:
                time.sleep(0.5)  # Check every 500ms

                # Check for new messages from database
                current_messages = state_manager.get_stream_output(session_id)
                if len(current_messages) > last_index:
                    # Send new messages
                    for i in range(last_index, len(current_messages)):
                        message = current_messages[i]
                        # Handle special stream cleared message
                        if message == "[STREAM_CLEARED]":
                            yield f"data: {json.dumps({'type': 'stream_cleared', 'message': 'Stream cleared'})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'message', 'message': message})}\n\n"
                    last_index = len(current_messages)

                # Send ping to keep connection alive
                yield f"data: {json.dumps({'type': 'ping', 'timestamp': time.time()})}\n\n"

        except GeneratorExit:
            logger.info(f"Stream connection closed for session {session_id}")
        except Exception as e:
            logger.error(f"Stream error for session {session_id}: {e}")
            # Try to send error message
            try:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Stream error occurred'})}\n\n"
            except:
                pass

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/stream/<session_id>/history")
def get_stream_history(session_id: str):
    """
    Get the stream history for a session.

    Parameters
    ----------
    session_id : str
        The session ID.

    Returns
    -------
    JSON response with stream history.
    """
    try:
        messages = state_manager.get_stream_output(session_id)
        return jsonify({"success": True, "messages": messages})
    except Exception as e:
        logger.error(f"Error getting stream history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stream/<session_id>/trace/<trace_id>/history")
def get_trace_stream_history(session_id: str, trace_id: str):
    """
    Get the stream history for a specific trace.

    Parameters
    ----------
    session_id : str
        The session ID.
    trace_id : str
        The trace ID.

    Returns
    -------
    JSON response with trace stream history.
    """
    try:
        # Verify session exists
        session_data = state_manager.session_repo.get_session(session_id)
        if not session_data:
            return jsonify({"success": False, "error": "Session not found"}), 404

        # Get trace stream history
        messages = state_manager.get_trace_stream_output(trace_id)
        return jsonify({"success": True, "messages": messages})

    except Exception as e:
        logger.error(f"Error getting trace stream history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/session/<session_id>/history", methods=["GET"])
def get_conversation_history(session_id: str):
    """
    Get the conversation history for a session.

    Parameters
    ----------
    session_id : str
        The session ID.

    Returns
    -------
    JSON response with conversation history.
    """
    try:
        # Get session from database
        session = state_manager.get_or_create_session(session_id)
        return jsonify({"success": True, "session": session.to_dict()})
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/session/<session_id>/files", methods=["GET"])
def get_session_files(session_id: str):
    """
    Get the current files for a session.

    Parameters
    ----------
    session_id : str
        The session ID.

    Returns
    -------
    JSON response with session files.
    """
    try:
        # Get session files from database
        files = state_manager.session_file_repo.get_session_files(session_id)
        return jsonify({"success": True, "files": files})
    except Exception as e:
        logger.error(f"Error getting session files: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """
    List all active sessions.

    Returns
    -------
    JSON response with session list.
    """
    try:
        # Get all sessions from database
        sessions = state_manager.list_sessions()
        # Add trace count for each session
        for session in sessions:
            traces = state_manager.trace_repo.get_session_traces(session["session_id"])
            session["trace_count"] = len(traces)

        return jsonify({"success": True, "sessions": sessions})
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/session/end", methods=["POST"])
def end_session():
    """
    End the observability session for the current/active session.
    Call this when a script or client is done with a session to ensure
    trace.end() and session.end() are properly called.

    Expected JSON payload:
    {
        "session_id": "optional - if provided, only end if it matches active session"
    }

    Returns
    -------
    JSON response indicating success.
    """
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id")

        active_session = state_manager.get_active_session()
        print(f"Active session: {active_session}")
        if not active_session and session_id:
            observability.end_session_sessionId(session_id)
            return jsonify({"success": True, "message": "No active session to end"}), 200

        if session_id and active_session.session_id != session_id:
            return jsonify(
                {"success": False, "error": f"Session {session_id} is not the active session"}
            ), 400

        if hasattr(active_session, "engine") and active_session.engine:
            active_session.engine.cleanup()
            logger.info(f"Ended observability session: {active_session.session_id}")

        return jsonify({"success": True, "message": "Session ended"}), 200
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/session/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """
    Delete a session.

    Parameters
    ----------
    session_id : str
        The session ID.

    Returns
    -------
    JSON response indicating success.
    """
    try:
        # Check if it's the active session and clean up engine
        active_session = state_manager.get_active_session()
        if active_session and active_session.session_id == session_id:
            # Clean up observability session
            if hasattr(active_session, "engine") and active_session.engine:
                active_session.engine.cleanup()

        # Delete session from database (cascades to all related data)
        success = state_manager.delete_session(session_id)

        if success:
            return jsonify({"success": True, "message": "Session deleted"})
        else:
            return jsonify({"success": False, "error": "Session not found"}), 404
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_all_sessions():
    """
    Reset all sessions (clean up when page is refreshed).

    Returns
    -------
    JSON response indicating success.
    """
    try:
        # Clean up active session if it exists
        active_session = state_manager.get_active_session()
        if (
            active_session
            and hasattr(active_session, "engine")
            and active_session.engine
        ):
            active_session.engine.cleanup()

        # Note: We don't actually delete sessions from database on reset
        # as this is just a page refresh. Sessions remain in database
        # for persistence across browser sessions.

        logger.info("All sessions reset")

        return jsonify({"success": True, "message": "All sessions reset"})
    except Exception as e:
        logger.error(f"Error resetting sessions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """
    Submit feedback for a specific response.

    Returns
    -------
    JSON response indicating success.
    """
    global active_session

    try:
        logger.info("🎯 Feedback endpoint called")

        data = request.get_json()
        logger.info(f"📥 Received feedback data: {data}")

        session_id = data.get("session_id")
        feedback_score = data.get("feedback_score")  # 1 for positive, 0 for negative
        trace_id = data.get("trace_id")
        is_update = data.get(
            "is_update", False
        )  # Whether this is updating existing feedback

        logger.info(
            f"🔍 Parsed feedback - Session: {session_id}, Score: {feedback_score}, Trace: {trace_id}, Update: {is_update}"
        )

        if not session_id:
            logger.error("❌ No session_id provided")
            return jsonify({"success": False, "error": "Session ID is required"}), 400

        if not trace_id:
            logger.error("❌ No trace_id provided")
            return jsonify({"success": False, "error": "Trace ID is required"}), 400

        # Verify session exists
        session_data = state_manager.session_repo.get_session(session_id)
        if not session_data:
            logger.error(f"❌ Session {session_id} not found in database")
            return jsonify({"success": False, "error": "Session not found"}), 404

        # Find the trace in database
        trace_data = state_manager.trace_repo.get_trace(trace_id)
        if not trace_data or trace_data["session_id"] != session_id:
            logger.error(f"❌ Trace {trace_id} not found in session {session_id}")
            return jsonify({"success": False, "error": "Trace not found"}), 404

        logger.info(f"✅ Found trace data for {trace_id}")

        # Update trace feedback in database
        state_manager.trace_repo.update_trace_feedback(trace_id, feedback_score)

        # Get active session for observability if available
        active_session = state_manager.get_active_session()
        engine = None

        # Check if we have an active session with an engine
        if (
            active_session
            and active_session.session_id == session_id
            and hasattr(active_session, "engine")
            and active_session.engine
        ):
            engine = active_session.engine
            logger.info("🔧 Engine found in session data")
        else:
            # No active engine - recreate for observability if this is the correct session
            if active_session and active_session.session_id == session_id:
                logger.info("🔄 Recreating engine for observability feedback")
                active_session.engine = recreate_engine_with_files(
                    session_id, active_session.project_path
                )
                engine = active_session.engine
            else:
                # Different session or no active session - create temporary session for feedback
                logger.info(
                    f"🔄 Creating temporary session for feedback observability in session {session_id}"
                )
                temp_session = state_manager.get_or_create_session(
                    session_id, session_data["project_path"]
                )
                temp_session.engine = recreate_engine_with_files(
                    session_id, session_data["project_path"]
                )
                engine = temp_session.engine

        if (
            engine
            and hasattr(engine, "observability")
            and engine.observability
            and engine.observability.is_enabled()
        ):
            logger.info("📊 Observability is enabled, adding feedback")
            try:
                # Validate session state before adding feedback
                if hasattr(engine.observability, "validate_session_state"):
                    if not engine.observability.validate_session_state():
                        logger.warning(
                            "Session state validation failed, attempting recovery"
                        )
                        if hasattr(engine.observability, "recover_session_state"):
                            if not engine.observability.recover_session_state():
                                logger.error(
                                    "Session recovery failed, skipping feedback"
                                )
                                return (
                                    jsonify(
                                        {
                                            "success": False,
                                            "error": "Session state validation failed",
                                        }
                                    ),
                                    500,
                                )
                        else:
                            logger.error(
                                "Session validation failed and no recovery method available"
                            )
                            return (
                                jsonify(
                                    {
                                        "success": False,
                                        "error": "Session state validation failed",
                                    }
                                ),
                                500,
                            )

                # Add feedback to trace
                engine.observability.add_feedback(feedback_score, trace_id=trace_id)

                # # Also add session-level feedback for comprehensive tracking
                # if hasattr(engine.observability, 'add_session_feedback'):
                #     # Create a simple review object for session feedback
                #     class SimpleReview:
                #         def __init__(self, score):
                #             self.perfect = score == 1
                #             self.works = score == 1
                #             self.ran = score == 1
                #             self.raw = f"Web UI feedback score: {score}"
                #             self.comments = f"Web UI user feedback: {score}"

                #     review = SimpleReview(feedback_score)

                #     # Validate session state before adding session feedback
                #     if hasattr(engine.observability, 'validate_session_state'):
                #         if not engine.observability.validate_session_state():
                #             logger.warning("Session state validation failed for session feedback, attempting recovery")
                #             if hasattr(engine.observability, 'recover_session_state'):
                #                 if not engine.observability.recover_session_state():
                #                     logger.error("Session recovery failed for session feedback")
                #                     # Continue without session feedback rather than failing completely
                #                 else:
                #                     engine.observability.safe_add_session_feedback(review)
                #             else:
                #                 logger.error("Session validation failed and no recovery method available for session feedback")
                #                 # Continue without session feedback rather than failing completely
                #         else:
                #             engine.observability.safe_add_session_feedback(review)
                #     else:
                #         # Fallback to direct call if validation not available
                #         engine.observability.add_session_feedback(review)

                # logger.info(
                #     f"✅ Feedback submitted successfully: {feedback_score} for session {session_id}, trace {trace_id}"
                # )
            except Exception as obs_error:
                logger.error(f"💥 Error adding feedback to observability: {obs_error}")
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": f"Observability error: {str(obs_error)}",
                        }
                    ),
                    500,
                )
        else:
            logger.warning("⚠️ Observability not available for feedback")
            # Still return success since feedback was saved to database

        logger.info("🎉 Feedback processing completed successfully")
        return jsonify(
            {
                "success": True,
                "message": "Feedback submitted successfully",
                "feedback_score": feedback_score,
                "trace_id": trace_id,
                "is_update": is_update,
            }
        )
    except Exception as e:
        logger.error(f"💥 Unexpected error in feedback endpoint: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    try:
        app.run(debug=True, host="0.0.0.0", port=5001)
    finally:
        # Clean up database connections
        cleanup_state_manager()

        # Clean up global observability
        if observability and observability.is_enabled():
            try:
                # Flush data before cleanup
                observability.flush_data()

                # Validate session state before cleanup
                if hasattr(observability, "validate_session_state"):
                    if not observability.validate_session_state():
                        logger.warning(
                            "Session state validation failed during global cleanup, attempting recovery"
                        )
                        if hasattr(observability, "recover_session_state"):
                            observability.recover_session_state()

                # Cleanup SDK
                observability.cleanup()
                logger.info("Cleaned up global observability")
            except Exception as e:
                logger.warning(f"Failed to cleanup global observability: {e}")
                # Try to force cleanup even if there are errors
                try:
                    if hasattr(observability, "flush_data"):
                        observability.flush_data()
                    if hasattr(observability, "cleanup"):
                        observability.cleanup()
                except Exception as force_cleanup_error:
                    logger.error(
                        f"Force global cleanup also failed: {force_cleanup_error}"
                    )
