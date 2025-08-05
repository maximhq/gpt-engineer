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
from typing import Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

# Type hints for function signatures
from gpt_engineer.core.ai import AI
from gpt_engineer.core.preprompts_holder import PrepromptsHolder
from gpt_engineer.core.prompt import Prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state - simplified to single session management
active_session = None
stream_sessions = {}

# Initialize observability globally
observability = None
try:
    from gpt_engineer.core.maxim_observability import init_observability

    observability = init_observability(enabled=True)
    logger.info("Observability initialized for web UI")
except ImportError:
    logger.warning("Observability not available")
    observability = None


class Trace:
    """Represents a single turn in a multi-turn conversation."""

    def __init__(self, trace_id: str, prompt: str, mode: str = None):
        self.trace_id = trace_id
        self.prompt = prompt
        self.mode = mode
        self.files = {}
        self.feedback = None
        self.execution_result = None
        self.created_at = time.time()

    def to_dict(self) -> Dict:
        """Convert trace to dictionary for JSON serialization."""
        return {
            "trace_id": self.trace_id,
            "prompt": self.prompt,
            "mode": self.mode,
            "files": self.files,
            "feedback": self.feedback,
            "execution_result": self.execution_result,
            "created_at": self.created_at,
        }


class Session:
    """Represents a complete multi-turn conversation session."""

    def __init__(self, session_id: str, project_path: str):
        self.session_id = session_id
        self.project_path = project_path
        self.traces: List[Trace] = []
        self.current_files = {}
        self.created_at = time.time()
        self.last_activity = time.time()

    def add_trace(self, trace: Trace) -> None:
        """Add a new trace to the session."""
        self.traces.append(trace)
        self.last_activity = time.time()

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a specific trace by ID."""
        for trace in self.traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def get_latest_trace(self) -> Optional[Trace]:
        """Get the most recent trace."""
        return self.traces[-1] if self.traces else None

    def to_dict(self) -> Dict:
        """Convert session to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "project_path": self.project_path,
            "traces": [trace.to_dict() for trace in self.traces],
            "current_files": self.current_files,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
        }


class WebMultiTurnEngine:
    """
    Web UI wrapper for the MultiTurnEngine that properly manages sessions and traces.
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
        self.trace_streams = {}  # Store stream output per trace
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

        # Initialize stream for this session
        if session_id:
            stream_sessions[session_id] = []

        # Initialize pending execution state
        self.pending_execution = None

    def add_stream_output(self, message: str, trace_id: str = None):
        """Add output to the stream for this session."""
        if self.session_id and self.session_id in stream_sessions:
            # Remove timestamp formatting - just use the message as is
            stream_sessions[self.session_id].append(message)

            # Store stream output per trace if trace_id is provided
            if trace_id:
                if trace_id not in self.trace_streams:
                    self.trace_streams[trace_id] = []
                self.trace_streams[trace_id].append(message)

    def clear_stream_output(self):
        """Clear stream output for this session."""
        if self.session_id and self.session_id in stream_sessions:
            # Add a special message to indicate stream was cleared
            stream_sessions[self.session_id] = ["[STREAM_CLEARED]"]
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
                    # Store the command for later execution
                    self.pending_execution = {
                        "command": command_to_execute,
                        "detected": True,
                    }

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
        trace_id = str(uuid.uuid4())
        trace = Trace(trace_id, prompt_text)

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

            # Update trace with results
            trace.mode = mode
            trace.files = files_dict
            trace.execution_result = {
                "success": True,
                "mode": mode,
                "files_count": len(files_dict),
            }

            # Add completion message
            self.add_stream_output(
                f"Completed {mode} mode - generated {len(files_dict)} files", trace_id
            )

            # Note: File attachments are already handled by the CLI multi-turn engine
            # No need to duplicate file attachment logging here

            # Store trace info for later ending (after execution completes)
            self.current_trace_id = trace_id
            self.current_trace_mode = mode
            self.current_trace_files = files_dict
            self.current_trace_prompt = prompt_text

            # Don't end trace here - wait until after execution completes
            # The trace will be ended in the execute endpoint after execute_entrypoint_next

            return {
                "success": True,
                "mode": mode,
                "files": files_dict,
                "files_count": len(files_dict),
                "message": f"Successfully executed {mode} mode",
                "trace_id": trace_id,
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
        return hasattr(self, "pending_execution") and self.pending_execution is not None

    def cleanup(self):
        """Clean up observability session when done."""
        if self.observability and self.observability.is_enabled():
            try:
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


# Create Flask app
app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    """Serve the main web interface."""
    return render_template("index.html")


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
    global active_session

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
        logger.info(f"Current active_session: {active_session}")

        # Create or get session
        if not active_session or active_session.session_id != session_id:
            # Create new session
            logger.info(f"Creating new session: {session_id}")
            project_path = f"projects/web_session_{session_id}"
            ai = create_ai_instance()
            preprompts_holder = create_preprompts_holder(project_path)

            active_session = Session(session_id, project_path)
            active_session.engine = WebMultiTurnEngine(
                ai, project_path, preprompts_holder, session_id
            )
            logger.info(f"Session created successfully: {active_session.session_id}")

            # Verify session was created properly
            if (
                not active_session
                or not hasattr(active_session, "engine")
                or not active_session.engine
            ):
                logger.error("Session creation failed - session or engine is None")
                return (
                    jsonify({"success": False, "error": "Session creation failed"}),
                    500,
                )
        else:
            # Reuse existing session for multi-turn conversation
            logger.info(f"Reusing existing session: {session_id}")
            # Clear stream output for new prompt in existing session
            active_session.engine.clear_stream_output()
            # Update last activity
            active_session.last_activity = time.time()

        # Process the prompt
        if not active_session:
            logger.error("Active session is None after session creation")
            return jsonify({"success": False, "error": "Session creation failed"}), 500

        if not hasattr(active_session, "engine") or not active_session.engine:
            logger.error("Session engine is not properly initialized")
            return (
                jsonify({"success": False, "error": "Session engine not initialized"}),
                500,
            )

        if not hasattr(active_session, "add_trace"):
            logger.error("Session is missing add_trace method")
            return (
                jsonify(
                    {"success": False, "error": "Session is not properly initialized"}
                ),
                500,
            )

        engine = active_session.engine

        # Check if there's pending execution and block new input
        if hasattr(engine, "pending_execution") and engine.pending_execution:
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

        result = engine.process_prompt(prompt_text)

        # Log turn completion event
        if observability and observability.is_enabled():
            try:
                from uuid import uuid4

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

        # Create trace and add to session
        try:
            if active_session and hasattr(active_session, "add_trace"):
                trace = Trace(result.get("trace_id"), prompt_text, result.get("mode"))
                trace.files = result.get("files", {})
                trace.execution_result = result

                active_session.add_trace(trace)
                logger.info(f"Trace added successfully: {trace.trace_id}")
            else:
                logger.warning(
                    "Active session is None or missing add_trace method - skipping trace creation"
                )
        except Exception as trace_error:
            logger.error(f"Error creating/adding trace: {trace_error}")
            # Continue without trace if there's an error
            pass

        # Add session ID to response
        result["session_id"] = session_id

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/execute", methods=["POST"])
def execute_code():
    """
    Handle code execution confirmation from the web interface.
    """
    global active_session

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

        # Check if session exists
        if not active_session or active_session.session_id != session_id:
            logger.error(f"Session {session_id} not found")
            return jsonify({"success": False, "error": "Session not found"}), 404

        # Check if user declined execution
        user_confirmed = user_response.lower() in ["", "y", "yes"]
        if not user_confirmed:
            logger.info("User declined execution")
            active_session.engine.add_stream_output("Ok, not executing the code.")
            # Clear pending execution when user declines
            active_session.engine.pending_execution = None
            # Clear stream output when user declines
            active_session.engine.clear_stream_output()
            return jsonify(
                {"success": True, "message": "Execution declined", "executed": False}
            )

        # User confirmed execution
        logger.info("User confirmed execution")
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
        if active_session.engine.pending_execution:
            active_session.engine.pending_execution["command"]

            try:
                # Import the execute_entrypoint_next function
                from gpt_engineer.core.default.steps import execute_entrypoint_next

                # Execute using the execute_entrypoint_next function with user response
                execute_entrypoint_next(
                    execution_env=active_session.engine.engine.execution_env,
                    files_dict=active_session.engine.engine.current_files,
                    user_response=user_response,
                )

                # Clear pending execution
                active_session.engine.pending_execution = None

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

                            # Format as markdown
                            trace_output = f"""# 🎯 Turn {active_session.engine.turn_number} Summary

## 📊 Overview
- **Mode**: {active_session.engine.current_trace_mode.title()}
- **Files Generated**: {len(files_dict)} files ({total_size:,} characters)
- **Model Used**: {active_session.engine.ai.model_name}
- **Cost**: ${cost_info.get('total_cost', 0):.4f} ({token_info.get('total_tokens', 0):,} tokens)

## 📁 Files by Type
{chr(10).join([f'- **{ext if ext != "no_extension" else "other"}**: {", ".join(files)}' for ext, files in {ext: [f for f in files_dict.keys() if Path(f).suffix == ext] for ext in file_types}.items()])}

## 🚀 Context
- **Project Path**: {str(Path(active_session.engine.project_path).absolute())}
- **Prompt Length**: {len(active_session.engine.current_trace_prompt)} characters
- **Turn Number**: {active_session.engine.turn_number}
- **Total Turns**: {len(active_session.engine.engine.conversation_history) if hasattr(active_session.engine.engine, 'conversation_history') else 0}

---
*Generated by GPT-Engineer Web UI*
"""

                            # Set trace output and end trace AFTER execution completes
                            observability.set_trace_output(trace_output)
                            observability.end_trace(
                                active_session.engine.current_trace_id
                            )

                            # Clear the current trace info
                            active_session.engine.current_trace_id = None
                            active_session.engine.current_trace_mode = None
                            active_session.engine.current_trace_files = None
                            active_session.engine.current_trace_prompt = None

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
                active_session.engine.add_stream_output(error_msg)
                # Clear pending execution on error
                active_session.engine.pending_execution = None
                # Clear stream output after error
                active_session.engine.clear_stream_output()

                # End the current trace with error output
                if (
                    hasattr(active_session.engine, "current_trace_id")
                    and active_session.engine.current_trace_id
                ):
                    try:
                        if observability and observability.is_enabled():
                            # Create error summary
                            error_summary = f"""# ❌ Turn {active_session.engine.turn_number} Failed

## 📊 Overview
- **Mode**: {active_session.engine.current_trace_mode.title() if active_session.engine.current_trace_mode else 'Unknown'}
- **Status**: ❌ Execution Failed
- **Error**: {str(e)}

## 🚀 Context
- **Project Path**: {str(Path(active_session.engine.project_path).absolute())}
- **Turn Number**: {active_session.engine.turn_number}

---
*Generated by GPT-Engineer Web UI*
"""

                            # Set error trace output and end trace
                            observability.set_trace_output(error_summary)
                            observability.end_trace(
                                active_session.engine.current_trace_id, error=e
                            )

                            # Clear the current trace info
                            active_session.engine.current_trace_id = None
                            active_session.engine.current_trace_mode = None
                            active_session.engine.current_trace_files = None
                            active_session.engine.current_trace_prompt = None

                    except Exception as trace_error:
                        logger.warning(
                            f"Failed to end trace after execution error: {trace_error}"
                        )
                        # Try to end trace without output if setting output fails
                        try:
                            if observability and observability.is_enabled():
                                observability.end_trace(
                                    active_session.engine.current_trace_id, error=e
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
    global active_session

    try:
        if not active_session or active_session.session_id != session_id:
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

        engine = active_session.engine
        has_pending_execution = (
            hasattr(engine, "pending_execution")
            and engine.pending_execution is not None
        )

        return jsonify(
            {
                "success": True,
                "session_exists": True,
                "has_pending_execution": has_pending_execution,
                "pending_execution": engine.pending_execution
                if has_pending_execution
                else None,
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

        # Get existing messages
        if session_id in stream_sessions:
            for message in stream_sessions[session_id]:
                # Handle special stream cleared message
                if message == "[STREAM_CLEARED]":
                    yield f"data: {json.dumps({'type': 'stream_cleared', 'message': 'Stream cleared'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'message', 'message': message})}\n\n"

        # Keep track of last message index
        last_index = len(stream_sessions.get(session_id, []))

        # Keep connection alive and check for new messages
        try:
            while True:
                time.sleep(0.5)  # Check every 500ms

                # Check for new messages
                if session_id in stream_sessions:
                    current_messages = stream_sessions[session_id]
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
    if session_id in stream_sessions:
        return jsonify({"success": True, "messages": stream_sessions[session_id]})
    else:
        return jsonify({"success": False, "error": "Session not found"}), 404


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
    global active_session

    try:
        if not active_session or active_session.session_id != session_id:
            return jsonify({"success": False, "error": "Session not found"}), 404

        engine = active_session.engine
        if hasattr(engine, "trace_streams") and trace_id in engine.trace_streams:
            return jsonify(
                {"success": True, "messages": engine.trace_streams[trace_id]}
            )
        else:
            return (
                jsonify({"success": False, "error": "Trace stream history not found"}),
                404,
            )

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
    global active_session

    if active_session and active_session.session_id == session_id:
        return jsonify({"success": True, "session": active_session.to_dict()})
    else:
        return jsonify({"success": False, "error": "Session not found"}), 404


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
    global active_session

    if active_session and active_session.session_id == session_id:
        return jsonify({"success": True, "files": active_session.current_files})
    else:
        return jsonify({"success": False, "error": "Session not found"}), 404


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    """
    List all active sessions.

    Returns
    -------
    JSON response with session list.
    """
    global active_session

    sessions = []
    if active_session:
        sessions.append(
            {
                "session_id": active_session.session_id,
                "created_at": active_session.created_at,
                "last_activity": active_session.last_activity,
                "trace_count": len(active_session.traces),
            }
        )

    return jsonify({"success": True, "sessions": sessions})


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
    global active_session

    if active_session and active_session.session_id == session_id:
        # Clean up observability session
        if hasattr(active_session, "engine"):
            active_session.engine.cleanup()

        # Clear stream session
        if session_id in stream_sessions:
            del stream_sessions[session_id]

        # Clear active session
        active_session = None

        return jsonify({"success": True, "message": "Session deleted"})
    else:
        return jsonify({"success": False, "error": "Session not found"}), 404


@app.route("/api/reset", methods=["POST"])
def reset_all_sessions():
    """
    Reset all sessions (clean up when page is refreshed).

    Returns
    -------
    JSON response indicating success.
    """
    global active_session

    try:
        # Clean up active session
        if active_session and hasattr(active_session, "engine"):
            active_session.engine.cleanup()

        # Clear all sessions
        active_session = None
        stream_sessions.clear()

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

        if not active_session or active_session.session_id != session_id:
            logger.error(f"❌ Session {session_id} not found in active sessions")
            return jsonify({"success": False, "error": "Session not found"}), 404

        # Find the trace
        trace = active_session.get_trace(trace_id)
        if not trace:
            logger.error(f"❌ Trace {trace_id} not found in session {session_id}")
            return jsonify({"success": False, "error": "Trace not found"}), 404

        logger.info(f"✅ Found trace data for {trace_id}")

        # Update trace feedback
        trace.feedback = feedback_score

        if hasattr(active_session, "engine") and active_session.engine:
            engine = active_session.engine
            logger.info("🔧 Engine found in session data")

            if (
                hasattr(engine, "observability")
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
                    logger.error(
                        f"💥 Error adding feedback to observability: {obs_error}"
                    )
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
                return (
                    jsonify({"success": False, "error": "Observability not available"}),
                    500,
                )
        else:
            logger.error(f"❌ No engine found in session {session_id}")
            return (
                jsonify({"success": False, "error": "Engine not found in session"}),
                500,
            )

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
