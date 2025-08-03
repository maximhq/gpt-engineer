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

        # Initialize observability for this session
        if observability and observability.is_enabled():
            try:
                session_tags = {
                    "operation": "web_ui_session",
                    "session_id": session_id,
                }

                session_metadata = {
                    "web_ui_session": {
                        "session_id": session_id,
                        "project_path": project_path,
                        "model": ai.model_name,
                        "temperature": ai.temperature,
                    },
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

        self.engine = MultiTurnEngine(ai, project_path, preprompts_holder)

        # Initialize stream for this session
        if session_id:
            stream_sessions[session_id] = []

    def add_stream_output(self, message: str):
        """Add output to the stream for this session."""
        if self.session_id and self.session_id in stream_sessions:
            # Remove timestamp formatting - just use the message as is
            stream_sessions[self.session_id].append(message)

    @contextmanager
    def capture_output(self):
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

            # Add to stream (only if we have content and filter out observability noise)
            if stdout_content:
                for line in stdout_content.strip().split("\n"):
                    if line.strip() and not self._should_filter_line(line.strip()):
                        self.add_stream_output(line.strip())

            if stderr_content:
                for line in stderr_content.strip().split("\n"):
                    if line.strip() and not self._should_filter_line(line.strip()):
                        self.add_stream_output(f"ERROR: {line.strip()}")

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
                }

                trace_metadata = {
                    "prompt_preview": prompt_text[:200] + "..."
                    if len(prompt_text) > 200
                    else prompt_text,
                    "prompt_length": len(prompt_text),
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
            from gpt_engineer.core.prompt import Prompt

            prompt = Prompt(prompt_text)

            # Add initial message to stream
            self.add_stream_output(
                f"Processing prompt: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}"
            )

            # Use the real engine to determine mode and execute with output capture
            with self.capture_output():
                mode = self.engine._determine_mode(prompt)
                self.add_stream_output(f"Determined mode: {mode}")
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
                f"Completed {mode} mode - generated {len(files_dict)} files"
            )

            # End trace with success
            if self.observability and self.observability.is_enabled() and trace_id:
                try:
                    # Calculate comprehensive metrics for trace output
                    total_size = sum(len(content) for content in files_dict.values())
                    file_types = list(
                        set(Path(fname).suffix for fname in files_dict.keys())
                    )

                    trace_output = f"""# 🎯 Web UI Prompt Summary

## 📊 Overview
- **Mode**: {mode.title()}
- **Files Generated**: {len(files_dict)} files ({total_size:,} characters)
- **Model Used**: {self.ai.model_name}
- **Session ID**: {self.session_id}

## 📁 Files by Type
{chr(10).join([f'- **{ext if ext != "no_extension" else "other"}**: {", ".join(files)}' for ext, files in {ext: [f for f in files_dict.keys() if Path(f).suffix == ext] for ext in file_types}.items()])}

## 🚀 Context
- **Project Path**: {str(Path(self.project_path).absolute())}
- **Prompt Length**: {len(prompt_text)} characters

---
*Generated by GPT-Engineer Web UI*
"""

                    self.observability.set_trace_output(trace_output)
                    self.observability.end_trace(trace_id)

                except Exception as e:
                    logger.warning(f"Failed to end trace: {e}")

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
            self.add_stream_output(error_msg)

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
                        },
                        tags={
                            "error_type": "web_ui_prompt_error",
                            "session_id": self.session_id,
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

    def cleanup(self):
        """Clean up observability session when done."""
        if self.observability and self.observability.is_enabled():
            try:
                self.observability.flush_data()
                if self.session_id:
                    self.observability.end_session()
                logger.info(f"Cleaned up observability session: {self.session_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup observability session {self.session_id}: {e}"
                )


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
        result = engine.process_prompt(prompt_text)

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
                            yield f"data: {json.dumps({'type': 'message', 'message': current_messages[i]})}\n\n"
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
                    engine.observability.add_feedback(feedback_score, trace_id=trace_id)
                    logger.info(
                        f"✅ Feedback submitted successfully: {feedback_score} for session {session_id}, trace {trace_id}"
                    )
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
                observability.cleanup()
                logger.info("Cleaned up global observability")
            except Exception as e:
                logger.warning(f"Failed to cleanup global observability: {e}")
