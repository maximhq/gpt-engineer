"""
Database-backed state management for Web UI.

This module provides database-backed replacements for the global state objects
used in the Web UI, ensuring all state is persisted in SQLite.
"""

import time
import uuid
from typing import Dict, List, Optional

from .database import (
    DatabaseManager,
    SessionRepository,
    TraceRepository,
    TraceFileRepository,
    CLIOutputRepository,
    CheckpointRepository,
    SessionFileRepository,
)


class DatabaseBackedTrace:
    """Database-backed replacement for the Trace class."""
    
    def __init__(self, state_manager: 'WebUIStateManager', trace_id: str, 
                 session_id: str = None, prompt: str = None, mode: str = None):
        """
        Initialize a database-backed trace.
        
        Parameters
        ----------
        state_manager : WebUIStateManager
            The state manager instance.
        trace_id : str
            The trace ID.
        session_id : str, optional
            The session ID (for creating new traces).
        prompt : str, optional
            The prompt text (for creating new traces).
        mode : str, optional
            The mode (for creating new traces).
        """
        self.state_manager = state_manager
        self.trace_id = trace_id
        self._data = None
        
        # If session_id and prompt are provided, create a new trace
        if session_id and prompt is not None:
            self.state_manager.trace_repo.create_trace(trace_id, session_id, prompt, mode)
    
    def _load_data(self) -> Dict:
        """Load trace data from database."""
        if self._data is None:
            self._data = self.state_manager.trace_repo.get_trace(self.trace_id)
            if self._data is None:
                raise ValueError(f"Trace {self.trace_id} not found in database")
        return self._data
    
    def _invalidate_cache(self):
        """Invalidate cached data to force reload."""
        self._data = None
    
    @property
    def prompt(self) -> str:
        """Get the trace prompt."""
        return self._load_data()['prompt']
    
    @property
    def mode(self) -> str:
        """Get the trace mode."""
        return self._load_data()['mode']
    
    @mode.setter
    def mode(self, mode: str):
        """Set the trace mode."""
        self.state_manager.trace_repo.update_trace_mode(self.trace_id, mode)
        self._invalidate_cache()
    
    @property
    def session_id(self) -> str:
        """Get the trace session ID."""
        return self._load_data()['session_id']
    
    @property
    def created_at(self) -> float:
        """Get the trace creation timestamp."""
        return self._load_data()['created_at']
    
    @property
    def files(self) -> Dict[str, str]:
        """Get the trace files."""
        return self.state_manager.trace_file_repo.get_trace_files(self.trace_id)
    
    @files.setter
    def files(self, files: Dict[str, str]):
        """Set the trace files."""
        self.state_manager.trace_file_repo.save_trace_files(self.trace_id, files)
    
    @property
    def feedback(self) -> Optional[int]:
        """Get the trace feedback."""
        return self._load_data()['feedback']
    
    @feedback.setter
    def feedback(self, feedback: int):
        """Set the trace feedback."""
        self.state_manager.trace_repo.update_trace_feedback(self.trace_id, feedback)
        self._invalidate_cache()
    
    @property
    def execution_result(self) -> Dict:
        """Get the trace execution result."""
        return self._load_data()['execution_result']
    
    @execution_result.setter
    def execution_result(self, result: Dict):
        """Set the trace execution result."""
        self.state_manager.trace_repo.update_trace_execution_result(self.trace_id, result)
        self._invalidate_cache()
    
    def to_dict(self) -> Dict:
        """Convert trace to dictionary for JSON serialization."""
        data = self._load_data()
        return {
            "trace_id": self.trace_id,
            "prompt": data['prompt'],
            "mode": data['mode'],
            "files": self.files,
            "feedback": data['feedback'],
            "execution_result": data['execution_result'],
            "created_at": data['created_at'],
        }


class DatabaseBackedSession:
    """Database-backed replacement for the Session class."""
    
    def __init__(self, state_manager: 'WebUIStateManager', session_id: str, 
                 project_path: str = None):
        """
        Initialize a database-backed session.
        
        Parameters
        ----------
        state_manager : WebUIStateManager
            The state manager instance.
        session_id : str
            The session ID.
        project_path : str, optional
            The project path (for creating new sessions).
        """
        self.state_manager = state_manager
        self.session_id = session_id
        self._data = None
        self.engine = None  # Will be set by the Web UI when engine is created
        
        # If project_path is provided, create a new session
        if project_path:
            self.state_manager.session_repo.create_session(session_id, project_path)
    
    def _load_data(self) -> Dict:
        """Load session data from database."""
        if self._data is None:
            self._data = self.state_manager.session_repo.get_session(self.session_id)
            if self._data is None:
                raise ValueError(f"Session {self.session_id} not found in database")
        return self._data
    
    def _invalidate_cache(self):
        """Invalidate cached data to force reload."""
        self._data = None
    
    @property
    def project_path(self) -> str:
        """Get the session project path."""
        return self._load_data()['project_path']
    
    @property
    def created_at(self) -> float:
        """Get the session creation timestamp."""
        return self._load_data()['created_at']
    
    @property
    def last_activity(self) -> float:
        """Get the session last activity timestamp."""
        return self._load_data()['last_activity']
    
    def update_last_activity(self):
        """Update the session last activity timestamp."""
        self.state_manager.session_repo.update_last_activity(self.session_id)
        self._invalidate_cache()
    
    @property
    def traces(self) -> List[DatabaseBackedTrace]:
        """Get all traces for this session."""
        trace_data_list = self.state_manager.trace_repo.get_session_traces(self.session_id)
        return [
            DatabaseBackedTrace(self.state_manager, trace_data['trace_id'])
            for trace_data in trace_data_list
        ]
    
    @property
    def current_files(self) -> Dict[str, str]:
        """Get the current files for this session."""
        return self.state_manager.session_file_repo.get_session_files(self.session_id)
    
    @current_files.setter
    def current_files(self, files: Dict[str, str]):
        """Set the current files for this session."""
        self.state_manager.session_file_repo.update_session_files(self.session_id, files)
    
    def add_trace(self, trace: DatabaseBackedTrace) -> None:
        """Add a trace to this session (trace should already be created)."""
        # The trace should already be in the database, just update activity
        self.update_last_activity()
    
    def get_trace(self, trace_id: str) -> Optional[DatabaseBackedTrace]:
        """Get a specific trace by ID."""
        trace_data = self.state_manager.trace_repo.get_trace(trace_id)
        if trace_data and trace_data['session_id'] == self.session_id:
            return DatabaseBackedTrace(self.state_manager, trace_id)
        return None
    
    def get_latest_trace(self) -> Optional[DatabaseBackedTrace]:
        """Get the most recent trace."""
        trace_data = self.state_manager.trace_repo.get_latest_trace(self.session_id)
        if trace_data:
            return DatabaseBackedTrace(self.state_manager, trace_data['trace_id'])
        return None
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary for JSON serialization."""
        data = self._load_data()
        return {
            "session_id": self.session_id,
            "project_path": data['project_path'],
            "traces": [trace.to_dict() for trace in self.traces],
            "current_files": self.current_files,
            "created_at": data['created_at'],
            "last_activity": data['last_activity'],
        }


class WebUIStateManager:
    """
    Central state manager for Web UI that coordinates all database operations.
    
    This replaces the global variables active_session and stream_sessions
    with database-backed state management.
    """
    
    def __init__(self, db_path: str = "web_ui_state.db"):
        """
        Initialize the Web UI state manager.
        
        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.
        """
        self.db_manager = DatabaseManager(db_path)
        self.session_repo = SessionRepository(self.db_manager)
        self.trace_repo = TraceRepository(self.db_manager)
        self.trace_file_repo = TraceFileRepository(self.db_manager)
        self.cli_output_repo = CLIOutputRepository(self.db_manager)
        self.checkpoint_repo = CheckpointRepository(self.db_manager)
        self.session_file_repo = SessionFileRepository(self.db_manager)
    
    def get_or_create_session(self, session_id: str, project_path: str = None) -> DatabaseBackedSession:
        """
        Get an existing session or create a new one.
        
        Parameters
        ----------
        session_id : str
            The session ID.
        project_path : str, optional
            Project path for creating new sessions.
            
        Returns
        -------
        DatabaseBackedSession
            The session object.
        """
        existing_session = self.session_repo.get_session(session_id)
        if existing_session:
            return DatabaseBackedSession(self, session_id)
        elif project_path:
            return DatabaseBackedSession(self, session_id, project_path)
        else:
            raise ValueError(f"Session {session_id} not found and no project_path provided")
    
    def set_active_session(self, session_id: str) -> bool:
        """Set a session as the active session."""
        return self.session_repo.set_active_session(session_id)
    
    def get_active_session(self) -> Optional[DatabaseBackedSession]:
        """Get the currently active session."""
        session_data = self.session_repo.get_active_session()
        if session_data:
            return DatabaseBackedSession(self, session_data['session_id'])
        return None
    
    def create_trace(self, session_id: str, prompt: str, mode: str = None) -> DatabaseBackedTrace:
        """
        Create a new trace for a session.
        
        Parameters
        ----------
        session_id : str
            The session ID.
        prompt : str
            The prompt text.
        mode : str, optional
            The mode.
            
        Returns
        -------
        DatabaseBackedTrace
            The new trace object.
        """
        trace_id = str(uuid.uuid4())
        return DatabaseBackedTrace(self, trace_id, session_id, prompt, mode)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data."""
        return self.session_repo.delete_session(session_id)
    
    def list_sessions(self) -> List[Dict]:
        """List all sessions."""
        return self.session_repo.list_sessions()
    
    # Stream/CLI output methods
    def add_stream_output(self, session_id: str, message: str, trace_id: str = None):
        """Add output to the stream for a session."""
        self.cli_output_repo.add_output(session_id, message, trace_id)
    
    def get_stream_output(self, session_id: str) -> List[str]:
        """Get stream output for a session."""
        return self.cli_output_repo.get_session_outputs(session_id)
    
    def get_trace_stream_output(self, trace_id: str) -> List[str]:
        """Get stream output for a specific trace."""
        return self.cli_output_repo.get_trace_outputs(trace_id)
    
    def clear_stream_output(self, session_id: str):
        """Clear stream output for a session."""
        self.cli_output_repo.clear_session_outputs(session_id)
    
    # Checkpoint methods
    def save_pending_execution(self, session_id: str, command: str, detected: bool = True) -> int:
        """Save a pending execution checkpoint."""
        checkpoint_data = {
            "command": command,
            "detected": detected,
        }
        return self.checkpoint_repo.save_checkpoint(session_id, "pending_execution", checkpoint_data)
    
    def get_pending_execution(self, session_id: str) -> Optional[Dict]:
        """Get the pending execution for a session."""
        checkpoint = self.checkpoint_repo.get_pending_checkpoint(session_id, "pending_execution")
        if checkpoint:
            return checkpoint['checkpoint_data']
        return None
    
    def has_pending_execution(self, session_id: str) -> bool:
        """Check if there's a pending execution for a session."""
        return self.get_pending_execution(session_id) is not None
    
    def clear_pending_execution(self, session_id: str) -> bool:
        """Clear pending execution for a session."""
        checkpoint = self.checkpoint_repo.get_pending_checkpoint(session_id, "pending_execution")
        if checkpoint:
            return self.checkpoint_repo.resolve_checkpoint(checkpoint['id'])
        return True
    
    def cleanup(self):
        """Clean up database connections."""
        self.db_manager.close_connections()


# Global state manager instance
_state_manager = None


def get_state_manager(db_path: str = "web_ui_state.db") -> WebUIStateManager:
    """Get the global state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = WebUIStateManager(db_path)
    return _state_manager


def cleanup_state_manager():
    """Clean up the global state manager."""
    global _state_manager
    if _state_manager:
        _state_manager.cleanup()
        _state_manager = None