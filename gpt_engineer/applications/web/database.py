"""
Database layer for Web UI state persistence.

This module provides SQLite-based persistence for all Web UI state including
sessions, traces, feedback, files, CLI outputs, and execution checkpoints.
"""

import json
import sqlite3
import threading
import time

from contextlib import contextmanager
from typing import Dict, List, Optional


class DatabaseManager:
    """
    Manages SQLite database connections and operations for Web UI state.

    Provides thread-safe database operations for sessions, traces, feedback,
    files, CLI outputs, and execution checkpoints.
    """

    def __init__(self, db_path: str = "web_ui_state.db"):
        """
        Initialize the database manager.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
        return self._local.connection

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic commit/rollback."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_database(self):
        """Initialize the database schema."""
        with self.get_connection() as conn:
            # Sessions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    project_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_activity REAL NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    metadata TEXT DEFAULT '{}'
                )
            """
            )

            # Traces table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    mode TEXT,
                    created_at REAL NOT NULL,
                    execution_result TEXT DEFAULT '{}',
                    feedback INTEGER DEFAULT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """
            )

            # Trace files table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trace_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (trace_id) REFERENCES traces (trace_id) ON DELETE CASCADE,
                    UNIQUE(trace_id, file_path)
                )
            """
            )

            # CLI outputs table for stream messages
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cli_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    trace_id TEXT,
                    message TEXT NOT NULL,
                    message_type TEXT DEFAULT 'message',
                    timestamp REAL NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                    FOREIGN KEY (trace_id) REFERENCES traces (trace_id) ON DELETE CASCADE
                )
            """
            )

            # Engine checkpoints table for execution state
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS engine_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    checkpoint_data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    is_resolved INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """
            )

            # Session current files table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_content TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                    UNIQUE(session_id, file_path)
                )
            """
            )

            # Create indexes for better performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_session_id ON traces(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trace_files_trace_id ON trace_files(trace_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cli_outputs_session_id ON cli_outputs(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cli_outputs_trace_id ON cli_outputs(trace_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cli_outputs_timestamp ON cli_outputs(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_engine_checkpoints_session_id ON engine_checkpoints(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_engine_checkpoints_unresolved ON engine_checkpoints(session_id, is_resolved)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_files_session_id ON session_files(session_id)"
            )

    def close_connections(self):
        """Close all database connections."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


class SessionRepository:
    """Repository for session data operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_session(
        self, session_id: str, project_path: str, metadata: Dict = None
    ) -> bool:
        """Create a new session."""
        current_time = time.time()
        metadata_json = json.dumps(metadata or {})

        with self.db.get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, project_path, created_at, last_activity, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        session_id,
                        project_path,
                        current_time,
                        current_time,
                        metadata_json,
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM sessions WHERE session_id = ?
            """,
                (session_id,),
            ).fetchone()

            if row:
                return {
                    "session_id": row["session_id"],
                    "project_path": row["project_path"],
                    "created_at": row["created_at"],
                    "last_activity": row["last_activity"],
                    "is_active": bool(row["is_active"]),
                    "metadata": json.loads(row["metadata"]),
                }
            return None

    def update_last_activity(self, session_id: str) -> bool:
        """Update session last activity timestamp."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE sessions SET last_activity = ? WHERE session_id = ?
            """,
                (time.time(), session_id),
            )
            return cursor.rowcount > 0

    def get_active_session(self) -> Optional[Dict]:
        """Get the currently active session."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM sessions WHERE is_active = 1 ORDER BY last_activity DESC LIMIT 1
            """
            ).fetchone()

            if row:
                return {
                    "session_id": row["session_id"],
                    "project_path": row["project_path"],
                    "created_at": row["created_at"],
                    "last_activity": row["last_activity"],
                    "is_active": bool(row["is_active"]),
                    "metadata": json.loads(row["metadata"]),
                }
            return None

    def set_active_session(self, session_id: str) -> bool:
        """Set a session as active and deactivate others."""
        with self.db.get_connection() as conn:
            # Deactivate all sessions
            conn.execute("UPDATE sessions SET is_active = 0")
            # Activate the specified session
            cursor = conn.execute(
                """
                UPDATE sessions SET is_active = 1, last_activity = ? WHERE session_id = ?
            """,
                (time.time(), session_id),
            )
            return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            return cursor.rowcount > 0

    def list_sessions(self) -> List[Dict]:
        """List all sessions."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM sessions ORDER BY last_activity DESC
            """
            ).fetchall()

            return [
                {
                    "session_id": row["session_id"],
                    "project_path": row["project_path"],
                    "created_at": row["created_at"],
                    "last_activity": row["last_activity"],
                    "is_active": bool(row["is_active"]),
                    "metadata": json.loads(row["metadata"]),
                }
                for row in rows
            ]


class TraceRepository:
    """Repository for trace data operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_trace(
        self, trace_id: str, session_id: str, prompt: str, mode: str = None
    ) -> bool:
        """Create a new trace."""
        with self.db.get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO traces (trace_id, session_id, prompt, mode, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (trace_id, session_id, prompt, mode, time.time()),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get trace by ID."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM traces WHERE trace_id = ?
            """,
                (trace_id,),
            ).fetchone()

            if row:
                return {
                    "trace_id": row["trace_id"],
                    "session_id": row["session_id"],
                    "prompt": row["prompt"],
                    "mode": row["mode"],
                    "created_at": row["created_at"],
                    "execution_result": json.loads(row["execution_result"]),
                    "feedback": row["feedback"],
                }
            return None

    def update_trace_execution_result(
        self, trace_id: str, execution_result: Dict
    ) -> bool:
        """Update trace execution result."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE traces SET execution_result = ? WHERE trace_id = ?
            """,
                (json.dumps(execution_result), trace_id),
            )
            return cursor.rowcount > 0

    def update_trace_feedback(self, trace_id: str, feedback: int) -> bool:
        """Update trace feedback score."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE traces SET feedback = ? WHERE trace_id = ?
            """,
                (feedback, trace_id),
            )
            return cursor.rowcount > 0

    def update_trace_mode(self, trace_id: str, mode: str) -> bool:
        """Update trace mode."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE traces SET mode = ? WHERE trace_id = ?
            """,
                (mode, trace_id),
            )
            return cursor.rowcount > 0

    def get_session_traces(self, session_id: str) -> List[Dict]:
        """Get all traces for a session."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM traces WHERE session_id = ? ORDER BY created_at
            """,
                (session_id,),
            ).fetchall()

            return [
                {
                    "trace_id": row["trace_id"],
                    "session_id": row["session_id"],
                    "prompt": row["prompt"],
                    "mode": row["mode"],
                    "created_at": row["created_at"],
                    "execution_result": json.loads(row["execution_result"]),
                    "feedback": row["feedback"],
                }
                for row in rows
            ]

    def get_latest_trace(self, session_id: str) -> Optional[Dict]:
        """Get the latest trace for a session."""
        with self.db.get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM traces WHERE session_id = ? ORDER BY created_at DESC LIMIT 1
            """,
                (session_id,),
            ).fetchone()

            if row:
                return {
                    "trace_id": row["trace_id"],
                    "session_id": row["session_id"],
                    "prompt": row["prompt"],
                    "mode": row["mode"],
                    "created_at": row["created_at"],
                    "execution_result": json.loads(row["execution_result"]),
                    "feedback": row["feedback"],
                }
            return None


class TraceFileRepository:
    """Repository for trace file operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_trace_files(self, trace_id: str, files: Dict[str, str]) -> bool:
        """Save files for a trace."""
        current_time = time.time()
        with self.db.get_connection() as conn:
            # Delete existing files for this trace
            conn.execute("DELETE FROM trace_files WHERE trace_id = ?", (trace_id,))

            # Insert new files
            for file_path, content in files.items():
                conn.execute(
                    """
                    INSERT INTO trace_files (trace_id, file_path, file_content, created_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (trace_id, file_path, content, current_time),
                )
            return True

    def get_trace_files(self, trace_id: str) -> Dict[str, str]:
        """Get all files for a trace."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT file_path, file_content FROM trace_files WHERE trace_id = ?
            """,
                (trace_id,),
            ).fetchall()

            return {row["file_path"]: row["file_content"] for row in rows}


class CLIOutputRepository:
    """Repository for CLI output/stream operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def add_output(
        self,
        session_id: str,
        message: str,
        trace_id: str = None,
        message_type: str = "message",
    ) -> bool:
        """Add a CLI output message."""
        with self.db.get_connection() as conn:
            # Get next sequence number for this session
            row = conn.execute(
                """
                SELECT COALESCE(MAX(sequence_number), 0) + 1 as next_seq
                FROM cli_outputs WHERE session_id = ?
            """,
                (session_id,),
            ).fetchone()
            next_seq = row["next_seq"]

            conn.execute(
                """
                INSERT INTO cli_outputs (session_id, trace_id, message, message_type, timestamp, sequence_number)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (session_id, trace_id, message, message_type, time.time(), next_seq),
            )
            return True

    def get_session_outputs(self, session_id: str) -> List[str]:
        """Get all CLI outputs for a session in order."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT message FROM cli_outputs
                WHERE session_id = ?
                ORDER BY sequence_number
            """,
                (session_id,),
            ).fetchall()

            return [row["message"] for row in rows]

    def get_trace_outputs(self, trace_id: str) -> List[str]:
        """Get all CLI outputs for a specific trace."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT message FROM cli_outputs
                WHERE trace_id = ?
                ORDER BY sequence_number
            """,
                (trace_id,),
            ).fetchall()

            return [row["message"] for row in rows]

    def clear_session_outputs(self, session_id: str) -> bool:
        """Clear all CLI outputs for a session."""
        with self.db.get_connection():
            # Add a special "stream cleared" message
            self.add_output(
                session_id, "[STREAM_CLEARED]", message_type="stream_cleared"
            )
            return True


class CheckpointRepository:
    """Repository for execution checkpoint operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def save_checkpoint(
        self, session_id: str, checkpoint_type: str, checkpoint_data: Dict
    ) -> int:
        """Save an execution checkpoint."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO engine_checkpoints (session_id, checkpoint_type, checkpoint_data, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (session_id, checkpoint_type, json.dumps(checkpoint_data), time.time()),
            )
            return cursor.lastrowid

    def get_pending_checkpoint(
        self, session_id: str, checkpoint_type: str = None
    ) -> Optional[Dict]:
        """Get the pending checkpoint for a session."""
        with self.db.get_connection() as conn:
            query = """
                SELECT * FROM engine_checkpoints
                WHERE session_id = ? AND is_resolved = 0
            """
            params = [session_id]

            if checkpoint_type:
                query += " AND checkpoint_type = ?"
                params.append(checkpoint_type)

            query += " ORDER BY created_at DESC LIMIT 1"

            row = conn.execute(query, params).fetchone()

            if row:
                return {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "checkpoint_type": row["checkpoint_type"],
                    "checkpoint_data": json.loads(row["checkpoint_data"]),
                    "created_at": row["created_at"],
                    "is_resolved": bool(row["is_resolved"]),
                }
            return None

    def resolve_checkpoint(self, checkpoint_id: int) -> bool:
        """Mark a checkpoint as resolved."""
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE engine_checkpoints SET is_resolved = 1 WHERE id = ?
            """,
                (checkpoint_id,),
            )
            return cursor.rowcount > 0

    def clear_session_checkpoints(self, session_id: str) -> bool:
        """Clear all checkpoints for a session."""
        with self.db.get_connection() as conn:
            conn.execute(
                "DELETE FROM engine_checkpoints WHERE session_id = ?", (session_id,)
            )
            return True


class SessionFileRepository:
    """Repository for session current files operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def update_session_files(self, session_id: str, files: Dict[str, str]) -> bool:
        """Update current files for a session."""
        current_time = time.time()
        with self.db.get_connection() as conn:
            # Delete existing files for this session
            conn.execute(
                "DELETE FROM session_files WHERE session_id = ?", (session_id,)
            )

            # Insert new files
            for file_path, content in files.items():
                conn.execute(
                    """
                    INSERT INTO session_files (session_id, file_path, file_content, updated_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (session_id, file_path, content, current_time),
                )
            return True

    def get_session_files(self, session_id: str) -> Dict[str, str]:
        """Get current files for a session."""
        with self.db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT file_path, file_content FROM session_files WHERE session_id = ?
            """,
                (session_id,),
            ).fetchall()

            return {row["file_path"]: row["file_content"] for row in rows}
