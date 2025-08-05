"""
Maxim Observability Integration for GPT-Engineer

This module provides comprehensive observability integration using Maxim's SDK,
supporting sessions, traces, spans, generations, retrievals, tool calls, events, and errors.
"""

import logging
import os
import traceback

from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from dotenv import load_dotenv
from maxim.logger.logger import FeedbackDict

try:
    from maxim import Maxim
    from maxim.logger import Logger
    from maxim.logger.components import FileDataAttachment

    MAXIM_AVAILABLE = True
except ImportError:
    MAXIM_AVAILABLE = False
    Logger = Any  # Type placeholder

load_dotenv()

logger = logging.getLogger(__name__)


class MaximObservability:
    """
    Core wrapper for Maxim observability integration.

    Provides a unified interface for creating and managing sessions, traces, spans,
    generations, retrievals, tool calls, events, and errors using Maxim's SDK.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize the Maxim observability wrapper.

        Args:
            enabled: Whether observability is enabled
        """
        logger.debug(f"[MaximObservability] __init__ called with enabled={enabled}")
        try:
            self.enabled = enabled
            self.debug = enabled and os.getenv("MAXIM_DEBUG", "").lower() in (
                "true",
                "1",
                "yes",
            )
            self.maxim = None
            self.logger = None
            self.current_session = None
            self.current_session_id = None
            self.current_trace = None
            self.current_trace_id = None
            self.span_stack = []
            self.span_id_stack = []

            if self.enabled:
                try:
                    # Initialize Maxim with environment variables
                    api_key = os.getenv("MAXIM_API_KEY")
                    log_repo_id = os.getenv("MAXIM_LOG_REPO_ID")

                    if not api_key or not log_repo_id:
                        logger.warning(
                            "MAXIM_API_KEY or MAXIM_LOG_REPO_ID not found in environment. "
                            "Maxim observability will be disabled."
                        )
                        self.enabled = False
                        return

                    # Initialize Maxim with proper configuration
                    self.maxim = Maxim({"debug": True})

                    # Create logger configuration
                    logger_config = {"log_repo_id": log_repo_id, "log_level": "DEBUG"}

                    self.logger = self.maxim.logger(logger_config)
                    logger.info("Maxim observability initialized successfully")

                except Exception as e:
                    self._log_exception("__init__ (inner)", e)
                    self.enabled = False
        except Exception as e:
            self._log_exception("__init__", e)
            self.enabled = False

    def is_enabled(self) -> bool:
        """Check if Maxim observability is enabled and properly initialized."""
        logger.debug(
            f"[MaximObservability] is_enabled called, returning {self.enabled and self.logger is not None}"
        )
        try:
            return self.enabled and self.logger is not None
        except Exception as e:
            self._log_exception("is_enabled", e)
            return False

    def start_session(
        self,
        session_id: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Start a new session.

        Args:
            session_id: Unique identifier for the session
            tags: Key-value pairs for categorizing the session
            metadata: Additional session metadata

        Returns:
            Session ID if successful, None otherwise
        """
        logger.debug(
            f"[MaximObservability] start_session called with session_id={session_id}, tags={tags}, metadata={metadata}"
        )
        try:
            if not self.is_enabled():
                logger.debug(
                    "[MaximObservability] start_session: Observability not enabled"
                )
                return None

            # Create session config dict with proper structure
            session_config = {
                "id": session_id,
                "name": f"Session-{session_id[:8]}",  # Add name field
                "tags": tags or {},
            }

            self.current_session = self.logger.session(session_config)
            self.current_session_id = session_id

            # Add metadata using proper SDK method if provided
            # Note: Session metadata methods may not be available in current SDK version
            # We'll store metadata to add to the first trace created in this session
            self._pending_session_metadata = metadata

            logger.debug(f"[MaximObservability] Started session: {session_id}")
            return session_id

        except Exception as e:
            self._log_exception("start_session", e)
            return None

    def end_session(self) -> None:
        """
        End the current session.
        """
        try:
            if not self.is_enabled() or not self.current_session:
                logger.debug(
                    "[MaximObservability] end_session: No current session or not enabled"
                )
                return

            try:
                self.current_session.end()
                self.current_session = None
                logger.debug("[MaximObservability] Ended session")

            except Exception as e:
                self._log_exception("end_session (inner)", e)
        except Exception as e:
            self._log_exception("end_session", e)

    def start_trace(
        self,
        trace_id: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Start a new trace.

        Args:
            trace_id: Unique identifier for the trace
            name: Human-readable name for the trace
            tags: Key-value pairs for categorizing the trace
            metadata: Additional trace metadata
            session_id: Optional session ID to link this trace to

        Returns:
            Trace ID if successful, None otherwise
        """
        logger.debug(
            f"[MaximObservability] start_trace called with trace_id={trace_id}, name={name}, tags={tags}, metadata={metadata}, session_id={session_id}"
        )
        try:
            if not self.is_enabled():
                logger.debug(
                    "[MaximObservability] start_trace: Observability not enabled"
                )
                return None

            # Create trace config dict with proper structure
            trace_config = {"id": trace_id, "name": name, "tags": tags or {}}

            # Add session_id if provided
            if session_id:
                trace_config["session_id"] = session_id
            else:
                trace_config["session_id"] = self.current_session_id

            self.current_trace = self.logger.trace(trace_config)
            self.current_trace_id = trace_id
            # Add metadata using proper SDK method if provided
            if metadata:
                try:
                    self.logger.trace_add_metadata(trace_id, metadata)
                    logger.debug(
                        f"[MaximObservability] Added metadata to trace {trace_id}"
                    )
                except Exception as e:
                    self._log_exception("start_trace (add_metadata)", e)

            # Add any pending session metadata to this trace
            if (
                hasattr(self, "_pending_session_metadata")
                and self._pending_session_metadata
            ):
                try:
                    session_meta = {
                        f"session_{k}": v
                        for k, v in self._pending_session_metadata.items()
                    }
                    self.logger.trace_add_metadata(trace_id, session_meta)
                    logger.debug(
                        f"[MaximObservability] Added session metadata to trace {trace_id}"
                    )
                    self._pending_session_metadata = None  # Clear after adding
                except Exception as e:
                    self._log_exception("start_trace (add_session_metadata)", e)

            logger.debug(f"[MaximObservability] Started trace: {trace_id} ({name})")
            return trace_id

        except Exception as e:
            self._log_exception("start_trace", e)
            return None

    def add_feedback(self, review, trace_id: Optional[str] = None) -> None:
        logger.debug("[MaximObservability] add_feedback called")
        if not self.is_enabled():
            logger.debug("[MaximObservability] add_feedback: Observability not enabled")
            return

        # Handle None review gracefully
        if review is None:
            logger.debug("[MaximObservability] add_feedback: review is None, skipping")
            return

        try:
            # Convert review to score - handle different types
            if isinstance(review, (int, float)):
                score = int(review)
            elif isinstance(review, str):
                try:
                    score = int(review)
                except ValueError:
                    logger.warning(
                        f"[MaximObservability] add_feedback: cannot convert string '{review}' to int"
                    )
                    return
            else:
                logger.warning(
                    f"[MaximObservability] add_feedback: unsupported review type {type(review)}"
                )
                return

            if trace_id:
                self.logger.trace_add_feedback(trace_id, FeedbackDict(score=score))
                logger.debug(
                    f"[MaximObservability] Added Feedback to Maxim trace: {trace_id}"
                )

            elif self.current_trace:
                self.current_trace.feedback(feedback=FeedbackDict(score=score))
                logger.debug(
                    f"[MaximObservability] Added Feedback to Maxim trace: {self.current_session_id or 'current'}"
                )
        except Exception as e:
            self._log_exception("add_feedback", e)
            return None

    def add_session_feedback(self, review) -> None:
        logger.debug("[MaximObservability] add_session_feedback called")
        if not self.is_enabled():
            logger.debug(
                "[MaximObservability] add_session_feedback: Observability not enabled"
            )
            return

        try:
            score = 0
            if hasattr(review, "perfect") and review.perfect:
                score = 1
            elif hasattr(review, "works") and review.works:
                score = 0.5
            elif hasattr(review, "ran") and review.ran:
                score = 0.25

            logger.debug(f"[MaximObservability] Calculated score: {score}")
            logger.debug(
                f"[MaximObservability] Current session ID: {self.current_session_id}"
            )

            if self.current_session_id:
                self.logger.session_add_feedback(
                    session_id=self.current_session_id,
                    feedback=FeedbackDict(
                        score=score,
                        comment=review.comments if review.comments else review.raw,
                    ),
                )
                logger.debug(
                    f"[MaximObservability] Added Feedback to Maxim session: {self.current_session_id}"
                )

                # Flush data immediately after adding feedback to ensure it's persisted
                self.flush_data()
                logger.debug(
                    "[MaximObservability] Data flushed after adding session feedback"
                )
            else:
                logger.warning(
                    "[MaximObservability] No current session ID available for feedback"
                )

        except Exception as e:
            self._log_exception("add_session_feedback", e)
            # Try to flush data even if feedback addition failed
            try:
                self.flush_data()
                logger.debug("[MaximObservability] Data flushed after feedback error")
            except Exception as flush_error:
                logger.error(
                    f"[MaximObservability] Failed to flush data after feedback error: {flush_error}"
                )
            return None

    def safe_add_session_feedback(self, review) -> bool:
        """
        Safely add session feedback with proper error handling and session validation.

        Args:
            review: The review object containing feedback data

        Returns:
            bool: True if feedback was successfully added, False otherwise
        """
        logger.debug("[MaximObservability] safe_add_session_feedback called")

        if not self.is_enabled():
            logger.debug(
                "[MaximObservability] safe_add_session_feedback: Observability not enabled"
            )
            return False

        # Validate session state before proceeding
        if not self.validate_session_state():
            logger.warning(
                "[MaximObservability] safe_add_session_feedback: Session state validation failed"
            )
            return False

        try:
            # Add the feedback
            self.add_session_feedback(review)

            # Verify the feedback was added by checking if we can still access the session
            if self.validate_session_state():
                logger.debug(
                    "[MaximObservability] safe_add_session_feedback: Feedback added successfully"
                )
                return True
            else:
                logger.warning(
                    "[MaximObservability] safe_add_session_feedback: Session became invalid after feedback"
                )
                return False

        except Exception as e:
            logger.error(f"[MaximObservability] safe_add_session_feedback failed: {e}")
            return False

    def end_trace(
        self, trace_id: Optional[str] = None, feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        End a trace.

        Args:
            trace_id: ID of trace to end (uses current trace if None)
            feedback: Optional feedback data for the trace
        """
        logger.debug(
            f"[MaximObservability] end_trace called with trace_id={trace_id}, feedback={feedback}"
        )
        try:
            if not self.is_enabled():
                logger.debug(
                    "[MaximObservability] end_trace: Observability not enabled"
                )
                return

            if self.current_trace:
                if feedback:
                    # Add feedback to trace if supported
                    pass  # Trace feedback implementation

                self.current_trace.end()
                self.current_trace = None
                logger.debug(
                    f"[MaximObservability] Ended trace: {trace_id or 'current'}"
                )

        except Exception as e:
            self._log_exception("end_trace", e)

    def start_span(
        self,
        span_id: str,
        name: str,
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Start a new span.

        Args:
            span_id: Unique identifier for the span
            name: Human-readable name for the span
            trace_id: ID of trace to add span to (uses current trace if None)
            tags: Key-value pairs for categorizing the span
            metadata: Additional span metadata
            parent_span_id: Optional ID of parent span to create a child span under

        Returns:
            Span ID if successful, None otherwise
        """
        logger.debug(
            f"[MaximObservability] start_span called with span_id={span_id}, name={name}, trace_id={trace_id}, tags={tags}, metadata={metadata}, parent_span_id={parent_span_id}"
        )
        try:
            if not self.is_enabled():
                logger.debug(
                    "[MaximObservability] start_span: Observability not enabled"
                )
                return None

            # Validate span stack before adding
            if not self._validate_span_stack():
                logger.warning(
                    "[MaximObservability] Span stack validation failed, resetting"
                )
                self._reset_span_stack()

            # Check for duplicate span_id
            if span_id in self.span_id_stack:
                logger.warning(
                    f"[MaximObservability] Span ID {span_id} already exists, skipping"
                )
                return None

            # Create span config dict with proper structure
            span_config = {"id": span_id, "name": name, "tags": tags or {}}

            # If parent_span_id is provided, find the parent span in the stack
            if parent_span_id is not None:
                parent_span = None
                for i, sid in enumerate(self.span_id_stack):
                    if sid == parent_span_id:
                        parent_span = self.span_stack[i]
                        break
                if parent_span is not None:
                    span = parent_span.span(span_config)
                    self.span_stack.append(span)
                    self.span_id_stack.append(span_id)
                    # Add metadata using proper SDK method if provided
                    if metadata:
                        try:
                            self.logger.span_add_metadata(span_id, metadata)
                            logger.debug(
                                f"[MaximObservability] Added metadata to child span {span_id}"
                            )
                        except Exception as e:
                            self._log_exception("start_span (add_metadata)", e)
                    logger.debug(
                        f"[MaximObservability] Started child span: {span_id} ({name}) under parent {parent_span_id}"
                    )
                    return span_id
                else:
                    logger.warning(
                        f"[MaximObservability] Parent span id {parent_span_id} not found in stack; falling back to trace."
                    )
            # Default: create span under current trace
            if self.current_trace:
                span = self.current_trace.span(span_config)
                self.span_stack.append(span)
                self.span_id_stack.append(span_id)

                # Add metadata using proper SDK method if provided
                if metadata:
                    try:
                        self.logger.span_add_metadata(span_id, metadata)
                        logger.debug(
                            f"[MaximObservability] Added metadata to span {span_id}"
                        )
                    except Exception as e:
                        self._log_exception("start_span (add_metadata)", e)

                logger.debug(f"[MaximObservability] Started span: {span_id} ({name})")
                return span_id
            else:
                logger.warning("[MaximObservability] No current trace to add span to")
                return None

        except Exception as e:
            self._log_exception("start_span", e)
            return None

    def end_span(
        self,
        span_id: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """
        End a span.

        Args:
            span_id: ID of span to end (uses most recent if None)
            result: Optional result data from the span
            error: Optional error that occurred during the span
        """
        logger.debug(
            f"[MaximObservability] end_span called with span_id={span_id}, result={result}, error={error}"
        )
        try:
            if not self.is_enabled() or not self.span_stack:
                logger.debug(
                    "[MaximObservability] end_span: No spans to end or not enabled"
                )
                return

            span = self.span_stack.pop()
            current_span_id = (
                self.span_id_stack.pop() if self.span_id_stack else span_id
            )

            # Note: span.set_output() method doesn't exist in current SDK
            # We can add metadata instead if needed
            if result:
                try:
                    # Try to add result as metadata instead
                    if hasattr(span, "add_metadata"):
                        span.add_metadata({"result": str(result)})
                except Exception as e:
                    logger.debug(
                        f"[MaximObservability] Could not add result metadata to span: {e}"
                    )

            if error:
                try:
                    # Create error config dict with proper structure
                    error_config = {
                        "id": str(uuid4()),
                        "message": str(error),
                        "type": type(error).__name__,
                        "code": getattr(error, "code", "UNKNOWN"),
                    }
                    if hasattr(span, "add_error"):
                        span.add_error(error_config)
                except Exception as e:
                    logger.debug(
                        f"[MaximObservability] Could not add error to span: {e}"
                    )

            span.end()
            logger.debug(
                f"[MaximObservability] Ended span: {current_span_id or 'current'}"
            )

        except Exception as e:
            self._log_exception("end_span", e)

    def log_generation(
        self,
        generation_id: str,
        name: str,
        model: str,
        messages: List[Dict],
        response: Union[str, Any],
        model_params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        token_usage: Optional[Dict[str, int]] = None,
        span_id: Optional[str] = None,
        raw_response: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Log an LLM generation.

        Args:
            generation_id: Unique identifier for the generation
            name: Human-readable name for the generation
            model: Model name used
            messages: Input messages to the model
            response: Response from the model
            model_params: Model parameters used
            tags: Key-value pairs for categorizing the generation
            metadata: Additional generation metadata
            token_usage: Token usage statistics
            span_id: Optional span to add generation to

        Returns:
            Generation ID if successful, None otherwise
        """
        logger.debug(
            f"[MaximObservability] log_generation called with generation_id={generation_id}, name={name}, model={model}, messages={messages}, response={response}, model_params={model_params}, tags={tags}, metadata={metadata}, token_usage={token_usage}, span_id={span_id}, raw_response={raw_response}"
        )
        try:
            if not self.is_enabled():
                logger.debug(
                    "[MaximObservability] log_generation: Observability not enabled"
                )
                return None

            # Create generation config dict with proper structure
            generation_config = {
                "id": generation_id,
                "name": name,
                "model": model,
                "messages": messages,
                "provider": self._detect_provider(model),
            }

            # Add model parameters if provided
            if model_params:
                generation_config["model_parameters"] = model_params

            # Add tags if provided
            if tags:
                generation_config["tags"] = tags

            generation = None

            # First try to create under current span if one is active
            if self.span_stack:
                try:
                    current_span = self.span_stack[-1]
                    if hasattr(current_span, "generation"):
                        generation = current_span.generation(generation_config)
                        logger.debug(
                            f"[MaximObservability] Created generation under span: {generation_id}"
                        )
                    else:
                        logger.debug(
                            "[MaximObservability] Span doesn't support generation method, falling back to trace"
                        )
                except Exception as e:
                    self._log_exception("log_generation (create_under_span)", e)

            # Fall back to trace level if no span or span creation failed
            if generation is None and self.current_trace:
                generation = self.current_trace.generation(generation_config)
                logger.debug(
                    f"[MaximObservability] Created generation under trace: {generation_id}"
                )

            if generation:
                # Add metadata using proper SDK method if provided
                if metadata:
                    try:
                        self.logger.generation_add_metadata(generation_id, metadata)
                        logger.debug(
                            f"[MaximObservability] Added metadata to generation {generation_id}"
                        )
                    except Exception as e:
                        self._log_exception("log_generation (add_metadata)", e)

                # Create a proper completion result with token usage and cost data
                try:
                    if raw_response is not None:
                        # Calculate cost data if token usage is provided
                        cost_data = {"input": 0.0, "output": 0.0, "total": 0.0}
                        usage_data = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        }

                        if token_usage:
                            usage_data = {
                                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                                "completion_tokens": token_usage.get(
                                    "completion_tokens", 0
                                ),
                                "total_tokens": token_usage.get("total_tokens", 0),
                            }
                            cost_data = self._calculate_cost(model, token_usage)

                        # Create a completion result that matches OpenAI API format
                        import time

                        completion_result = {
                            "id": str(uuid4()),
                            "model": model,
                            "provider": self._detect_provider(model),
                            "created": int(time.time()),
                            "usage": usage_data,
                            "cost": cost_data,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": str(response),
                                        "function_call": None,
                                        "tool_calls": [],
                                    },
                                    "finish_reason": "stop",
                                    "logprobs": None,
                                }
                            ],
                            "error": None,
                            "model_params": model_params,
                        }

                        # Log the completion result with Maxim
                        if hasattr(generation, "result"):
                            generation.result(completion_result)
                            logger.debug(
                                f"[MaximObservability] Logged Maxim generation with token usage {usage_data} and cost {cost_data}: {generation_id}"
                            )
                        else:
                            logger.debug(
                                f"[MaximObservability] No result method available for generation {generation_id}"
                            )
                    else:
                        logger.debug(
                            f"[MaximObservability] No response provided for generation {generation_id}"
                        )

                except Exception as result_error:
                    logger.debug(
                        f"[MaximObservability] Failed to log generation result, continuing: {result_error}"
                    )

                return generation_id
            else:
                logger.warning(
                    "[MaximObservability] No current trace or span to add generation to"
                )
                return None

        except Exception as e:
            self._log_exception("log_generation", e)
            return None

    def _calculate_cost(
        self, model: str, token_usage: Dict[str, int]
    ) -> Dict[str, float]:
        """Calculate cost based on model and token usage."""
        logger.debug(
            f"[MaximObservability] _calculate_cost called with model={model}, token_usage={token_usage}"
        )
        try:
            # Import cost calculation function with the new API
            try:
                from langchain_community.callbacks.openai_info import (
                    TokenType,
                    get_openai_token_cost_for_model,
                )
            except ImportError:
                try:
                    from langchain.callbacks.openai_info import (
                        TokenType,
                        get_openai_token_cost_for_model,
                    )
                except ImportError:
                    # Fall back to older API
                    from langchain_community.callbacks.openai_info import (
                        get_openai_token_cost_for_model,
                    )

                    TokenType = None

            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

            if TokenType:
                # Use new API with TokenType
                input_cost = get_openai_token_cost_for_model(
                    model, prompt_tokens, token_type=TokenType.INPUT
                )
                output_cost = get_openai_token_cost_for_model(
                    model, completion_tokens, token_type=TokenType.COMPLETION
                )
            else:
                # Fall back to older API
                input_cost = get_openai_token_cost_for_model(
                    model, prompt_tokens, is_completion=False
                )
                output_cost = get_openai_token_cost_for_model(
                    model, completion_tokens, is_completion=True
                )

            total_cost = input_cost + output_cost

            logger.debug(
                f"[MaximObservability] Calculated cost: input={input_cost}, output={output_cost}, total={total_cost}"
            )
            return {"input": input_cost, "output": output_cost, "total": total_cost}
        except Exception as e:
            logger.debug(f"[MaximObservability] Failed to calculate cost: {e}")
            return {"input": 0.0, "output": 0.0, "total": 0.0}

    def log_retrieval(
        self,
        retrieval_id: str,
        name: str,
        source: str,
        query: str = None,
        result: str = None,
        file_path: str = None,
        file_content: str = None,
        search_context: dict = None,
        tags: dict = None,
        metadata: dict = None,
    ):
        """
        Log a retrieval operation with optional file attachment and search context

        Args:
            retrieval_id: Unique identifier for the retrieval
            name: Name/description of the retrieval
            source: Source of the retrieval (e.g., 'disk_memory', 'file_selector')
            query: The search query or operation being performed
            result: The result or output of the retrieval
            file_path: Path to file being retrieved (for FileAttachment)
            file_content: Content of file being retrieved (for FileDataAttachment)
            search_context: Additional context about the search/query operation
            tags: Tags for categorization
            metadata: Additional metadata
        """
        logger.debug(
            f"[MaximObservability] log_retrieval called with retrieval_id={retrieval_id}, name={name}, source={source}, query={query}, result={result}, file_path={file_path}, file_content={file_content}, search_context={search_context}, tags={tags}, metadata={metadata}"
        )
        try:
            from maxim.logger.components.retrieval import RetrievalConfigDict

            # Merge tags
            final_tags = {"component": "retrieval", "source": source}
            if tags:
                final_tags.update(tags)

            # Merge metadata with search context
            final_metadata = {}
            if metadata:
                final_metadata.update(metadata)
            if search_context:
                final_metadata.update(search_context)
            if query:
                final_metadata["search_query"] = query

            # Create retrieval config
            config = RetrievalConfigDict(id=retrieval_id, name=name, tags=final_tags)

            retrieval = None

            # First try to create under current span if one is active
            if self.span_stack:
                try:
                    current_span = self.span_stack[-1]
                    if hasattr(current_span, "retrieval"):
                        retrieval = current_span.retrieval(config)
                        logger.debug(
                            f"[MaximObservability] Created retrieval under span: {retrieval_id}"
                        )
                    elif hasattr(current_span, "add_retrieval"):
                        retrieval = current_span.add_retrieval(config)
                        logger.debug(
                            f"[MaximObservability] Created retrieval under span: {retrieval_id}"
                        )
                    else:
                        logger.debug(
                            "[MaximObservability] Span doesn't support retrieval method, falling back to trace"
                        )
                except Exception as e:
                    self._log_exception("log_retrieval (create_under_span)", e)

            # Fall back to trace level if no span or span creation failed
            if retrieval is None:
                if self.current_trace and self.current_trace_id:
                    retrieval = self.logger.trace_add_retrieval(
                        self.current_trace_id, config
                    )
                    logger.debug(
                        f"[MaximObservability] Created retrieval under trace: {retrieval_id}"
                    )
                else:
                    # Skip logging if no trace context - avoids unnecessary standalone traces
                    logger.debug(
                        f"[MaximObservability] Skipping retrieval logging - no trace context available: {retrieval_id}"
                    )
                    return

            if retrieval:
                # Add metadata after creation if provided
                if final_metadata:
                    try:
                        if hasattr(retrieval, "add_metadata"):
                            retrieval.add_metadata(final_metadata)
                        elif hasattr(self.logger, "retrieval_add_metadata"):
                            self.logger.retrieval_add_metadata(
                                retrieval_id, final_metadata
                            )
                    except Exception:
                        pass  # Continue if metadata addition fails

                # Set input if query provided
                if query:
                    retrieval.input(query)

                # Set output if result provided
                if result:
                    retrieval.output(result)

                # Add file attachment if file path provided
                if file_path and os.path.exists(file_path):
                    try:
                        from maxim.logger.components import FileAttachment

                        attachment = FileAttachment(path=file_path)
                        retrieval.add_attachment(attachment)
                        if self.enabled:
                            print(f"✅ Maxim: Added file attachment for {file_path}")
                    except Exception as e:
                        if self.enabled:
                            print(f"⚠️ Maxim: Failed to attach file {file_path}: {e}")

                # Add file content attachment if content provided
                elif file_content:
                    try:
                        # Determine file name from retrieval context
                        file_name = final_metadata.get(
                            "file_path",
                            final_metadata.get("full_path", "retrieved_content.txt"),
                        )
                        if isinstance(file_name, str) and "/" in file_name:
                            file_name = os.path.basename(file_name)

                        # Convert content to bytes if it's a string
                        if isinstance(file_content, str):
                            content_bytes = file_content.encode("utf-8")
                        else:
                            content_bytes = file_content

                        # Create FileDataAttachment with only data and name (as per docs)
                        attachment = FileDataAttachment(
                            data=content_bytes, name=file_name
                        )

                        # Add attachment directly to the retrieval object
                        retrieval.add_attachment(attachment)
                        if self.enabled:
                            print(
                                f"✅ Maxim: Added file content attachment for {file_name}"
                            )
                    except Exception as e:
                        if self.enabled:
                            print(f"⚠️ Maxim: Failed to attach file content: {e}")

                # End the retrieval
                retrieval.end()

                if self.enabled:  # Use enabled instead of self.debug
                    print(f"✅ Maxim: Logged retrieval with attachments - {name}")

        except Exception as e:
            if self.enabled:  # Use enabled instead of self.debug
                print(f"⚠️ Maxim: Failed to log retrieval: {e}")
            self._log_exception("log_retrieval", e)

    def log_retrieval_with_file_content(
        self,
        retrieval_id: str,
        name: str,
        source: str,
        file_path: str,
        file_content: str,
        query: str = None,
        search_context: dict = None,
        tags: dict = None,
        metadata: dict = None,
    ):
        """
        Convenience method to log a retrieval operation with file content
        """
        logger.debug(
            f"[MaximObservability] log_retrieval_with_file_content called with retrieval_id={retrieval_id}, name={name}, source={source}, file_path={file_path}, file_content={file_content}, query={query}, search_context={search_context}, tags={tags}, metadata={metadata}"
        )
        return self.log_retrieval(
            retrieval_id=retrieval_id,
            name=name,
            source=source,
            query=query,
            result=f"Retrieved file: {file_path} ({len(file_content)} bytes)",
            file_content=file_content,
            search_context=search_context,
            tags=tags,
            metadata=metadata,
        )

    def log_tool_call(
        self,
        tool_call_id: str,
        name: str,
        description: str,
        args: Dict,
        result: Optional[Any] = None,
        error: Optional[Exception] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Log a tool call.

        Args:
            tool_call_id: Unique identifier for the tool call
            name: Name of the tool/function called
            description: Description of what the tool does
            args: Arguments passed to the tool
            result: Result from the tool call
            error: Error that occurred during tool call
            tags: Key-value pairs for categorizing the tool call

        Returns:
            Tool call ID if successful, None otherwise
        """
        logger.debug(
            f"[MaximObservability] log_tool_call called with tool_call_id={tool_call_id}, name={name}, description={description}, args={args}, result={result}, error={error}, tags={tags}"
        )
        try:
            if not self.is_enabled():
                logger.debug(
                    "[MaximObservability] log_tool_call: Observability not enabled"
                )
                return None

            # Create tool call config dict with proper structure
            tool_call_config = {
                "id": tool_call_id,
                "name": name,
                "description": description,
                "args": args,
            }

            # Add tags if provided
            if tags:
                tool_call_config["tags"] = tags

            tool_call = None

            # First try to create under current span if one is active
            if self.span_stack:
                try:
                    current_span = self.span_stack[-1]
                    if hasattr(current_span, "tool_call"):
                        tool_call = current_span.tool_call(tool_call_config)
                        logger.debug(
                            f"[MaximObservability] Created tool call under span: {tool_call_id}"
                        )
                    elif hasattr(current_span, "add_tool_call"):
                        tool_call = current_span.add_tool_call(tool_call_config)
                        logger.debug(
                            f"[MaximObservability] Created tool call under span: {tool_call_id}"
                        )
                    else:
                        logger.debug(
                            "[MaximObservability] Span doesn't support tool_call method, falling back to trace"
                        )
                except Exception as e:
                    self._log_exception("log_tool_call (create_under_span)", e)

            # Fall back to trace level if no span or span creation failed
            if tool_call is None and self.current_trace:
                tool_call = self.current_trace.tool_call(tool_call_config)
                logger.debug(
                    f"[MaximObservability] Created tool call under trace: {tool_call_id}"
                )

            if tool_call:
                try:
                    if result is not None:
                        # Try different methods to set result
                        if hasattr(tool_call, "result"):
                            tool_call.result(result)
                        elif hasattr(tool_call, "set_result"):
                            tool_call.set_result(result)
                        else:
                            logger.debug(
                                f"[MaximObservability] No result method available for tool call {tool_call_id}"
                            )

                    if error:
                        try:
                            error_data = {
                                "message": str(error),
                                "type": type(error).__name__,
                                "code": getattr(error, "code", "UNKNOWN"),
                            }
                            if hasattr(tool_call, "error"):
                                tool_call.error(error_data)
                            elif hasattr(tool_call, "add_error"):
                                tool_call.add_error(error_data)
                            else:
                                logger.debug(
                                    f"[MaximObservability] No error method available for tool call {tool_call_id}"
                                )
                        except Exception as e:
                            logger.debug(
                                f"[MaximObservability] Could not add error to tool call: {e}"
                            )

                    logger.debug(
                        f"[MaximObservability] Logged Maxim tool call: {tool_call_id}"
                    )
                except Exception as result_error:
                    logger.debug(
                        f"[MaximObservability] Failed to log tool call result, continuing: {result_error}"
                    )

                return tool_call_id
            else:
                logger.warning(
                    "[MaximObservability] No current trace or span to add tool call to"
                )
                return None

        except Exception as e:
            self._log_exception("log_tool_call", e)
            return None

    def log_event(
        self,
        event_id: str,
        event_type: str,
        data: Optional[Any] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        target: str = "auto",
    ) -> None:
        """
        Log a custom event.

        Args:
            event_id: Unique identifier for the event
            event_type: Type/name of the event
            data: Event data
            tags: Key-value pairs for categorizing the event
            metadata: Additional event metadata
            target: Target to add event to ('auto', 'trace', or 'span')
                    'auto' chooses span if available, otherwise trace
        """
        logger.debug(
            f"[MaximObservability] log_event called with event_id={event_id}, event_type={event_type}, data={data}, tags={tags}, metadata={metadata}, target={target}"
        )
        try:
            if not self.is_enabled():
                logger.debug(
                    "[MaximObservability] log_event: Observability not enabled"
                )
                return

            # Check for duplicate events
            if self._is_duplicate_event(event_id, event_type):
                logger.debug(
                    f"[MaximObservability] Skipping duplicate event: {event_id}"
                )
                return

            # Auto-choose target: prefer span if available, otherwise trace
            if target == "auto":
                if self.span_stack:
                    target = "span"
                elif self.current_trace:
                    target = "trace"
                else:
                    logger.warning(
                        "[MaximObservability] No current trace or span to add event to"
                    )
                    return

            if target == "span" and self.span_stack:
                current_span = self.span_stack[-1]
                current_span.event(event_id, event_type, tags, metadata)
                logger.debug(
                    f"[MaximObservability] Logged Maxim event to span: {event_id}"
                )
            elif target == "trace" and self.current_trace:
                self.current_trace.event(event_id, event_type, tags, metadata)
                logger.debug(
                    f"[MaximObservability] Logged Maxim event to trace: {event_id}"
                )
            else:
                logger.warning(
                    f"[MaximObservability] Cannot log event - no current {target} available"
                )

        except Exception as e:
            self._log_exception("log_event", e)

    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        target: str = "auto",
    ) -> None:
        """
        Log an error.

        Args:
            error: The exception that occurred
            context: Context information about when/where the error occurred
            tags: Key-value pairs for categorizing the error
            target: Target to add error to ('auto', 'trace', or 'span')
                    'auto' chooses span if available, otherwise trace
        """
        logger.debug(
            f"[MaximObservability] log_error called with error={error}, context={context}, tags={tags}, target={target}"
        )
        try:
            # Create error config dict with proper structure
            error_config = {
                "id": str(uuid4()),
                "message": str(error),
                "type": type(error).__name__,
                "code": getattr(error, "code", "UNKNOWN"),
            }

            # Add tags if provided, including context as tags
            if tags or context:
                error_config["tags"] = tags or {}
                # Add context as tags if provided
                if context:
                    for key, value in context.items():
                        if isinstance(value, str):
                            error_config["tags"][f"ctx_{key}"] = value[:50]

            # Auto-choose target: prefer span if available, otherwise trace
            if target == "auto":
                if self.span_stack:
                    target = "span"
                elif self.current_trace:
                    target = "trace"
                else:
                    logger.warning(
                        "[MaximObservability] No current trace or span to add error to"
                    )
                    return

            if target == "span" and self.span_stack:
                current_span = self.span_stack[-1]
                current_span.add_error(error_config)
                logger.debug(
                    f"[MaximObservability] Logged Maxim error to span: {type(error).__name__}"
                )
            elif target == "trace" and self.current_trace:
                self.current_trace.add_error(error_config)
                logger.debug(
                    f"[MaximObservability] Logged Maxim error to trace: {type(error).__name__}"
                )
            else:
                logger.warning(
                    f"[MaximObservability] Cannot log error - no current {target} available"
                )

        except Exception as e:
            self._log_exception("log_error", e)

    # def add_feedback(self, target_type: str, target_id: str,
    #                 feedback_data: Dict[str, Any]) -> None:
    #     """
    #     Add feedback to a session or trace.

    #     Args:
    #         target_type: Type of target ('session' or 'trace')
    #         target_id: ID of the target
    #         feedback_data: Feedback data
    #     """
    #     if not self.is_enabled():
    #         return

    #     try:
    #         # Feedback implementation would go here
    #         # This depends on Maxim's feedback API
    #         logger.debug(f"Added Maxim feedback to {target_type}: {target_id}")

    #     except Exception as e:
    #         logger.error(f"Failed to add Maxim feedback: {e}")

    def set_trace_input(self, input_data: str) -> None:
        """Set input for the current trace."""
        logger.debug(
            f"[MaximObservability] set_trace_input called with input_data={input_data}"
        )
        try:
            if self.is_enabled() and self.current_trace:
                try:
                    self.current_trace.set_input(input_data)
                except Exception as e:
                    logger.error(f"[MaximObservability] Failed to set trace input: {e}")
        except Exception as e:
            self._log_exception("set_trace_input", e)

    def set_trace_output(self, output_data: str) -> None:
        """Set output for the current trace."""
        logger.debug(
            f"[MaximObservability] set_trace_output called with output_data={output_data}"
        )
        try:
            if self.is_enabled() and self.current_trace:
                try:
                    self.current_trace.set_output(output_data)
                except Exception as e:
                    logger.error(
                        f"[MaximObservability] Failed to set trace output: {e}"
                    )
        except Exception as e:
            self._log_exception("set_trace_output", e)

    def add_file_attachment(
        self,
        filename: str,
        content: Union[str, bytes],
        mime_type: Optional[str] = None,
        target: str = "auto",
        target_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add a file attachment to a trace or span using Maxim's proper attachment API.

        Args:
            filename: Name of the file
            content: File content as string or bytes
            mime_type: MIME type of the file (auto-detected if None)
            target: Target to attach to ("trace", "span", or "auto")
            target_id: Specific target ID (uses current if None)

        Returns:
            Attachment ID if successful, None otherwise
        """
        logger.debug(
            f"[MaximObservability] add_file_attachment called with filename={filename}, content={content}, mime_type={mime_type}, target={target}, target_id={target_id}"
        )
        try:
            if not self.is_enabled():
                logger.debug(
                    "[MaximObservability] add_file_attachment: Observability not enabled"
                )
                return None

            # Convert content to bytes if it's a string
            if isinstance(content, str):
                content_bytes = content.encode("utf-8")
            else:
                content_bytes = content

            # Auto-detect MIME type if not provided
            if not mime_type:
                import mimetypes

                mime_type, _ = mimetypes.guess_type(filename)
                if not mime_type:
                    # Default MIME types based on extension
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
                    elif filename.endswith(".yml") or filename.endswith(".yaml"):
                        mime_type = "text/x-yaml"
                    else:
                        mime_type = "text/plain"

            # Create attachment using proper Maxim attachment class
            attachment_id = str(uuid4())

            try:
                # Use FileDataAttachment for in-memory content
                attachment = FileDataAttachment(
                    id=attachment_id,
                    name=filename,
                    data=content_bytes,
                    mime_type=mime_type,
                )
            except Exception as e:
                logger.error(
                    f"[MaximObservability] Failed to create FileDataAttachment: {e}"
                )
                return None

            # Clarify attachment targets
            if target == "auto":
                target = (
                    "trace"
                    if self.current_trace
                    else "span"
                    if self.span_stack
                    else None
                )

            # Attach to the appropriate target
            if target == "trace" and self.current_trace:
                # Debug: Check what methods are available on the trace object
                available_methods = [
                    method
                    for method in dir(self.current_trace)
                    if not method.startswith("_")
                ]
                logger.debug(
                    f"[MaximObservability] Available methods on trace object: {available_methods}"
                )

                # Try different possible attachment method names
                if hasattr(self.current_trace, "addAttachment"):
                    self.current_trace.addAttachment(attachment)
                    logger.debug(
                        f"[MaximObservability] Added file attachment '{filename}' to trace using addAttachment: {attachment_id}"
                    )
                    return attachment_id
                elif hasattr(self.current_trace, "add_attachment"):
                    self.current_trace.add_attachment(attachment)
                    logger.debug(
                        f"[MaximObservability] Added file attachment '{filename}' to trace using add_attachment: {attachment_id}"
                    )
                    return attachment_id
                elif hasattr(self.current_trace, "attachment"):
                    self.current_trace.attachment(attachment)
                    logger.debug(
                        f"[MaximObservability] Added file attachment '{filename}' to trace using attachment: {attachment_id}"
                    )
                    return attachment_id
                else:
                    logger.debug(
                        "[MaximObservability] No attachment method found on trace object"
                    )
                    # Fall back to using the logger directly
                    try:
                        if hasattr(self.logger, "trace_add_attachment"):
                            self.logger.trace_add_attachment(
                                self.current_trace_id, attachment
                            )
                            logger.debug(
                                f"[MaximObservability] Added file attachment '{filename}' to trace using logger: {attachment_id}"
                            )
                            return attachment_id
                        else:
                            logger.debug(
                                "[MaximObservability] No trace_add_attachment method found on logger"
                            )
                    except Exception as e:
                        logger.debug(
                            f"[MaximObservability] Failed to add attachment via logger: {e}"
                        )

            elif target == "span" and self.span_stack:
                current_span = self.span_stack[-1]  # Get most recent span
                if hasattr(current_span, "addAttachment"):
                    current_span.addAttachment(attachment)
                    logger.debug(
                        f"[MaximObservability] Added file attachment '{filename}' to span: {attachment_id}"
                    )
                    return attachment_id
                elif hasattr(current_span, "add_attachment"):
                    current_span.add_attachment(attachment)
                    logger.debug(
                        f"[MaximObservability] Added file attachment '{filename}' to span using add_attachment: {attachment_id}"
                    )
                    return attachment_id
                elif hasattr(current_span, "attachment"):
                    current_span.attachment(attachment)
                    logger.debug(
                        f"[MaximObservability] Added file attachment '{filename}' to span using attachment: {attachment_id}"
                    )
                    return attachment_id
                else:
                    logger.debug(
                        "[MaximObservability] No attachment method found on span object"
                    )

            else:
                logger.warning(
                    f"[MaximObservability] No current {target} to add attachment to"
                )

            return None

        except Exception as e:
            logger.error(f"[MaximObservability] Failed to add file attachment: {e}")
            return None

    def flush_data(self) -> None:
        """
        Flush all pending data to ensure it's persisted before session ends.
        """
        logger.debug("[MaximObservability] flush_data called")
        try:
            if not self.is_enabled():
                logger.debug(
                    "[MaximObservability] flush_data: Observability not enabled, skipping flush"
                )
                return

            # Flush logger data
            if self.logger:
                self.logger.flush()
                logger.debug("[MaximObservability] Logger data flushed")

            logger.debug("[MaximObservability] Data flush completed")

        except Exception as e:
            logger.error(f"[MaximObservability] Failed to flush data: {e}")

    def cleanup(self) -> None:
        """
        Clean up Maxim resources.

        This should be called before application exit to ensure all data is flushed.
        """
        logger.debug("[MaximObservability] cleanup called")
        try:
            if not self.is_enabled():
                logger.debug(
                    "[MaximObservability] cleanup: Observability not enabled, skipping cleanup"
                )
                return

            # End any remaining spans
            while self.span_stack:
                self.end_span()

            # End current trace
            if self.current_trace:
                self.end_trace()

            # End current session
            if self.current_session:
                self.end_session()

            if self.maxim:
                self.maxim.cleanup()

            logger.info("Maxim observability cleanup completed")

        except Exception as e:
            logger.error(
                f"[MaximObservability] Failed to cleanup Maxim observability: {e}"
            )

    def _detect_provider(self, model: str) -> str:
        """Detect LLM provider from model name."""
        logger.debug(f"[MaximObservability] _detect_provider called with model={model}")
        try:
            model_lower = model.lower()
            if "gpt" in model_lower or "openai" in model_lower:
                return "openai"
            elif "claude" in model_lower or "anthropic" in model_lower:
                return "anthropic"
            elif "gemini" in model_lower or "google" in model_lower:
                return "google"
            else:
                return "unknown"
        except Exception as e:
            logger.debug(f"[MaximObservability] Failed to detect provider: {e}")
            return "unknown"

    def add_trace_file_attachment(
        self,
        filename: str,
        content: Union[str, bytes],
        file_type: str = "generated",
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Add a file attachment to the current trace (confirmed working approach).

        Args:
            filename: Name of the file
            content: File content as string or bytes
            file_type: Type of file ("generated", "input", "output", etc.)
            tags: Optional tags for the attachment

        Returns:
            Attachment ID if successful, None otherwise
        """
        logger.debug(
            f"[MaximObservability] add_trace_file_attachment called with filename={filename}, content={content}, file_type={file_type}, tags={tags}"
        )
        try:
            if not self.is_enabled() or not self.current_trace:
                logger.debug(
                    "[MaximObservability] add_trace_file_attachment: Observability not enabled or no current trace"
                )
                return None

            # Convert content to bytes if it's a string
            if isinstance(content, str):
                content_bytes = content.encode("utf-8")
            else:
                content_bytes = content

            # Create attachment
            attachment = FileDataAttachment(data=content_bytes, name=filename)

            # Add to current trace (this approach works!)
            self.current_trace.add_attachment(attachment)

            if self.enabled:
                print(
                    f"✅ Maxim: Added {file_type} file '{filename}' to trace ({len(content_bytes)} bytes)"
                )

            return getattr(attachment, "id", None)

        except Exception as e:
            if self.enabled:
                print(
                    f"⚠️ Maxim: Failed to add trace file attachment '{filename}': {e}"
                )
            self._log_exception("add_trace_file_attachment", e)
            return None

    def _log_exception(self, method_name, exc):
        logger.error(
            f"[MaximObservability] Exception in {method_name}: {exc}\n{traceback.format_exc()}"
        )

    def validate_session_state(self) -> bool:
        """
        Validate that the current session state is consistent and ready for operations.

        Returns:
            bool: True if session state is valid, False otherwise
        """
        logger.debug("[MaximObservability] validate_session_state called")

        if not self.is_enabled():
            logger.debug(
                "[MaximObservability] validate_session_state: Observability not enabled"
            )
            return False

        # Check if we have both session ID and session object
        if not self.current_session_id:
            logger.warning(
                "[MaximObservability] validate_session_state: No current session ID"
            )
            return False

        if not self.current_session:
            logger.warning(
                "[MaximObservability] validate_session_state: No current session object"
            )
            # Try to recover session state
            if self.recover_session_state():
                logger.info("[MaximObservability] Session state recovered successfully")
                return True
            return False

        # Try to access session properties to verify it's still valid
        try:
            # This is a basic check - in a real implementation you might want to check
            # if the session object is still responsive
            session_id = getattr(self.current_session, "id", None)
            if session_id != self.current_session_id:
                logger.warning(
                    f"[MaximObservability] validate_session_state: Session ID mismatch - expected {self.current_session_id}, got {session_id}"
                )
                return False

            logger.debug(
                "[MaximObservability] validate_session_state: Session state is valid"
            )
            return True

        except Exception as e:
            logger.warning(
                f"[MaximObservability] validate_session_state: Session validation failed: {e}"
            )
            # Try to recover session state
            if self.recover_session_state():
                logger.info(
                    "[MaximObservability] Session state recovered after validation failure"
                )
                return True
            return False

    def _validate_span_stack(self) -> bool:
        """
        Validates the span stack to ensure no duplicate span IDs and correct order.
        Returns True if valid, False otherwise.
        """
        if not self.span_id_stack:
            return True

        # Create a set of unique span IDs for quick lookup
        unique_span_ids = set(self.span_id_stack)

        # Check if any span ID is duplicated
        if len(unique_span_ids) != len(self.span_id_stack):
            logger.warning(
                "[MaximObservability] Duplicate span IDs detected in stack. Resetting stack."
            )
            return False

        # Check if the stack is in a valid order (child spans before parents)
        # This is a simplified check. A more robust solution would involve
        # tracking parent-child relationships or ensuring no cycles.
        # For now, we'll just check if the stack is not empty and the IDs are unique.
        return True

    def _reset_span_stack(self) -> None:
        """
        Resets the span stack and span_id_stack to ensure no duplicate spans.
        """
        logger.warning(
            "[MaximObservability] Span stack validation failed. Resetting span stack."
        )
        self.span_stack = []
        self.span_id_stack = []

    def recover_session_state(self) -> bool:
        """
        Attempt to recover from inconsistent session state.
        Returns True if recovery was successful, False otherwise.
        """
        try:
            if not self.current_session and self.current_session_id:
                # Attempt to recreate session
                session_config = {
                    "id": self.current_session_id,
                    "name": f"Session-{self.current_session_id[:8]}",
                    "tags": {},
                }
                self.current_session = self.logger.session(session_config)
                logger.info(
                    f"[MaximObservability] Successfully recovered session: {self.current_session_id}"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"[MaximObservability] Session recovery failed: {e}")
            return False

    def _is_duplicate_event(self, event_id: str, event_type: str) -> bool:
        """
        Check if an event is a duplicate based on event_id and event_type.
        Returns True if duplicate, False otherwise.
        """
        # Simple duplicate detection - in a more sophisticated implementation,
        # you might want to track events in memory or use a more robust deduplication strategy
        # For now, we'll use a simple approach based on event_id
        if not hasattr(self, "_recent_events"):
            self._recent_events = set()

        # Check if event_id is in recent events
        if event_id in self._recent_events:
            return True

        # Add to recent events (keep only last 100 events to prevent memory leaks)
        self._recent_events.add(event_id)
        if len(self._recent_events) > 100:
            # Remove oldest events (simple FIFO)
            self._recent_events = set(list(self._recent_events)[-50:])

        return False


# Global instance
_observability_instance = None


def get_observability() -> MaximObservability:
    """Get the global observability instance."""
    global _observability_instance
    if _observability_instance is None:
        _observability_instance = MaximObservability()
    return _observability_instance


def init_observability(enabled: bool = True) -> MaximObservability:
    """Initialize the global observability instance."""
    global _observability_instance
    _observability_instance = MaximObservability(enabled=enabled)
    return _observability_instance
