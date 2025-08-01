"""
Observability helper functions for common operations.

This module provides reusable functions for common observability patterns
used throughout the gpt-engineer codebase.
"""

from contextlib import contextmanager
from typing import Any, Dict, Optional

from gpt_engineer.core.preprompts_holder import PrepromptsHolder

# Import observability
try:
    from gpt_engineer.core.maxim_observability import get_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


@contextmanager
def preprompt_collection_span(
    preprompts_holder: PrepromptsHolder,
    parent_span_id: Optional[str] = None,
    operation_name: str = "Preprompt Collection",
) -> Dict[str, Any]:
    """
    Context manager for preprompt collection observability.

    This function handles the creation and cleanup of preprompt collection spans
    with consistent metadata across different operations.

    Parameters
    ----------
    preprompts_holder : PrepromptsHolder
        The preprompts holder instance to collect file information from.
    parent_span_id : Optional[str], optional
        The ID of the parent span, by default None.
    operation_name : str, optional
        The name for the span operation, by default "Preprompt Collection".

    Yields
    ------
    Dict[str, Any]
        A dictionary containing observability information:
        - 'observability': The observability instance if available
        - 'span_id': The span ID if created
        - 'preprompts': The collected preprompts
        - 'file_names': List of preprompt file names
        - 'file_count': Number of preprompt files
    """
    observability = None
    span_id = None
    preprompt_file_names = []
    preprompt_file_count = 0
    preprompts = {}

    if OBSERVABILITY_AVAILABLE:
        try:
            observability = get_observability()
            if observability and observability.is_enabled():
                from uuid import uuid4

                # Collect preprompt file names and count before reading contents
                preprompt_file_names = list(preprompts_holder.get_file_names())
                preprompt_file_count = len(preprompt_file_names)
                span_id = str(uuid4())

                start_span_kwargs = dict(
                    span_id=span_id,
                    name=operation_name,
                    tags={"operation": "preprompt_collection"},
                    metadata={
                        "preprompt_file_count": preprompt_file_count,
                        "preprompt_file_names": preprompt_file_names,
                    },
                )

                if parent_span_id:
                    start_span_kwargs["parent_span_id"] = parent_span_id

                observability.start_span(**start_span_kwargs)
        except Exception:
            pass  # Continue without observability

    try:
        # Suppress log_retrievals for preprompt file reads
        preprompts = preprompts_holder.get_preprompts(suppress_observability=True)

        yield {
            "observability": observability,
            "span_id": span_id,
            "preprompts": preprompts,
            "file_names": preprompt_file_names,
            "file_count": preprompt_file_count,
        }

    finally:
        if observability and observability.is_enabled() and span_id:
            try:
                observability.end_span(span_id=span_id)
            except Exception:
                pass  # Continue even if span ending fails


def create_preprompt_collection_span(
    preprompts_holder: PrepromptsHolder,
    parent_span_id: Optional[str] = None,
    operation_name: str = "Preprompt Collection",
) -> Dict[str, Any]:
    """
    Create a preprompt collection span and return observability info.

    This is a simpler alternative to the context manager for cases where
    you need more control over the span lifecycle.

    Parameters
    ----------
    preprompts_holder : PrepromptsHolder
        The preprompts holder instance to collect file information from.
    parent_span_id : Optional[str], optional
        The ID of the parent span, by default None.
    operation_name : str, optional
        The name for the span operation, by default "Preprompt Collection".

    Returns
    -------
    Dict[str, Any]
        A dictionary containing observability information:
        - 'observability': The observability instance if available
        - 'span_id': The span ID if created
        - 'file_names': List of preprompt file names
        - 'file_count': Number of preprompt files
    """
    observability = None
    span_id = None
    preprompt_file_names = []
    preprompt_file_count = 0

    if OBSERVABILITY_AVAILABLE:
        try:
            observability = get_observability()
            if observability and observability.is_enabled():
                from uuid import uuid4

                # Collect preprompt file names and count before reading contents
                preprompt_file_names = list(preprompts_holder.get_file_names())
                preprompt_file_count = len(preprompt_file_names)
                span_id = str(uuid4())

                start_span_kwargs = dict(
                    span_id=span_id,
                    name=operation_name,
                    tags={"operation": "preprompt_collection"},
                    metadata={
                        "preprompt_file_count": preprompt_file_count,
                        "preprompt_file_names": preprompt_file_names,
                    },
                )

                if parent_span_id:
                    start_span_kwargs["parent_span_id"] = parent_span_id

                observability.start_span(**start_span_kwargs)
        except Exception:
            pass  # Continue without observability

    return {
        "observability": observability,
        "span_id": span_id,
        "file_names": preprompt_file_names,
        "file_count": preprompt_file_count,
    }


def end_preprompt_collection_span(span_info: Dict[str, Any]) -> None:
    """
    End a preprompt collection span.

    Parameters
    ----------
    span_info : Dict[str, Any]
        The span information dictionary returned by create_preprompt_collection_span.
    """
    observability = span_info.get("observability")
    span_id = span_info.get("span_id")

    if observability and observability.is_enabled() and span_id:
        try:
            observability.end_span(span_id=span_id)
        except Exception:
            pass  # Continue even if span ending fails
