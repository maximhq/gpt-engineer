from typing import Any
from uuid import uuid4

from langchain_community.cache import SQLiteCache

try:
    from gpt_engineer.core.maxim_observability import get_observability

    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


class ObservedSQLiteCache(SQLiteCache):
    def lookup(self, prompt: Any, llm_string: str) -> Any:
        result = super().lookup(prompt, llm_string)
        cache_status = "hit" if result is not None else "miss"
        if OBSERVABILITY_AVAILABLE:
            try:
                observability = get_observability()
                if observability.is_enabled():
                    observability.log_event(
                        event_id=str(uuid4()),
                        event_type=f"cache_{cache_status}",
                        metadata={
                            "llm_string": llm_string,
                            "prompt_repr": repr(prompt),
                            "cache_backend": "ObservedSQLiteCache",
                        },
                        tags={
                            "component": "llm_cache",
                            "status": cache_status,
                        },
                    )
            except Exception:
                pass
        return result

    def update(self, prompt: Any, llm_string: str, return_val: Any) -> None:
        # Optionally, you could log cache insertions here as well
        return super().update(prompt, llm_string, return_val)
