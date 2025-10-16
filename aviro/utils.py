"""Utility functions and thread-local storage for Aviro SDK"""

import threading
import requests
import httpx

# Thread-local storage for patch management and prompt tracking
_thread_local = threading.local()

# Global singleton for automatic Aviro instance creation
_global_aviro = None

# Store original functions
_original_requests_session_send = requests.Session.send
_original_httpx_async_client_send = httpx.AsyncClient.send
_original_httpx_client_send = httpx.Client.send


def _init_thread_local():
    """Initialize thread-local storage if needed."""
    if not hasattr(_thread_local, 'patch_count'):
        _thread_local.patch_count = 0
    if not hasattr(_thread_local, 'httpx_patch_count'):
        _thread_local.httpx_patch_count = 0
    if not hasattr(_thread_local, 'pending_compiled_prompts'):
        _thread_local.pending_compiled_prompts = []
    if not hasattr(_thread_local, 'current_span_instance'):
        _thread_local.current_span_instance = None
    if not hasattr(_thread_local, 'original_messages_context'):
        _thread_local.original_messages_context = None
    if not hasattr(_thread_local, 'observation_enabled'):
        _thread_local.observation_enabled = False
    if not hasattr(_thread_local, 'current_agent_id'):
        _thread_local.current_agent_id = None
    if not hasattr(_thread_local, 'agent_stack'):
        _thread_local.agent_stack = []


def get_current_span():
    """Get the current active span instance (if any)."""
    _init_thread_local()
    return getattr(_thread_local, 'current_span_instance', None)


class MarkedResponse(str):
    """A string subclass that carries Aviro marker metadata and can self-mark.

    Usage:
        resp = await llm.ask(...)
        resp = resp.mark_response("marker_name")
        await llm.ask([{"role": "user", "content": resp}], stream=False)
    """

    def __new__(cls, text: str, marker_name: str = None):
        obj = str.__new__(cls, text)
        # Store optional marker name; will be set on mark_response if not provided
        obj.marker_name = marker_name
        return obj

    def mark_response(self, marker_name: str = None) -> 'MarkedResponse':
        """Mark this response on the current span and return self for chaining.

        This records the producing call id programmatically (no string matching).
        """
        span = get_current_span()
        if not span:
            return self

        import uuid
        name_to_use = marker_name or (self.marker_name or f"marked_{uuid.uuid4().hex[:8]}")
        self.marker_name = name_to_use

        # Use the most recent call id (the call that produced this response)
        from_call_id = None
        if getattr(span, "current_call_record", None):
            from_call_id = span.current_call_record.get("call_id")

        span.mark_response(name_to_use, str(self), from_call_id=from_call_id)
        return self

