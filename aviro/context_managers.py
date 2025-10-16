"""Context manager and decorator classes for Aviro SDK"""

import asyncio
import uuid
import warnings
from datetime import datetime

from .utils import _init_thread_local, _thread_local


class SpanDecoratorContextManager:
    """A class that can be used as both a decorator and context manager"""
    def __init__(self, aviro_instance, span_name: str, organization_name: str = None):
        self.aviro = aviro_instance
        self.span_name = span_name
        self.organization_name = organization_name
        self._context_manager = None

    def __call__(self, func):
        """When used as decorator"""
        def wrapper(*args, **kwargs):
            with self.aviro._create_span(self.span_name, self.organization_name) as span:
                return func(*args, **kwargs)
        return wrapper

    def __enter__(self):
        """When used as context manager"""
        self._context_manager = self.aviro._create_span(self.span_name, self.organization_name)
        return self._context_manager.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """When used as context manager"""
        if self._context_manager:
            return self._context_manager.__exit__(exc_type, exc_val, exc_tb)


class LoopDecoratorContextManager:
    """Use as a decorator or context manager to create a span+loop in one.

    Supports:
        - @observe.loop("agent_span")
        - with observe.loop("agent_span") as span:
    """
    def __init__(self, aviro_instance, loop_name: str):
        self.aviro = aviro_instance
        self.loop_name = loop_name
        self._span_cm = None
        self._loop_cm = None
        self._span = None

    def __call__(self, func):
        """Decorator usage: wraps function inside span + loop."""
        def wrapper(*args, **kwargs):
            with self.aviro._create_span(self.loop_name) as span:
                with span.loop(self.loop_name):
                    return func(*args, **kwargs)
        return wrapper

    def __enter__(self):
        """Context manager usage: enters span then loop, returns span."""
        self._span_cm = self.aviro._create_span(self.loop_name)
        span = self._span_cm.__enter__()
        self._loop_cm = span.loop(self.loop_name)
        self._loop_cm.__enter__()
        self._span = span
        return span

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exit loop first, then span
        if self._loop_cm:
            self._loop_cm.__exit__(exc_type, exc_val, exc_tb)
        if self._span_cm:
            return self._span_cm.__exit__(exc_type, exc_val, exc_tb)


class _AgentTracker:
    """Context manager and decorator for tracking LLM calls under a group scope (formerly agent).

    Note: agent_id is deprecated in favor of group_name. Internally we continue to store under
    tree["agents"][group_name] for backwards compatibility with existing UI/export code.
    """
    def __init__(self, agent_id: str, policy: str = "last_only", aviro_instance = None, organization_name: str = None):
        self.agent_id = agent_id
        self.group_name = agent_id
        self.policy = policy
        self.organization_name = organization_name
        self._aviro = aviro_instance
        self._span_created = False

    def _get_aviro(self):
        if self._aviro is None:
            # Use configured global instance; do not auto-create from environment
            from .utils import _global_aviro
            if _global_aviro is None:
                raise RuntimeError("Aviro client not configured. Call observe.configure(AviroClient(...)) before using observe().")
            self._aviro = _global_aviro
        return self._aviro

    def __enter__(self):
        _init_thread_local()
        # Enable observation
        _thread_local.observation_enabled = True

        # Get or create span and apply patches
        aviro = self._get_aviro()
        span = aviro._get_or_create_temp_span()

        # Ensure the span name reflects the agent id when using observe(...)
        # so that UI grouping by span_name matches the agent/run group name.
        try:
            if getattr(span, 'span_name', None) != self.agent_id:
                span.span_name = self.agent_id
                if hasattr(span, 'tree') and isinstance(span.tree, dict):
                    span.tree["span_name"] = self.agent_id
        except Exception:
            pass

        # Set organization_name on the span if provided
        if self.organization_name:
            span.organization_name = self.organization_name

        span._apply_http_patches()

        # Set up agent scope
        _thread_local.agent_stack.append(getattr(_thread_local, 'current_agent_id', None))
        _thread_local.current_agent_id = self.agent_id

        # Configure policy
        span._ensure_agent_bucket(self.agent_id)
        if isinstance(self.policy, dict) and 'window' in self.policy:
            span._agents[self.agent_id]['follow_policy'] = {"type": "window", "n": int(self.policy['window'])}
            span.tree['agents'][self.agent_id]['policy'] = {"type": "window", "n": int(self.policy['window'])}
        elif self.policy == 'last_only':
            span._agents[self.agent_id]['follow_policy'] = {"type": "last_only"}
            span.tree['agents'][self.agent_id]['policy'] = {"type": "last_only"}
        elif self.policy == 'fanout_all':
            span._agents[self.agent_id]['follow_policy'] = {"type": "fanout_all"}
            span.tree['agents'][self.agent_id]['policy'] = {"type": "fanout_all"}
        else:
            span._agents[self.agent_id]['follow_policy'] = {"type": "none"}
            span.tree['agents'][self.agent_id]['policy'] = {"type": "none"}

        return self

    def __exit__(self, exc_type, exc, tb):
        _init_thread_local()
        # Restore previous agent
        prev = _thread_local.agent_stack.pop() if _thread_local.agent_stack else None
        _thread_local.current_agent_id = prev

        # If no more agents in stack, finalize and submit
        if not _thread_local.agent_stack or not any(_thread_local.agent_stack):
            aviro = self._get_aviro()
            if aviro._temp_span:
                aviro._temp_span.finalize()
                if aviro.auto_submit:
                    aviro.finalize_span(aviro._temp_span)
                # Clear temp span so next observe() gets a fresh span
                aviro._temp_span = None
            # Disable observation
            _thread_local.observation_enabled = False

    def __call__(self, func):
        """Decorator support"""
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with self:
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)
            return sync_wrapper

    # Marker methods
    def follow(self, marker_id: str):
        aviro = self._get_aviro()
        span = aviro._get_or_create_temp_span()
        span._ensure_agent_bucket(self.agent_id)
        span._agents[self.agent_id]['active_markers'].add(marker_id)

    def unfollow(self, marker_id: str):
        aviro = self._get_aviro()
        span = aviro._get_or_create_temp_span()
        span._ensure_agent_bucket(self.agent_id)
        span._agents[self.agent_id]['active_markers'].discard(marker_id)

    def expire_all(self):
        aviro = self._get_aviro()
        span = aviro._get_or_create_temp_span()
        span._ensure_agent_bucket(self.agent_id)
        span._agents[self.agent_id]['active_markers'] = set()

    def mark(self, text: str, tag: str = None) -> str:
        aviro = self._get_aviro()
        span = aviro._get_or_create_temp_span()
        span._ensure_agent_bucket(self.agent_id)
        marker_id = f"m_{uuid.uuid4().hex[:12]}"
        created_by = getattr(span, 'current_call_record', {}).get('call_id') if getattr(span, 'current_call_record', None) else None
        span.tree['agents'][self.agent_id]['markers'][marker_id] = {
            'marker_id': marker_id,
            'tag': tag,
            'content_length': len(text) if text is not None else 0,
            'created_by_call': created_by,
            'created_at': datetime.now().isoformat()
        }
        return marker_id

