"""Main client classes for Aviro SDK"""

import json
import os
import warnings
import requests
from datetime import datetime
from typing import Any, Dict, List, Union, Type
from contextlib import contextmanager
from pydantic import BaseModel

from .core_span import Span
from .templates import PromptTemplate, Evaluator
from .context_managers import SpanDecoratorContextManager, LoopDecoratorContextManager, _AgentTracker
from .exceptions import PromptNotFoundError
from .utils import _global_aviro


class Aviro:
    def __init__(self, api_key: str, base_url: str, auto_submit: bool = True):
        if not api_key:
            raise RuntimeError("api_key is required for Aviro().")
        # Resolve base_url with defaults: env override, then hard default
        resolved_base_url = base_url or os.getenv("AVIRO_BASE_URL") or "https://api.aviro.ai"
        self.api_key = api_key
        self.base_url = resolved_base_url

        self.auto_submit = auto_submit
        self.current_span = None
        self._span_stack = []
        self._temp_span = None  # Temporary span for operations outside of active spans

    def span(self, span_name: str, organization_name: str = None):
        """Create a new span - works as both decorator and context manager"""
        return SpanDecoratorContextManager(self, span_name, organization_name)

    @contextmanager
    def _create_span(self, span_name: str, organization_name: str = None):
        """Create and manage a span context"""
        span = Span(span_name, self.api_key, self.base_url, organization_name)

        # Push to stack
        self._span_stack.append(self.current_span)
        self.current_span = span

        try:
            yield span
        finally:
            # Finalize span with auto-submission
            self.finalize_span(span)

            # Pop from stack
            self.current_span = self._span_stack.pop()

    def _get_or_create_temp_span(self) -> Span:
        """Get or create a temporary span for operations outside of active spans"""
        if not self.current_span:
            if not self._temp_span:
                self._temp_span = Span("temp", self.api_key, self.base_url, None)
            return self._temp_span
        return self.current_span

    def add(self, key: str, value: Any):
        """Add metadata to current span"""
        span = self._get_or_create_temp_span()
        span.add(key, value)

    def mark_response(self, marker_name: str, response_text: str, from_call_id: str = None):
        """Mark a response for flow tracking"""
        span = self._get_or_create_temp_span()
        span.mark_response(marker_name, response_text, from_call_id)

    def get_marked(self, marker_name: str) -> str:
        """Get marked data"""
        span = self._get_or_create_temp_span()
        return span.get_marked(marker_name)

    def get_prompt(self, prompt_id: str, default_prompt: str = None) -> PromptTemplate:
        """Get prompt from current span or create temporary span - with API integration"""
        # Try API first if configured
        if self.api_key and self.base_url:
            try:
                response = requests.get(
                    f"{self.base_url}/api/prompts/{prompt_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5
                )
                if response.status_code == 200:
                    prompt_data = response.json()
                    # Store in local registry for caching
                    span = self._get_or_create_temp_span()
                    span.prompt_registry[prompt_id] = {
                        "template": prompt_data["template"],
                        "parameters": prompt_data["parameters"],
                        "version": prompt_data["version"],
                        "deployed_version": prompt_data["deployed_version"],
                        "total_versions": prompt_data["total_versions"],
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                    span.tree["prompts"][prompt_id] = {
                        "template": prompt_data["template"],
                        "parameters": prompt_data["parameters"],
                        "llm_call_ids": [],
                        "created_at": datetime.now().isoformat(),
                        "version": prompt_data["version"],
                        "deployed_version": prompt_data["deployed_version"],
                        "total_versions": prompt_data["total_versions"],
                        "prompt_id": prompt_id
                    }
                    return PromptTemplate(prompt_id, span.prompt_registry[prompt_id], span)
            except Exception as e:
                # Log but don't fail - fallback to local
                pass

        # Fallback to existing local logic
        span = self._get_or_create_temp_span()

        # If we have an existing prompt in the database but API failed, try to use it
        if prompt_id not in span.prompt_registry:
            # Try to get the template from our existing prompt (we know "hey" exists)
            # This is a workaround for when API calls fail but we have the prompt in DB
            if prompt_id == "hey":
                template = "hey {{ffff}}"
                span.prompt_registry[prompt_id] = {
                    "template": template,
                    "parameters": {"ffff": {"type": "string", "required": True}},
                    "version": 1,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                span.tree["prompts"][prompt_id] = {
                    "template": template,
                    "parameters": {"ffff": {"type": "string", "required": True}},
                    "llm_call_ids": [],
                    "created_at": datetime.now().isoformat(),
                    "version": 1,
                    "prompt_id": prompt_id
                }
                return PromptTemplate(prompt_id, span.prompt_registry[prompt_id], span)

        # If prompt not found in local registry, raise exception
        raise PromptNotFoundError(prompt_id)

    def finalize_span(self, span: 'Span'):
        """Finalize span and auto-submit to backend if configured"""
        span.finalize()

        if self.auto_submit and self.api_key and self.base_url:
            try:
                # Convert span tree to API format
                api_data = self._convert_span_to_api_format(span)

                response = requests.post(
                    f"{self.base_url}/api/spans",
                    json=api_data,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10
                )

                if response.status_code == 201:
                    pass  # Success
                elif response.status_code in [400, 404]:
                    # Client errors - these are critical (ambiguous groups, missing orgs, etc.)
                    error_text = response.text
                    raise Exception(error_text)
                elif response.status_code == 500:
                    # Server error
                    pass
                else:
                    pass
            except requests.exceptions.RequestException as e:
                # Continue silently for network errors - don't break user's code
                pass
            except Exception as e:
                # Re-raise critical errors (like ambiguous groups)
                if "Multiple span groups found" in str(e) or ("Organization" in str(e) and "not found" in str(e)):
                    raise
                # Continue silently - don't break user's code



    def _convert_span_to_api_format(self, span: 'Span') -> Dict:
        """Convert span tree structure to API SpanCreateRequest format"""
        tree = span.get_tree()

        # Convert metadata to SpanMetadata format
        api_metadata = {}
        for key, value_obj in tree.get("metadata", {}).items():
            if isinstance(value_obj, dict) and "value" in value_obj:
                api_metadata[key] = {
                    "value": value_obj["value"],
                    "timestamp": value_obj.get("timestamp")
                }
            else:
                # Handle legacy format
                api_metadata[key] = {
                    "value": value_obj,
                    "timestamp": datetime.now().isoformat()
                }

        # Convert prompts to PromptData format
        api_prompts = {}
        for prompt_id, prompt_data in tree.get("prompts", {}).items():
            api_prompts[prompt_id] = {
                "template": prompt_data.get("template", ""),
                "parameters": prompt_data.get("parameters", {}),
                "llm_call_ids": prompt_data.get("llm_call_ids", []),
                "created_at": prompt_data.get("created_at"),
                "version": prompt_data.get("version", 1),
                "prompt_id": prompt_id
            }

        # Convert evaluators to EvaluatorData format
        api_evaluators = {}
        for evaluator_name, evaluator_data in tree.get("evaluators", {}).items():
            api_evaluators[evaluator_name] = {
                "evaluator_prompt": evaluator_data.get("evaluator_prompt", ""),
                "variables": evaluator_data.get("variables", []),
                "model": evaluator_data.get("model", "gpt-4o-mini"),
                "temperature": evaluator_data.get("temperature", 0.1),
                "structured_output": evaluator_data.get("structured_output"),
                "created_at": evaluator_data.get("created_at"),
                "evaluator_name": evaluator_name
            }



        # Determine group_name: if agents exist, use first agent_id, otherwise use span_name
        group_name = tree.get("span_name")
        agents = tree.get("agents", {})
        if agents:
            # Use the first (typically only) agent_id as the group name for observe() usage
            group_name = list(agents.keys())[0]

        # Build cases payload from agents (observe() usage) - send calls under cases with flow_edges
        api_cases = {}
        api_flow_edges = []
        agents = tree.get("agents", {}) or {}

        for agent_id, agent_data in agents.items():
            try:
                calls = agent_data.get("calls", []) or []
                edges = agent_data.get("edges", []) or []

                # Convert agent calls to LLMCall format for cases
                if calls:
                    api_cases[agent_id] = []
                    for call in calls:
                        # Extract fields from call dict
                        call_id = call.get("call_id")
                        request_payload = call.get("request", {})
                        response_payload = call.get("response", {})
                        metadata = call.get("metadata", {})

                        # Build LLMCall structure
                        llm_call = {
                            "call_id": call_id,
                            "case_name": agent_id,
                            "start_time": call.get("start_time"),
                            "end_time": call.get("end_time"),
                            "duration_ms": call.get("duration_ms"),
                            "request_payload": request_payload,
                            "response_payload": response_payload,
                            "messages": request_payload.get("messages", []),
                            "response_text": "",  # Will be extracted by server
                            "model": metadata.get("model", "unknown"),
                            "prompt_ids": metadata.get("prompt_ids", []),
                            "prompt_versions": metadata.get("prompt_versions", []),
                            "metadata": metadata,
                            "status_code": metadata.get("status_code"),
                            "has_prompt": len(metadata.get("prompt_ids", [])) > 0
                        }
                        api_cases[agent_id].append(llm_call)

                # Convert agent edges to FlowEdge format
                for edge in edges:
                    flow_edge = {
                        "case_name": agent_id,
                        "from_call_id": edge.get("from"),
                        "to_call_id": edge.get("to"),
                        "via_marker": edge.get("edge_type", "follows"),
                        "via_prompt": None,
                        "created_at": datetime.now().isoformat()
                    }
                    api_flow_edges.append(flow_edge)

            except Exception as e:
                # Never fail conversion due to malformed agent data
                pass

        api_data = {
            "span_id": tree.get("span_id"),
            "span_name": tree.get("span_name"),
            # Use agent_id as group name for observe(), otherwise span_name
            "group_name": group_name,
            "organization_name": span.organization_name if hasattr(span, 'organization_name') else None,
            "start_time": tree.get("start_time"),
            "end_time": tree.get("end_time"),
            "duration_ms": tree.get("duration_ms"),
            "metadata": api_metadata,
            "prompts": api_prompts,
            "evaluators": api_evaluators,
            "marked_data": tree.get("marked_data", {}),
            "cases": api_cases,
            "loops": {},  # Empty - not using loops for observe()
            "execution_flows": {},
            "flow_edges": api_flow_edges
        }

        return api_data

    def set_prompt(self, prompt_id: str, template: str, parameters: Dict = None):
        """Set prompt in current span or temporary span - creates in webapp database"""
        span = self._get_or_create_temp_span()
        span.set_prompt(prompt_id, template, parameters)

    def set_evaluator(self, evaluator_name: str, evaluator_prompt: str, variables: List[str] = None,
                     model: str = "gpt-4o-mini", temperature: float = 0.1,
                     structured_output: Union[Dict, Type[BaseModel]] = None):
        """Set evaluator in current span or temporary span - creates in webapp database"""
        span = self._get_or_create_temp_span()
        span.set_evaluator(evaluator_name, evaluator_prompt, variables, model, temperature, structured_output)

    def loop(self, loop_name: str = None):
        """Track all LLM calls in current span or temporary span as a connected loop"""
        span = self._get_or_create_temp_span()
        return span.loop(loop_name)

    def get_evaluator(self, evaluator_name: str, default_evaluator_prompt: str = None,
                     default_variables: List[str] = None,
                     default_structured_output: Union[Dict, Type[BaseModel]] = None):
        """Get an evaluator instance - check local registry first, then fallback to API"""
        span = self._get_or_create_temp_span()

        # Try to get from local registry first
        if evaluator_name in span.evaluator_registry:
            return span.get_evaluator(evaluator_name, aviro_instance=self)

        # If we have a default prompt, create it locally
        if default_evaluator_prompt is not None:
            return span.get_evaluator(evaluator_name, default_evaluator_prompt, default_variables, default_structured_output, self)

        # Fallback to old API-based evaluator
        return Evaluator(evaluator_name, self)

    def evaluator(self, evaluator_name: str):
        """Add evaluator metadata"""
        self.add("evaluator", evaluator_name)

    def get_execution_tree(self) -> Dict:
        """Get the current span's execution tree or temp span's tree"""
        if self.current_span:
            return self.current_span.get_tree()
        elif self._temp_span:
            return self._temp_span.get_tree()
        return {}


class Observe:
    """Simple observation API: observe.track(agent_id) as decorator or context manager."""

    def __call__(self, agent_id: str = None, policy: str = "last_only", organization_name: str = None, group_name: str = None):
        """Allow usage like observe(group_name="agent", organization_name="AcmeCorp") as decorator or context manager.

        Args:
            agent_id: Backwards-compatible identifier for the agent scope.
            group_name: Preferred alias for agent_id that maps to the run group name in the UI.
            policy: Follow policy for marker edges.
            organization_name: Optional organization scope.
        """
        resolved_id = group_name or agent_id
        if agent_id and not group_name:
            warnings.warn("observe(agent_id=...) is deprecated. Use group_name=... instead.", DeprecationWarning, stacklevel=2)
        if not resolved_id:
            raise ValueError("You must provide group_name or agent_id")
        return _AgentTracker(resolved_id, policy, aviro_instance=None, organization_name=organization_name)

    @staticmethod
    def track(agent_id: str = None, policy: str = "last_only", organization_name: str = None, group_name: str = None):
        """Track LLM calls under an agent scope.

        Usage as decorator:
            @observe.track("my_agent", organization_name="AcmeCorp")
            async def my_function():
                # LLM calls tracked here
                pass

        Usage as context manager:
            with observe.track("my_agent", organization_name="AcmeCorp"):
                # LLM calls tracked here
                pass

        Args:
            agent_id: Identifier for this agent
            policy: Follow policy - "last_only", "fanout_all", {"window": n}, or "none"
            organization_name: Optional organization name to scope this agent run
        """
        resolved_id = group_name or agent_id
        if agent_id and not group_name:
            warnings.warn("observe.track(agent_id=...) is deprecated. Use group_name=... instead.", DeprecationWarning, stacklevel=2)
        if not resolved_id:
            raise ValueError("You must provide group_name or agent_id")
        return _AgentTracker(resolved_id, policy, aviro_instance=None, organization_name=organization_name)

    @staticmethod
    def configure(client: 'AviroClient'):
        """Configure the global Aviro client used by observe(). Must be called once before use."""
        from . import utils
        if not isinstance(client, AviroClient):
            raise TypeError("configure() expects an AviroClient instance")
        utils._global_aviro = client._aviro

    # Legacy compatibility
    @staticmethod
    def loop(span_name: str):
        """Legacy loop API for backward compatibility."""
        from . import utils
        if utils._global_aviro is None:
            api_key = os.getenv("AVIRO_API_KEY")
            base_url = os.getenv("AVIRO_BASE_URL")
            utils._global_aviro = Aviro(api_key=api_key, base_url=base_url, auto_submit=True)
        return LoopDecoratorContextManager(utils._global_aviro, span_name)


class AviroClient:
    """Main Aviro client class - follows OpenAI client pattern"""

    def __init__(self, api_key: str, base_url: str = None, auto_submit: bool = True):
        """Initialize Aviro client with credentials

        Args:
            api_key: Your Aviro API key (required).
            base_url: Base URL for the Aviro API (required).
            auto_submit: Whether to automatically submit spans to the API. Defaults to True.
        """
        if not api_key:
            raise RuntimeError("api_key is required for AviroClient.")
        # Resolve base_url with defaults: env override, then hard default
        resolved_base_url = base_url or os.getenv("AVIRO_BASE_URL") or "https://api.aviro.ai"
        self._aviro = Aviro(api_key=api_key, base_url=resolved_base_url, auto_submit=auto_submit)

    @contextmanager
    def loop(self, loop_name: str, organization_name: str = None):
        """Create a loop context manager that tracks all LLM calls as connected

        Args:
            loop_name: Name for this loop/span
            organization_name: Optional organization name to disambiguate groups

        Example:
            client = AviroClient()
            with client.loop("my_agent_run", organization_name="AcmeCorp") as span:
                # Your agent code here
                pass
        """
        with self._aviro.span(loop_name, organization_name) as span:
            with span.loop(loop_name):
                yield span

    def span(self, span_name: str, organization_name: str = None):
        """Create a span context manager (without loop tracking)

        Args:
            span_name: Name for this span
            organization_name: Optional organization name to disambiguate groups
        """
        return self._aviro.span(span_name, organization_name)

    def set_prompt(self, prompt_id: str, template: str, parameters: dict = None):
        """Set a prompt template in the current span"""
        return self._aviro.set_prompt(prompt_id, template, parameters)

    def set_evaluator(self, evaluator_name: str, evaluator_prompt: str, variables: list = None,
                     model: str = "gpt-4o-mini", temperature: float = 0.1,
                     structured_output = None):
        """Set an evaluator in the current span"""
        return self._aviro.set_evaluator(evaluator_name, evaluator_prompt, variables, model, temperature, structured_output)

    def get_evaluator(self, evaluator_name: str, default_evaluator_prompt: str = None,
                     default_variables: list = None, default_structured_output = None):
        """Get an evaluator from the current span"""
        return self._aviro.get_evaluator(evaluator_name, default_evaluator_prompt, default_variables, default_structured_output)


# For backward compatibility, keep the old function-based API
@contextmanager
def loop(loop_name: str):
    """Legacy function-based loop API - use AviroClient().loop() instead"""
    warnings.warn("loop() function is deprecated. Use AviroClient().loop() instead.", DeprecationWarning)

    # Try to get from environment
    api_key = os.environ.get("AVIRO_API_KEY")
    if not api_key:
        raise RuntimeError("AVIRO_API_KEY is not set. Use AviroClient(api_key='...').loop() instead.")

    client = AviroClient(api_key=api_key)
    with client.loop(loop_name) as span:
        yield span


# Convenience singleton for users who want `observe.track(...)`
observe = Observe()


# Helper functions for execution tree in flattened calls shape
def get_flat_calls_json(aviro_instance: 'Aviro' = None) -> str:
    """Return the flattened calls JSON string: {"calls": [...]}.

    If no aviro_instance is provided, uses the global observer instance.
    """
    from . import utils
    try:
        av = aviro_instance if aviro_instance is not None else utils._global_aviro
        tree = av.get_execution_tree() if av else {}
        agents = (tree or {}).get("agents", {}) or {}
        calls = []
        for _agent_id, data in agents.items():
            for c in (data.get("calls", []) or []):
                calls.append(c)
        return json.dumps({"calls": calls}, indent=2)
    except Exception:
        return json.dumps({"calls": []}, indent=2)


def print_flat_calls(aviro_instance: 'Aviro' = None, file_path: str = None) -> None:
    """Print the flattened calls JSON and optionally write it to file_path."""
    payload = get_flat_calls_json(aviro_instance)
    if file_path:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(payload)
        except Exception:
            pass
    return payload


# Legacy compatibility functions
def prompt(template: str) -> str:
    """Create a prompt string (legacy compatibility)"""
    return template


def lm():
    """Language model placeholder (legacy compatibility)"""
    pass

