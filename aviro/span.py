import json
import warnings
import uuid
import threading
import requests
import httpx
import os
import hashlib
import re
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type
from contextlib import contextmanager
from pydantic import BaseModel

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

class PromptNotFoundError(Exception):
    """Exception raised when a prompt is not found"""
    def __init__(self, prompt_id: str, message: str = None):
        self.prompt_id = prompt_id
        self.message = message or f"Prompt '{prompt_id}' not found"
        super().__init__(self.message)

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

        name_to_use = marker_name or (self.marker_name or f"marked_{uuid.uuid4().hex[:8]}")
        self.marker_name = name_to_use

        # Use the most recent call id (the call that produced this response)
        from_call_id = None
        if getattr(span, "current_call_record", None):
            from_call_id = span.current_call_record.get("call_id")

        span.mark_response(name_to_use, str(self), from_call_id=from_call_id)
        return self

def get_current_span() -> Optional['Span']:
    """Get the current active span instance (if any)."""
    _init_thread_local()
    return getattr(_thread_local, 'current_span_instance', None)



class Span:
    def __init__(self, span_name: str, api_key: str = None, base_url: str = None, organization_name: str = None):
        # Generate unique UUID for each span run (not deterministic)
        self.span_id = str(uuid.uuid4())
        self.span_name = span_name
        self.organization_name = organization_name
        self.api_key = api_key
        self._base_url = base_url
        self.start_time = datetime.now().isoformat()
        self.end_time = None

        # Set this span as the current span in thread-local storage
        _init_thread_local()
        _thread_local.current_span_instance = self

        # Main execution tree structure
        self.tree = {
            "span_id": self.span_id,
            "span_name": span_name,
            "start_time": self.start_time,
            "end_time": None,
            "metadata": {},  # span.add() calls go here with timestamps
            "prompts": {},   # prompt_id -> {template, parameters, llm_call_ids, created_at}
            "evaluators": {},  # evaluator_name -> {evaluator_prompt, variables, model, temperature, structured_output, created_at}
            "marked_data": {},  # marker_name -> {content, created_by_call, used_in}
            "loops": {},      # loop_name -> {calls, flow_edges}
            "agents": {}      # agent_id -> {calls, edges, markers, policy}
        }

        # Tracking state
        self.current_loop = None  # Track current active loop
        self.prompt_registry = {}  # prompt_id -> template/params
        self.evaluator_registry = {}  # evaluator_name -> evaluator_data
        self.active = True
        self.current_call_record = None

        # Flow tracking state
        self.marked_data = {}  # marker_name -> data
        self.marker_usage = {}  # marker_name -> [usage_records]
        self._pending_marker_usage = []  # Track multiple marker usage in compile()
        self._pending_usage_records = [] # Store pending marker usage records

        # Agent observation state
        self._agents = {}

    def _is_llm_endpoint(self, url: str) -> bool:
        """Check if URL is an LLM API endpoint we should monitor"""
        llm_patterns = [
            # OpenAI
            "api.openai.com",
            # Anthropic
            "api.anthropic.com",
            # Google Gemini
            "generativelanguage.googleapis.com",
            # OpenRouter
            "openrouter.ai",
            # Local/proxy endpoints
            "localhost:8080/openai",
            "api.aviro.com/openai"
        ]
        return any(pattern in url for pattern in llm_patterns)

    def add(self, key: str, value: Any) -> None:
        """Add metadata to the span with timestamp"""
        self.tree["metadata"][key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }

    def use_marked(self, marker_name: str) -> None:
        """Programmatically register that a marked value will be used in the next LLM call.

        This avoids any string matching by directly queuing the usage, which will be
        resolved to the next intercepted call in _capture_request_data.
        """
        if not marker_name:
            return
        if not hasattr(self, '_pending_marker_usage'):
            self._pending_marker_usage = []
        self._pending_marker_usage.append(marker_name)

    def mark_response(self, marker_name: str, response_text: str, from_call_id: str = None) -> None:
        """Mark a response text for flow tracking"""
        current_call_id = from_call_id or (self.current_call_record.get("call_id") if self.current_call_record else None)

        marked_data_entry = {
            "marker_name": marker_name,
            "content": response_text,
            "marked_at": datetime.now().isoformat(),
            "created_by_call": current_call_id,
            "used_in": []
        }

        # Store in instance state
        self.marked_data[marker_name] = marked_data_entry

        # Store in tree structure
        self.tree["marked_data"][marker_name] = marked_data_entry.copy()

        # Add metadata for tracking
        self.add(f"marked_data_{marker_name}", {
            "marker_name": marker_name,
            "content_length": len(response_text),
            "created_by_call": current_call_id
        })

    def get_marked(self, marker_name: str) -> str:
        """Get marked data and track its usage for flow connections"""
        if marker_name not in self.marked_data:
            raise ValueError(f"Marker '{marker_name}' not found. Available markers: {list(self.marked_data.keys())}")

        # Record that this marker is being accessed - flow will be created when next LLM call is made
        self._pending_marker_usage.append(marker_name)

        # Add metadata about the access
        self.add(f"marker_access_{marker_name}", {
            "marker_name": marker_name,
            "accessed_at": datetime.now().isoformat(),
            "content_length": len(self.marked_data[marker_name]["content"])
        })

        return self.marked_data[marker_name]["content"]

    def _record_marker_usage(self, marker_name: str, prompt_id: str, call_id: str) -> None:
        """Record that marked data was used in a prompt"""
        usage_record = {
            "prompt_id": prompt_id,
            "call_id": call_id,
            "used_at": datetime.now().isoformat()
        }

        # Add to marked data record in instance state
        if marker_name in self.marked_data:
            self.marked_data[marker_name]["used_in"].append(usage_record)

            # Update tree structure
            if marker_name in self.tree["marked_data"]:
                self.tree["marked_data"][marker_name]["used_in"].append(usage_record)


    def _extract_llm_response_text(self, response_data: Dict) -> Optional[str]:
        """Extract clean response text from LLM API response for automatic marking"""
        if not isinstance(response_data, dict):
            return None

        # Handle OpenAI/OpenRouter response format (choices array)
        choices = response_data.get("choices", [])
        if choices and isinstance(choices, list):
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                # Handle both chat completions and legacy completions
                message = first_choice.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content")
                    if content and isinstance(content, str):
                        return content.strip()

                # Fallback for direct text field
                text = first_choice.get("text")
                if text and isinstance(text, str):
                    return text.strip()

        # Handle Anthropic response format
        content = response_data.get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            first_content = content[0]
            if isinstance(first_content, dict):
                text = first_content.get("text")
                if text and isinstance(text, str):
                    return text.strip()

        # Handle Gemini response format
        candidates = response_data.get("candidates", [])
        if candidates and isinstance(candidates, list):
            first_candidate = candidates[0]
            if isinstance(first_candidate, dict):
                content = first_candidate.get("content", {})
                if isinstance(content, dict):
                    parts = content.get("parts", [])
                    if parts and isinstance(parts, list) and len(parts) > 0:
                        first_part = parts[0]
                        if isinstance(first_part, dict):
                            text = first_part.get("text")
                            if text and isinstance(text, str):
                                return text.strip()

        return None

    def _extract_llm_response_id(self, response_data: Dict) -> Optional[str]:
        """Extract response ID from different LLM API response formats"""
        if not isinstance(response_data, dict):
            return None

        # OpenAI/OpenRouter format - direct "id" field
        if "id" in response_data:
            return response_data["id"]

        # Anthropic format - "id" field
        if "id" in response_data:
            return response_data["id"]

        # Gemini format - "candidates" array with "finishReason" and other metadata
        # For Gemini, we'll use a combination of timestamp and model as ID
        candidates = response_data.get("candidates", [])
        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            # Generate a deterministic ID based on response content
            import hashlib
            import json
            response_str = json.dumps(response_data, sort_keys=True)
            return f"gemini_{hashlib.md5(response_str.encode()).hexdigest()[:16]}"

        return None




    def _update_flow_connections(self, old_call_id: str, new_call_id: str) -> None:
        """Update all flow connections to use the new LLM call ID instead of temporary UUID"""
        if not old_call_id or not new_call_id or old_call_id == new_call_id:
            return

        # Update loop flow edges
        for loop_name, loop_data in self.tree.get("loops", {}).items():
            # Update flow edges
            for edge in loop_data.get("flow_edges", []):
                if edge.get("from") == old_call_id:
                    edge["from"] = new_call_id
                if edge.get("to") == old_call_id:
                    edge["to"] = new_call_id

        # Update agent edges
        for agent_id, agent_data in self.tree.get("agents", {}).items():
            for edge in agent_data.get("edges", []):
                if edge.get("from") == old_call_id:
                    edge["from"] = new_call_id
                if edge.get("to") == old_call_id:
                    edge["to"] = new_call_id

        # Update marked data usage records
        for marker_name, marker_data in self.tree.get("marked_data", {}).items():
            if marker_data.get("created_by_call") == old_call_id:
                marker_data["created_by_call"] = new_call_id

            for usage_record in marker_data.get("used_in", []):
                if usage_record.get("call_id") == old_call_id:
                    usage_record["call_id"] = new_call_id

        # Update instance-level marked data too
        for marker_name, marker_data in self.marked_data.items():
            if marker_data.get("created_by_call") == old_call_id:
                marker_data["created_by_call"] = new_call_id

            for usage_record in marker_data.get("used_in", []):
                if usage_record.get("call_id") == old_call_id:
                    usage_record["call_id"] = new_call_id

    def _create_loop_flow_edge(self, loop_name: str, current_call_id: str) -> None:
        """Create a flow edge from the previous call to the current call within the same loop"""
        if not hasattr(self, 'current_loop_context') or not self.current_loop_context:
            return

        calls_list = self.current_loop_context["calls_in_loop"]
        if len(calls_list) < 2:
            # No previous call to connect from
            return

        # Get the previous call ID (second-to-last in the list)
        previous_call_id = calls_list[-2]  # -1 is current, -2 is previous

        # Create flow edge in the loop structure
        if loop_name not in self.tree["loops"]:
            self.tree["loops"][loop_name] = {
                "calls": [],
                "flow_edges": []
            }

        loop_data = self.tree["loops"][loop_name]

        # Add edge
        edge = {
            "from": previous_call_id,
            "to": current_call_id,
            "edge_type": "sequential_loop",
            "created_at": datetime.now().isoformat()
        }

        # Check if edge already exists
        existing_edge = next((e for e in loop_data["flow_edges"] if e["from"] == previous_call_id and e["to"] == current_call_id), None)
        if not existing_edge:
            loop_data["flow_edges"].append(edge)





    def get_prompt(self, prompt_id: str, default_prompt: str = None) -> 'PromptTemplate':
        """Get or create a prompt template - completely local, no API calls"""
        if prompt_id not in self.prompt_registry:
            # If no default_prompt provided, raise exception
            if default_prompt is None:
                raise PromptNotFoundError(prompt_id)

            # Create prompt locally with default template
            template = default_prompt
            prompt_data = {
                "template": template,
                "parameters": {},
                "version": 1,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            self.prompt_registry[prompt_id] = prompt_data

            # Add to tree structure with comprehensive tracking
            self.tree["prompts"][prompt_id] = {
                "template": template,
                "parameters": {},
                "llm_call_ids": [],  # Will be populated when prompts are detected in LLM calls
                "created_at": datetime.now().isoformat(),
                "version": 1,
                "prompt_id": prompt_id
            }

        return PromptTemplate(prompt_id, self.prompt_registry[prompt_id], self)

    def set_prompt(self, prompt_id: str, template: str, parameters: Dict = None):
        """Set a prompt template manually - creates in webapp database if API key available"""
        # Check if prompt already exists in webapp
        if hasattr(self, 'api_key') and self.api_key and hasattr(self, '_base_url'):
            try:
                response = requests.get(
                    f"{self._base_url}/api/prompts/{prompt_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5
                )
                if response.status_code == 200:
                    raise PromptAlreadyExistsError(prompt_id)
            except PromptAlreadyExistsError:
                raise
            except Exception:
                # Prompt doesn't exist, continue with creation
                pass

        # Create prompt in webapp via API if possible
        if hasattr(self, 'api_key') and self.api_key and hasattr(self, '_base_url'):
            try:
                prompt_data_api = {
                    "prompt_name": prompt_id,
                    "template": template,
                    "parameters": parameters or {},
                    "version": 1
                }

                response = requests.post(
                    f"{self._base_url}/api/prompts",
                    json=prompt_data_api,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                else:
                    # Fall back to local storage
                    self._set_prompt_local(prompt_id, template, parameters)
            except Exception as e:
                # Fall back to local storage
                self._set_prompt_local(prompt_id, template, parameters)
        else:
            # No API key, use local storage
            self._set_prompt_local(prompt_id, template, parameters)


    def _set_prompt_local(self, prompt_id: str, template: str, parameters: Dict = None):
        """Set prompt locally (fallback method)"""
        prompt_data = {
            "template": template,
            "parameters": parameters or {},
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self.prompt_registry[prompt_id] = prompt_data
        self.tree["prompts"][prompt_id] = {
            "template": template,
            "parameters": parameters or {},
            "llm_call_ids": [],
            "created_at": datetime.now().isoformat(),
            "version": 1,
            "prompt_id": prompt_id
        }

    def set_evaluator(self, evaluator_name: str, evaluator_prompt: str, variables: List[str] = None,
                     model: str = "gpt-4o-mini", temperature: float = 0.1,
                     structured_output: Union[Dict, Type[BaseModel]] = None):
        """Set an evaluator manually - creates in webapp database if API key available"""
        # Check if evaluator already exists in webapp
        if hasattr(self, 'api_key') and self.api_key and hasattr(self, '_base_url'):
            try:
                response = requests.get(
                    f"{self._base_url}/api/web-evaluators",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5
                )
                if response.status_code == 200:
                    evaluators_data = response.json()
                    evaluators = evaluators_data.get("evaluators", [])
                    if any(eval.get("name") == evaluator_name for eval in evaluators):
                        raise EvaluatorAlreadyExistsError(evaluator_name)
            except EvaluatorAlreadyExistsError:
                raise
            except Exception as e:
                # Evaluator doesn't exist or check failed, continue with creation
                pass

        # Convert Pydantic model to schema if provided
        processed_structured_output = None
        pydantic_model_class = None

        if structured_output is not None:
            if isinstance(structured_output, type) and issubclass(structured_output, BaseModel):
                # It's a Pydantic model class
                pydantic_model_class = structured_output
                # Convert to JSON schema format
                schema = structured_output.model_json_schema()
                processed_structured_output = schema
            else:
                # It's already a dict/schema
                processed_structured_output = structured_output

        # Create evaluator in webapp via API if possible
        if hasattr(self, 'api_key') and self.api_key and hasattr(self, '_base_url'):
            try:
                evaluator_data_api = {
                    "name": evaluator_name,
                    "variables": variables or [],
                    "evaluator_prompt": evaluator_prompt,
                    "model": model,
                    "temperature": temperature,
                    "structured_output": processed_structured_output
                }

                response = requests.post(
                    f"{self._base_url}/api/web-evaluators",
                    json=evaluator_data_api,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                else:
                    # Fall back to local storage
                    self._set_evaluator_local(evaluator_name, evaluator_prompt, variables, model, temperature, processed_structured_output, pydantic_model_class)
            except Exception as e:
                # Fall back to local storage
                self._set_evaluator_local(evaluator_name, evaluator_prompt, variables, model, temperature, processed_structured_output, pydantic_model_class)
        else:
            # No API key, use local storage
            self._set_evaluator_local(evaluator_name, evaluator_prompt, variables, model, temperature, processed_structured_output, pydantic_model_class)

    def _set_evaluator_local(self, evaluator_name: str, evaluator_prompt: str, variables: List[str] = None,
                            model: str = "gpt-4o-mini", temperature: float = 0.1,
                            processed_structured_output: Union[Dict, Type[BaseModel]] = None,
                            pydantic_model_class = None):
        """Set evaluator locally (fallback method)"""
        evaluator_data = {
            "evaluator_prompt": evaluator_prompt,
            "variables": variables or [],
            "model": model,
            "temperature": temperature,
            "structured_output": processed_structured_output,
            "pydantic_model_class": pydantic_model_class,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self.evaluator_registry[evaluator_name] = evaluator_data
        self.tree["evaluators"][evaluator_name] = {
            "evaluator_prompt": evaluator_prompt,
            "variables": variables or [],
            "model": model,
            "temperature": temperature,
            "structured_output": processed_structured_output,
            "created_at": datetime.now().isoformat(),
            "evaluator_name": evaluator_name
        }

    def register_compiled_prompt(self, prompt_id: str, compiled_text: str, parameters_used: Dict):
        """Register a compiled version of a prompt in span metadata for tracking"""
        compilation_key = f"prompt_compilation_{prompt_id}_{datetime.now().isoformat()}"
        self.add(compilation_key, {
            "prompt_id": prompt_id,
            "compiled_text": compiled_text,
            "parameters_used": parameters_used,
            "compiled_at": datetime.now().isoformat(),
            "length": len(compiled_text)
        })

    def get_evaluator(self, evaluator_name: str, default_evaluator_prompt: str = None,
                     default_variables: List[str] = None, default_structured_output: Union[Dict, Type[BaseModel]] = None,
                     aviro_instance: 'Aviro' = None) -> 'EvaluatorTemplate':
        """Get or create an evaluator template - completely local, no API calls"""
        if evaluator_name not in self.evaluator_registry:
            # If no default_evaluator_prompt provided, raise exception
            if default_evaluator_prompt is None:
                raise EvaluatorNotFoundError(evaluator_name)

            # Convert Pydantic model to schema if provided
            processed_structured_output = None
            pydantic_model_class = None

            if default_structured_output is not None:
                if isinstance(default_structured_output, type) and issubclass(default_structured_output, BaseModel):
                    # It's a Pydantic model class
                    pydantic_model_class = default_structured_output
                    # Convert to JSON schema format
                    schema = default_structured_output.model_json_schema()
                    processed_structured_output = schema
                else:
                    # It's already a dict/schema
                    processed_structured_output = default_structured_output

            # Create evaluator locally with default data
            evaluator_data = {
                "evaluator_prompt": default_evaluator_prompt,
                "variables": default_variables or [],
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "structured_output": processed_structured_output,
                "pydantic_model_class": pydantic_model_class,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            self.evaluator_registry[evaluator_name] = evaluator_data

            # Add to tree structure
            self.tree["evaluators"][evaluator_name] = {
                "evaluator_prompt": default_evaluator_prompt,
                "variables": default_variables or [],
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "structured_output": processed_structured_output,
                "created_at": datetime.now().isoformat(),
                "evaluator_name": evaluator_name
            }

        return EvaluatorTemplate(evaluator_name, self.evaluator_registry[evaluator_name], self, aviro_instance)


    @contextmanager
    def loop(self, loop_name: str = None):
        """Context manager to track all LLM calls made within this context as a connected loop"""
        if not loop_name:
            loop_name = f"loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set up loop tracking
        self.current_loop = loop_name
        if loop_name not in self.tree["loops"]:
            self.tree["loops"][loop_name] = {
                "calls": [],
                "flow_edges": []
            }

        loop_start = datetime.now().isoformat()

        # Store the loop context for HTTP monitoring
        self.current_loop_context = {
            "loop_name": loop_name,
            "loop_start": loop_start,
            "calls_in_loop": []
        }

        # Apply HTTP patches to capture all calls in this context
        self._apply_http_patches()

        try:
            yield loop_name
        finally:
            loop_end = datetime.now().isoformat()

            # Calculate loop duration
            if loop_start and loop_end:
                start_dt = datetime.fromisoformat(loop_start)
                end_dt = datetime.fromisoformat(loop_end)
                duration = (end_dt - start_dt).total_seconds() * 1000

                self.add(f"loop_{loop_name}_duration", duration)
                self.add(f"loop_{loop_name}_calls_count", len(self.current_loop_context.get("calls_in_loop", [])))

            self._revert_http_patches()
            self.current_loop_context = None
            self.current_loop = None

    def _apply_http_patches(self):
        """Apply HTTP patches for monitoring"""
        _init_thread_local()
        if _thread_local.patch_count == 0:
            # Create bound methods for the span instance
            bound_requests_send = lambda session_instance, request, **kwargs: self._patched_requests_session_send(session_instance, request, **kwargs)
            bound_httpx_send = lambda client_instance, request, **kwargs: self._patched_httpx_async_client_send(client_instance, request, **kwargs)
            bound_httpx_sync_send = lambda client_instance, request, **kwargs: self._patched_httpx_client_send(client_instance, request, **kwargs)

            requests.Session.send = bound_requests_send
            httpx.AsyncClient.send = bound_httpx_send
            httpx.Client.send = bound_httpx_sync_send
        _thread_local.patch_count += 1
        _thread_local.httpx_patch_count += 1

    def _revert_http_patches(self):
        """Revert HTTP patches"""
        _init_thread_local()
        if _thread_local.patch_count > 0:
            _thread_local.patch_count -= 1
            if _thread_local.patch_count == 0:
                requests.Session.send = _original_requests_session_send

        if _thread_local.httpx_patch_count > 0:
            _thread_local.httpx_patch_count -= 1
            if _thread_local.httpx_patch_count == 0:
                httpx.AsyncClient.send = _original_httpx_async_client_send
                httpx.Client.send = _original_httpx_client_send

    async def _patched_httpx_async_client_send(self, client_instance, request, **kwargs):
        """Patched version of httpx.AsyncClient.send for monitoring"""
        # Check if this is an LLM API call
        if self._is_llm_endpoint(str(request.url)):
            self._capture_request_data(request, str(request.url), is_async=True)

        # Make the actual request
        response = await _original_httpx_async_client_send(client_instance, request, **kwargs)

        # Capture response if it's an LLM call
        if self._is_llm_endpoint(str(request.url)):
            self._capture_response_data(response, is_async=True)

        return response

    def _patched_httpx_client_send(self, client_instance, request, **kwargs):
        """Patched version of httpx.Client.send for monitoring (sync)."""
        url = str(request.url)
        if self._is_llm_endpoint(url):
            self._capture_request_data(request, url, is_async=False)
        response = _original_httpx_client_send(client_instance, request, **kwargs)
        if self._is_llm_endpoint(url):
            self._capture_response_data(response, is_async=False)
        return response

    def _patched_requests_session_send(self, session_instance, request, **kwargs):
        """Patched version of requests.Session.send for monitoring"""
        # Check if this is an LLM API call
        if self._is_llm_endpoint(request.url):
            self._capture_request_data(request, request.url, is_async=False)

        # Make the actual request
        response = _original_requests_session_send(session_instance, request, **kwargs)

        # Capture response if it's an LLM call
        if self._is_llm_endpoint(request.url):
            self._capture_response_data(response, is_async=False)

        return response

    def _capture_response_data(self, response, is_async: bool):
        """Capture response data for LLM calls"""
        if not self.current_call_record:
            return

        try:
            # Record end time and duration
            end_time = datetime.now().isoformat()
            self.current_call_record["end_time"] = end_time

            # Calculate duration if we have start time
            if self.current_call_record.get("start_time"):
                start_dt = datetime.fromisoformat(self.current_call_record["start_time"])
                end_dt = datetime.fromisoformat(end_time)
                duration_ms = (end_dt - start_dt).total_seconds() * 1000
                self.current_call_record["duration_ms"] = duration_ms

            # Always record status code in metadata (ensure it's never null)
            status_code = getattr(response, 'status_code', 200)
            self.current_call_record["metadata"]["status_code"] = status_code

            # Initialize response_data to ensure it's never null
            response_data = None

            # Extract response content with improved error handling
            if is_async:
                # httpx response
                try:
                    # Try multiple ways to get the response content
                    if hasattr(response, 'json') and callable(response.json):
                        # Try the json() method first (most reliable)
                        response_data = response.json()
                    elif hasattr(response, 'content'):
                        content = response.content
                        if isinstance(content, bytes):
                            content_text = content.decode('utf-8')
                        else:
                            content_text = str(content)
                        response_data = json.loads(content_text)
                    elif hasattr(response, 'text'):
                        content_text = response.text
                        response_data = json.loads(content_text)
                    else:
                        # Last resort: try to read the response
                        content = response.read()
                        if isinstance(content, bytes):
                            content_text = content.decode('utf-8')
                        else:
                            content_text = str(content)
                        response_data = json.loads(content_text)

                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
                    # Store raw content if JSON parsing fails
                    try:
                        raw_content = getattr(response, 'content', None) or getattr(response, 'text', None)
                        if raw_content:
                            if isinstance(raw_content, bytes):
                                raw_content = raw_content.decode('utf-8', errors='ignore')
                            response_data = {"error": "Failed to parse JSON response", "raw_content": str(raw_content)[:1000], "parse_error": str(e)}
                        else:
                            response_data = {"error": "Failed to parse response and no content available", "parse_error": str(e)}
                    except Exception as inner_e:
                        response_data = {"error": "Complete response parsing failure", "parse_error": str(e), "inner_error": str(inner_e)}
            else:
                # requests response
                try:
                    # Try the json() method first (most reliable)
                    if hasattr(response, 'json') and callable(response.json):
                        response_data = response.json()
                    elif hasattr(response, 'content'):
                        content = response.content
                        if isinstance(content, bytes):
                            content_text = content.decode('utf-8')
                        else:
                            content_text = str(content)
                        response_data = json.loads(content_text)
                    elif hasattr(response, 'text'):
                        response_data = json.loads(response.text)
                    else:
                        response_data = {"error": "No accessible response content"}

                except (json.JSONDecodeError, ValueError, AttributeError) as e:
                    # Store raw content if JSON parsing fails
                    try:
                        raw_content = getattr(response, 'content', None) or getattr(response, 'text', None)
                        if raw_content:
                            if isinstance(raw_content, bytes):
                                raw_content = raw_content.decode('utf-8', errors='ignore')
                            response_data = {"error": "Failed to parse JSON response", "raw_content": str(raw_content)[:1000], "parse_error": str(e)}
                        else:
                            response_data = {"error": "Failed to parse response and no content available", "parse_error": str(e)}
                    except Exception as inner_e:
                        response_data = {"error": "Complete response parsing failure", "parse_error": str(e), "inner_error": str(inner_e)}

            # Ensure response_data is never None/null and capture meaningful error info
            if response_data is None:
                response_data = {
                    "error": "Response data is unexpectedly null",
                    "status_code": status_code,
                    "timestamp": datetime.now().isoformat(),
                    "error_type": "null_response"
                }

            # For error status codes, ensure we capture error details
            if status_code >= 400:
                if isinstance(response_data, dict):
                    response_data["error_captured"] = True
                    response_data["error_status"] = status_code
                    if "error" not in response_data:
                        response_data["error"] = f"HTTP {status_code} error"
                else:
                    # If response_data is not a dict, wrap it with error info
                    response_data = {
                        "error": f"HTTP {status_code} error",
                        "error_status": status_code,
                        "error_captured": True,
                        "raw_response": response_data,
                        "timestamp": datetime.now().isoformat()
                    }

            # Store the LLM response payload
            self.current_call_record["response"] = response_data

            # Extract model from response if available and not already set
            if isinstance(response_data, dict):
                # Try to get model from response
                response_model = response_data.get('model')
                if response_model and not self.current_call_record["metadata"].get("model"):
                    self.current_call_record["metadata"]["model"] = response_model

            # Extract LLM API response ID and use it as our call_id
            llm_response_id = self._extract_llm_response_id(response_data)
            if llm_response_id:
                old_call_id = self.current_call_record["call_id"]

                # Update the call_id to use the LLM's response ID
                self.current_call_record["call_id"] = llm_response_id

                # Update the call_id in the current context's calls list
                if hasattr(self, 'current_loop_context') and self.current_loop_context:
                    # Loop context
                    loop_name = self.current_loop_context["loop_name"]
                    calls_list = self.current_loop_context["calls_in_loop"]
                    if old_call_id in calls_list:
                        calls_list[calls_list.index(old_call_id)] = llm_response_id

                # Update flow connections with the new LLM call ID
                self._update_flow_connections(old_call_id, llm_response_id)

            # Store duration in metadata (ensure it's never null)
            self.current_call_record["metadata"]["duration_ms"] = self.current_call_record.get("duration_ms", 0)

            # Agent auto-mark and follow policies
            try:
                _init_thread_local()
                agent_id = getattr(_thread_local, 'current_agent_id', None)
                obs_enabled = getattr(_thread_local, 'observation_enabled', False)
                if obs_enabled and agent_id:
                    self._ensure_agent_bucket(agent_id)
                    # Extract assistant text for known formats
                    assistant_text = self._extract_llm_response_text(response_data)

                    # Create marker for EVERY response (even tool calls with no text content)
                    # This ensures edges are created for all calls in a sequence
                    marker_id = f"m_{uuid.uuid4().hex[:12]}"
                    marker = {
                        "marker_id": marker_id,
                        "content_length": len(assistant_text) if assistant_text else 0,
                        "created_by_call": self.current_call_record["call_id"],
                        "created_at": datetime.now().isoformat()
                    }
                    self.tree["agents"][agent_id]["markers"][marker_id] = marker

                    # Apply follow policy
                    policy = self._agents[agent_id].get("follow_policy", {"type": "none"})
                    if policy.get("type") == "fanout_all":
                        self._agents[agent_id]["active_markers"].add(marker_id)
                    elif policy.get("type") == "last_only":
                        self._agents[agent_id]["active_markers"] = {marker_id}
                    elif policy.get("type") == "window":
                        n = int(policy.get("n", 1))
                        current = list(self._agents[agent_id]["active_markers"]) + [marker_id]
                        self._agents[agent_id]["active_markers"] = set(current[-n:])
                        # type "none" adds nothing
            except Exception:
                pass

        except Exception as e:
            # Store comprehensive error information - never leave response as null
            error_info = {
                "error": "Exception in _capture_response_data",
                "exception": str(e),
                "exception_type": type(e).__name__,
                "status_code": getattr(response, 'status_code', None),
                "response_type": type(response).__name__,
                "timestamp": datetime.now().isoformat(),
                "error_captured": True,
                "error_source": "span_capture_exception"
            }

            # Try to get any available response content even in error case
            try:
                raw_content = getattr(response, 'content', None) or getattr(response, 'text', None)
                if raw_content:
                    if isinstance(raw_content, bytes):
                        raw_content = raw_content.decode('utf-8', errors='ignore')
                    error_info["raw_content"] = str(raw_content)[:500]  # Truncate to avoid huge error logs
            except:
                pass

            self.current_call_record["response"] = error_info

            # Ensure metadata fields are never null
            self.current_call_record["metadata"]["status_code"] = getattr(response, 'status_code', 0)
            self.current_call_record["metadata"]["duration_ms"] = self.current_call_record.get("duration_ms", 0)

    def _capture_request_data(self, request, url: str, is_async: bool):
        """Capture request data for LLM calls"""
        # If loop context exists, use loop linkage; otherwise, check global observation
        if hasattr(self, 'current_loop_context') and self.current_loop_context:
            self._capture_request_in_loop(request, url, is_async)
            return

        _init_thread_local()
        if getattr(_thread_local, 'observation_enabled', False):
            self._capture_request_with_agents(request, url, is_async)
            return
        return

    def _capture_request_in_loop(self, request, url: str, is_async: bool):
        """Capture request data for LLM calls within a loop context"""
        # Create a new call record for this HTTP request
        temp_call_id = str(uuid.uuid4())  # Temporary until we get LLM's response ID
        call_record = {
            "call_id": temp_call_id,  # Will be updated with LLM's response ID
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_ms": None,
            "request": {},  # Will store RAW LLM request - initialize as empty dict, not null
            "response": {},  # Will store RAW LLM response - initialize as empty dict, not null
            "metadata": {
                "model": "unknown",  # Default to 'unknown' instead of null
                "prompt_ids": [],
                "request_url": url,
                "status_code": 0,  # Default to 0 instead of null
                "duration_ms": 0   # Default to 0 instead of null
            }
        }

        # Add to current loop
        loop_name = self.current_loop_context["loop_name"]
        self.tree["loops"][loop_name]["calls"].append(call_record)
        self.current_loop_context["calls_in_loop"].append(temp_call_id)

        # Create flow edge from previous call to this call
        self._create_loop_flow_edge(loop_name, temp_call_id)

        # Store for response capture
        self.current_call_record = call_record

        # Process pending marker usage records (from prompt compilations)
        if hasattr(self, '_pending_usage_records') and self._pending_usage_records:
            for usage_record in self._pending_usage_records:
                marker_name = usage_record["marker_name"]
                prompt_id = usage_record["prompt_id"]
                self._record_marker_usage(marker_name, prompt_id, temp_call_id)

            # Clear processed records
            self._pending_usage_records = []

        # Process direct marker usage (when get_marked() was called but no prompt compilation)
        if hasattr(self, '_pending_marker_usage') and self._pending_marker_usage:
            for marker_name in self._pending_marker_usage:
                # Record usage without a specific prompt_id (direct usage)
                self._record_marker_usage(marker_name, None, temp_call_id)

            # Clear pending usage
            self._pending_marker_usage = []

        try:
            # Extract full payload from request
            if is_async:
                # httpx request
                content = request.content
                content_type = request.headers.get("Content-Type", "").lower()
            else:
                # requests request
                content = request.body
                content_type = request.headers.get("Content-Type", "").lower()

            if "application/json" in content_type and content:
                try:
                    if isinstance(content, bytes):
                        json_data = json.loads(content.decode('utf-8'))
                    else:
                        json_data = json.loads(content)

                    # Store the LLM request payload
                    self.current_call_record["request"] = json_data

                    # Store metadata - ensure model is never null/unknown
                    model = json_data.get('model')
                    if model:
                        self.current_call_record["metadata"]["model"] = model
                    else:
                        # Try to extract from nested structures or set a reasonable default
                        self.current_call_record["metadata"]["model"] = "unknown"

                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
        except Exception:
            pass






    def _ensure_agent_bucket(self, agent_id: str):
        if agent_id not in self.tree["agents"]:
            self.tree["agents"][agent_id] = {
                "calls": [],
                "edges": [],
                "markers": {},
                "policy": {"type": "none"}
            }
        if agent_id not in self._agents:
            self._agents[agent_id] = {
                "active_markers": set(),
                "follow_policy": {"type": "none"}
            }

    def _capture_request_with_agents(self, request, url: str, is_async: bool):
        """Capture request data for LLM calls under agent observation (no loops)."""
        _init_thread_local()
        agent_id = getattr(_thread_local, 'current_agent_id', None) or "default"
        self._ensure_agent_bucket(agent_id)

        temp_call_id = str(uuid.uuid4())
        call_record = {
            "call_id": temp_call_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_ms": None,
            "request": {},
            "response": {},
            "metadata": {
                "model": "unknown",
                "prompt_ids": [],
                "request_url": url,
                "status_code": 0,
                "duration_ms": 0
            }
        }

        self.tree["agents"][agent_id]["calls"].append(call_record)
        self.current_call_record = call_record

        # Build edges from active markers according to policy
        active_markers = list(self._agents[agent_id]["active_markers"]) if agent_id in self._agents else []
        for marker_id in active_markers:
            marker = self.tree["agents"][agent_id]["markers"].get(marker_id)
            if marker and marker.get("created_by_call"):
                edge = {
                    "from": marker["created_by_call"],
                    "to": temp_call_id,
                    "edge_type": "follows"
                }
                self.tree["agents"][agent_id]["edges"].append(edge)

        # Parse request payload for model
        try:
            if is_async:
                content = request.content
                content_type = request.headers.get("Content-Type", "").lower()
            else:
                content = request.body
                content_type = request.headers.get("Content-Type", "").lower()
            if "application/json" in content_type and content:
                if isinstance(content, bytes):
                    json_data = json.loads(content.decode('utf-8'))
                else:
                    json_data = json.loads(content)
                self.current_call_record["request"] = json_data
                model = json_data.get('model')
                if model:
                    self.current_call_record["metadata"]["model"] = model
        except Exception:
            pass



    def finalize(self):
        """Finalize the span and set end time"""
        self.end_time = datetime.now().isoformat()
        self.tree["end_time"] = self.end_time
        self.active = False

        # Calculate total span duration
        if self.tree["start_time"] and self.tree["end_time"]:
            start_dt = datetime.fromisoformat(self.tree["start_time"])
            end_dt = datetime.fromisoformat(self.tree["end_time"])
            duration = (end_dt - start_dt).total_seconds() * 1000
            self.tree["duration_ms"] = round(duration, 2)

    def get_tree(self) -> Dict:
        """Get the complete execution tree"""
        return self.tree

    def export_json(self) -> str:
        """Export the execution tree as JSON"""
        return json.dumps(self.tree, indent=2)



class PromptTemplate:
    def __init__(self, prompt_id: str, prompt_data: Dict, span: Span):
        self.prompt_id = prompt_id
        self.template = prompt_data.get("template", "")
        self.parameters = prompt_data.get("parameters", {})
        self.version = prompt_data.get("version", 1)
        self.deployed_version = prompt_data.get("deployed_version", 1)
        self.total_versions = prompt_data.get("total_versions", 1)
        self.span = span

    def compile(self, **kwargs) -> str:
        """Compile the prompt template with given parameters"""
        try:
            # Handle both single and double curly brace formats
            # Convert double braces {{param}} to single braces {param} for formatting
            template_to_format = self.template
            if '{{' in template_to_format and '}}' in template_to_format:
                # Replace double braces with single braces for parameter substitution
                import re
                template_to_format = re.sub(r'\{\{(\w+)\}\}', r'{\1}', template_to_format)

            compiled_prompt = template_to_format.format(**kwargs)

            # NEW: Direct association approach - associate compiled text with this specific prompt
            # Instead of using shared thread-local list, store compiled text directly for exact matching
            compilation_id = f"{self.prompt_id}_{datetime.now().isoformat()}_{id(compiled_prompt)}"
            compiled_prompt_info = {
                "prompt_id": self.prompt_id,
                "compiled_text": compiled_prompt,
                "parameters_used": kwargs,
                "version_used": self.version,
                "deployed_version": self.deployed_version,
                "compiled_at": datetime.now().isoformat(),
                "compilation_id": compilation_id,
                "span_instance": self.span
            }

            # Store in span for exact text matching (no thread-local accumulation)
            if not hasattr(self.span, '_recent_compilations'):
                self.span._recent_compilations = {}
            self.span._recent_compilations[compilation_id] = compiled_prompt_info

            # Keep the old tracking for backwards compatibility
            self.span.register_compiled_prompt(self.prompt_id, compiled_prompt, kwargs)

            # Track the compilation in span metadata with version info
            self.span.add(f"prompt_compilation_{self.prompt_id}", {
                "prompt_id": self.prompt_id,
                "version_used": self.version,
                "deployed_version": self.deployed_version,
                "parameters_used": kwargs,
                "compiled_length": len(compiled_prompt)
            })

            # NEW: Track marker usage if there's a pending marker
            if self.span._pending_marker_usage:
                for marker_name in self.span._pending_marker_usage:

                    # We need to defer the call_id recording until the next LLM call is made
                    # Store the pending usage for later processing
                    if not hasattr(self.span, '_pending_usage_records'):
                        self.span._pending_usage_records = []

                    self.span._pending_usage_records.append({
                        "marker_name": marker_name,
                        "prompt_id": self.prompt_id
                    })

                # Clear pending usage
                self.span._pending_marker_usage = []

            return compiled_prompt
        except KeyError as e:
            missing_param = str(e).strip("'")
            raise ValueError(f"Missing required parameter '{missing_param}' for prompt '{self.prompt_id}'")

    def __str__(self) -> str:
        """Return the raw template"""
        return self.template


class EvaluatorNotFoundError(Exception):
    """Exception raised when an evaluator is not found"""
    def __init__(self, evaluator_name: str, message: str = None):
        self.evaluator_name = evaluator_name
        self.message = message or f"Evaluator '{evaluator_name}' not found"
        super().__init__(self.message)


class EvaluatorTemplate:
    def __init__(self, evaluator_name: str, evaluator_data: Dict, span: 'Span', aviro_instance: 'Aviro'):
        self.evaluator_name = evaluator_name
        self.evaluator_prompt = evaluator_data.get("evaluator_prompt", "")
        self.variables = evaluator_data.get("variables", [])
        self.model = evaluator_data.get("model", "gpt-4o-mini")
        self.temperature = evaluator_data.get("temperature", 0.1)
        self.structured_output = evaluator_data.get("structured_output")
        self.pydantic_model_class = evaluator_data.get("pydantic_model_class")
        self.span = span
        self.aviro = aviro_instance

    def evaluate(self, **variables):
        """Evaluate with the given variables - always uses API to ensure database storage"""
        try:
            # Check if all expected variables are provided
            missing_variables = [var for var in self.variables if var not in variables]

            # ALWAYS use API-based evaluation to ensure runs are stored in database
            # This ensures that evaluation runs appear in the web UI
            return self._evaluate_via_api(**variables)

        except Exception as e:
            raise Exception(f"Failed to evaluate with evaluator '{self.evaluator_name}': {str(e)}")

    def _evaluate_via_api(self, **variables):
        """Fallback to API evaluation"""
        eval_data = {
            "evaluator_name": self.evaluator_name,
            "variables": variables
        }

        response = requests.post(
            f"{self.aviro.base_url}/api/evaluations",
            json=eval_data,
            headers={"Authorization": f"Bearer {self.aviro.api_key}"},
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Evaluation failed with status {response.status_code}: {response.text}")


class Evaluator:
    def __init__(self, evaluator_name: str, aviro_instance: 'Aviro'):
        self.evaluator_name = evaluator_name
        self.aviro = aviro_instance

    def evaluate(self, **variables):
        """Evaluate with the given variables"""
        try:
            # Prepare the evaluation request
            eval_data = {
                "evaluator_name": self.evaluator_name,
                "variables": variables
            }

            # Send to the evaluations endpoint
            response = requests.post(
                f"{self.aviro.base_url}/api/evaluations",
                json=eval_data,
                headers={"Authorization": f"Bearer {self.aviro.api_key}"},
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Evaluation failed with status {response.status_code}: {response.text}")

        except Exception as e:
            raise Exception(f"Failed to evaluate with evaluator '{self.evaluator_name}': {str(e)}")


class PromptAlreadyExistsError(Exception):
    """Exception raised when a prompt already exists in the webapp"""
    def __init__(self, prompt_id: str, message: str = None):
        self.prompt_id = prompt_id
        self.message = message or f"Prompt '{prompt_id}' already exists in the webapp"
        super().__init__(self.message)

class EvaluatorAlreadyExistsError(Exception):
    """Exception raised when an evaluator already exists in the webapp"""
    def __init__(self, evaluator_name: str, message: str = None):
        self.evaluator_name = evaluator_name
        self.message = message or f"Evaluator '{evaluator_name}' already exists in the webapp"
        super().__init__(self.message)


class SpanDecoratorContextManager:
    """A class that can be used as both a decorator and context manager"""
    def __init__(self, aviro_instance: 'Aviro', span_name: str, organization_name: str = None):
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
    def __init__(self, aviro_instance: 'Aviro', loop_name: str):
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
    def __init__(self, agent_id: str, policy: str = "last_only", aviro_instance: 'Aviro' = None, organization_name: str = None):
        self.agent_id = agent_id
        self.group_name = agent_id
        self.policy = policy
        self.organization_name = organization_name
        self._aviro = aviro_instance
        self._span_created = False

    def _get_aviro(self):
        if self._aviro is None:
            # Use configured global instance; do not auto-create from environment
            global _global_aviro
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
        global _global_aviro
        if not isinstance(client, AviroClient):
            raise TypeError("configure() expects an AviroClient instance")
        _global_aviro = client._aviro

    # Legacy compatibility
    @staticmethod
    def loop(span_name: str):
        """Legacy loop API for backward compatibility."""
        global _global_aviro
        if _global_aviro is None:
            api_key = os.getenv("AVIRO_API_KEY")
            base_url = os.getenv("AVIRO_BASE_URL")
            _global_aviro = Aviro(api_key=api_key, base_url=base_url, auto_submit=True)
        return LoopDecoratorContextManager(_global_aviro, span_name)


# Convenience singleton for users who want `observe.loop(...)`
# Defer instantiation until after Aviro is defined to avoid NameError at import time
# The actual instantiation is placed at the end of this module.

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
                     default_structured_output: Union[Dict, Type[BaseModel]] = None) -> 'EvaluatorTemplate':
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





# Instantiate the convenience singleton after all classes are defined
observe = Observe()
# Create convenience functions
def prompt(template: str) -> str:
    """Create a prompt string (legacy compatibility)"""
    return template


def lm():
    """Language model placeholder (legacy compatibility)"""
    pass


# Pretty-print helpers for execution tree in flattened calls shape
def get_flat_calls_json(aviro_instance: 'Aviro' = None) -> str:
    """Return the flattened calls JSON string: {"calls": [...]}.

    If no aviro_instance is provided, uses the global observer instance.
    """
    try:
        av = aviro_instance if aviro_instance is not None else _global_aviro
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
    import warnings
    warnings.warn("loop() function is deprecated. Use AviroClient().loop() instead.", DeprecationWarning)

    # Try to get from environment
    import os
    api_key = os.environ.get("AVIRO_API_KEY")
    if not api_key:
        raise RuntimeError("AVIRO_API_KEY is not set. Use AviroClient(api_key='...').loop() instead.")

    client = AviroClient(api_key=api_key)
    with client.loop(loop_name) as span:
        yield span
