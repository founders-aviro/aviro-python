"""Core Span class for tracking LLM execution"""

import json
import uuid
import requests
import httpx
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import contextmanager

from .utils import (
    _init_thread_local,
    _thread_local,
    _original_requests_session_send,
    _original_httpx_async_client_send,
    _original_httpx_client_send,
)


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


    def get_prompt(self, prompt_id: str, default_prompt: str = None):
        """Get or create a prompt template - completely local, no API calls"""
        from .templates import PromptTemplate
        from .exceptions import PromptNotFoundError
        
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
        from .exceptions import PromptAlreadyExistsError
        
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

    def set_evaluator(self, evaluator_name: str, evaluator_prompt: str, variables: list = None,
                     model: str = "gpt-4o-mini", temperature: float = 0.1,
                     structured_output = None):
        """Set an evaluator manually - creates in webapp database if API key available"""
        from pydantic import BaseModel
        from .exceptions import EvaluatorAlreadyExistsError
        
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

    def _set_evaluator_local(self, evaluator_name: str, evaluator_prompt: str, variables: list = None,
                            model: str = "gpt-4o-mini", temperature: float = 0.1,
                            processed_structured_output = None,
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
                     default_variables: list = None, default_structured_output = None,
                     aviro_instance = None):
        """Get or create an evaluator template - completely local, no API calls"""
        from pydantic import BaseModel
        from .templates import EvaluatorTemplate
        from .exceptions import EvaluatorNotFoundError
        
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

