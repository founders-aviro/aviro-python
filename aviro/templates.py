"""Template classes for prompts and evaluators"""

import re
import requests
from datetime import datetime
from typing import Dict
from pydantic import BaseModel


class PromptTemplate:
    def __init__(self, prompt_id: str, prompt_data: Dict, span):
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


class EvaluatorTemplate:
    def __init__(self, evaluator_name: str, evaluator_data: Dict, span, aviro_instance):
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
    def __init__(self, evaluator_name: str, aviro_instance):
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

