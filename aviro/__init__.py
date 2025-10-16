"""Aviro - A Python SDK for LLM observability and prompt management."""

__version__ = "0.1.5"
__author__ = "Aviro Team"
__email__ = "support@aviro.ai"

# Import main public API classes and functions
from .client import (
    AviroClient,
    Aviro,
    observe,
    loop,
    prompt,
    lm,
    get_flat_calls_json,
    print_flat_calls,
)
from .core_span import Span
from .templates import PromptTemplate, EvaluatorTemplate, Evaluator
from .exceptions import (
    PromptNotFoundError,
    PromptAlreadyExistsError,
    EvaluatorNotFoundError,
    EvaluatorAlreadyExistsError,
)
from .utils import MarkedResponse, get_current_span

__all__ = [
    # Main client classes
    "AviroClient",
    "Aviro",
    "observe",
    # Core classes
    "Span",
    "PromptTemplate",
    "EvaluatorTemplate",
    "Evaluator",
    # Exceptions
    "PromptNotFoundError",
    "PromptAlreadyExistsError",
    "EvaluatorNotFoundError",
    "EvaluatorAlreadyExistsError",
    # Utilities
    "MarkedResponse",
    "get_current_span",
    # Legacy functions
    "loop",
    "prompt",
    "lm",
    "get_flat_calls_json",
    "print_flat_calls",
]
