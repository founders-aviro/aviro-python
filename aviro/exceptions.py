"""Exception classes for Aviro SDK"""


class PromptNotFoundError(Exception):
    """Exception raised when a prompt is not found"""
    def __init__(self, prompt_id: str, message: str = None):
        self.prompt_id = prompt_id
        self.message = message or f"Prompt '{prompt_id}' not found"
        super().__init__(self.message)


class PromptAlreadyExistsError(Exception):
    """Exception raised when a prompt already exists in the webapp"""
    def __init__(self, prompt_id: str, message: str = None):
        self.prompt_id = prompt_id
        self.message = message or f"Prompt '{prompt_id}' already exists in the webapp"
        super().__init__(self.message)


class EvaluatorNotFoundError(Exception):
    """Exception raised when an evaluator is not found"""
    def __init__(self, evaluator_name: str, message: str = None):
        self.evaluator_name = evaluator_name
        self.message = message or f"Evaluator '{evaluator_name}' not found"
        super().__init__(self.message)


class EvaluatorAlreadyExistsError(Exception):
    """Exception raised when an evaluator already exists in the webapp"""
    def __init__(self, evaluator_name: str, message: str = None):
        self.evaluator_name = evaluator_name
        self.message = message or f"Evaluator '{evaluator_name}' already exists in the webapp"
        super().__init__(self.message)

