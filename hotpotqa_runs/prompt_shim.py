from typing import List

class PromptTemplate:
    """Minimal shim of LangChain's PromptTemplate used by this repo.

    Provides `input_variables` and a `format(**kwargs)` method that
    performs Python str.format on the stored template.
    """
    def __init__(self, input_variables: List[str], template: str):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs) -> str:
        # Simple wrapper around Python formatting. Caller is responsible
        # for providing required keys.
        return self.template.format(**kwargs)
