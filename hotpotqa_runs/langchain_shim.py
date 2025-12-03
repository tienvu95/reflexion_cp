"""Minimal shim for a few langchain types used in the repo's agents.

This provides lightweight stand-ins so the repo can run without installing
the full `langchain` package in minimal environments (e.g., Colab).

Only the small set of symbols used by `agents.py` are defined here.
"""
from typing import Any


class BaseLLM:
    """Minimal stand-in for langchain.llms.base.BaseLLM"""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError("BaseLLM shim does not implement generation. Provide a real LLM callable.")


class BaseChatModel(BaseLLM):
    """Minimal stand-in for langchain.chat_models.base.BaseChatModel"""
    pass


class SystemMessage:
    def __init__(self, content: str = ""):
        self.content = content


class HumanMessage(SystemMessage):
    pass


class AIMessage(SystemMessage):
    pass


# Export a minimal __all__ so `from langchain_shim import ...` works nicely
__all__ = ["BaseLLM", "BaseChatModel", "SystemMessage", "HumanMessage", "AIMessage"]
