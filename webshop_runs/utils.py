import os
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)

from typing import Optional, List, Union, Callable

# Optional: if OPENAI_API_KEY is set and openai package is available, the code will
# continue to use OpenAI. If you don't have OpenAI access, you can register a local
# completion function via `set_local_completion_fn(fn)` that accepts the same
# signature as get_completion and returns a string (or list of strings if batched).

LOCAL_COMPLETION_FN: Optional[Callable[..., Union[str, List[str]]]] = None

def set_local_completion_fn(fn: Callable[..., Union[str, List[str]]]) -> None:
    """Register a local completion function to be used when OpenAI is not available.

    The function should have signature:
        fn(prompt: Union[str, List[str]], max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> Union[str, List[str]]
    """
    global LOCAL_COMPLETION_FN
    LOCAL_COMPLETION_FN = fn


try:
    import openai
    openai.api_key = os.getenv('OPENAI_API_KEY')
    _HAS_OPENAI = bool(openai.api_key)
except Exception:
    openai = None  # type: ignore
    _HAS_OPENAI = False


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: Union[str, List[str]], max_tokens: int = 256, stop_strs: Optional[List[str]] = None, is_batched: bool = False) -> Union[str, List[str]]:
    """Get completion text either from OpenAI (if configured) or from a registered local function.

    If a local function is registered via `set_local_completion_fn`, it will be used in
    preference to OpenAI. This lets you plug-in an UnsLoth/HF model adapter without
    changing other code.
    """
    assert (not is_batched and isinstance(prompt, str)) or (is_batched and isinstance(prompt, list))

    # If user has registered a local completion function, use it.
    if LOCAL_COMPLETION_FN is not None:
        return LOCAL_COMPLETION_FN(prompt, max_tokens=max_tokens, stop_strs=stop_strs, is_batched=is_batched)

    # Otherwise, fall back to OpenAI if configured
    if _HAS_OPENAI and openai is not None:
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=stop_strs,
        )
        if is_batched:
            res: List[str] = [""] * len(prompt)
            for choice in response.choices:
                res[choice.index] = choice.text
            return res
        return response.choices[0].text

    raise RuntimeError("No completion backend available: set OPENAI_API_KEY or register a local completion function using set_local_completion_fn(fn)")
