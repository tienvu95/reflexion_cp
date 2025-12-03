"""Small demo: load an HF transformers model and call run_pubmedqa.run(..., external_llm=llm)

This file is safe to import (it won't load the model on import). Run it as a script
to load the model and execute the runner.

Usage (bash/zsh):
  # export HF token if the model is gated
  export HUGGINGFACE_HUB_TOKEN=your_token_here
  python hotpotqa_runs/run_local_llm_demo.py

Notes:
 - Start with `load_in_4bit=False` for debugging; enable 4-bit only after confirming
   the model loads without errors and `bitsandbytes` is installed.
 - If you hit model-specific attribute errors (e.g., LlamaAttention.apply_qkv), try
   upgrading `transformers`, `accelerate`, `safetensors`, and `bitsandbytes`.
"""
import os
from types import SimpleNamespace

def make_args(model_id=None, load_in_4bit=False, device=None):
    # Prefer explicit device if provided; otherwise, try torch if available.
    dev = device
    if dev is None:
        try:
            import torch  # optional
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            dev = None

    return SimpleNamespace(
        # Dataset config
        dataset='qiaojin/PubMedQA',
        dataset_config='pqa_labeled',
        split='train',
        limit=5,  # small test

        # Agent + reflexion
        agent='cot',
        reflexion_strategy='reflexion',

        # Model + loading preferences (not used when passing external_llm, but kept for completeness)
        model=model_id,
        use_transformers=True,
        use_unsloth=False,
        hf_token=os.environ.get('HF_API_TOKEN'),
        out=None,

        # Field mapping
        question_field='question',
        context_field='context',
        answer_field='final_decision',
        long_answer_field='long_answer',

        # Agent controls
        max_steps=6,
        print_debug=True,
        print_logit_debug=True,  # enable to see yes/no/maybe probs for Brier
        keep_fewshot_examples=True,

        # Local transformers options
        device=dev,
        load_in_4bit=bool(load_in_4bit),
        max_seq_length=8192,

        # Confidence/reflexion loop
        confidence_threshold=0.6,
        max_reflect_attempts=3,
        force_finish_format=False,
        force_argmax_final=False,

        # Readability / rewrite / enforcement controls
        readability_min=6.0,           # Flesch-Kincaid lower bound (grade)
        readability_max=10.0,          # Flesch-Kincaid upper bound (grade)
        rewrite_on_readability=True,   # fallback rewrite prompt if reflexion fails
        enforce_readability_reflexion=True,  # trigger agent.reflect + rerun to enforce FK
        enforce_length=True,                 # try to match rationale length to long_answer
        length_tolerance=0.20,               # +/-20% around gold long_answer word count
        max_readability_rewrites=1,          # Option A acceptance loop attempts
        rouge_drop_threshold=0.05,           # Accept rewrite only if ROUGE-1 drop <= this
    )


def demo_run(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", load_in_4bit=False, device=None):
    """Load HF model and call run_pubmedqa.run with external_llm.

    This intentionally keeps `limit` small for a quick smoke test.
    """
    # Import locally to avoid heavy deps on module import
    from hotpotqa_runs.hf_transformers_llm import HFTransformersLLM
    from hotpotqa_runs import run_pubmedqa

    args = make_args(model_id=model_id, load_in_4bit=load_in_4bit, device=device)

    print("Creating HFTransformersLLM (this will download/load weights)...")
    llm = HFTransformersLLM(model_id=model_id, load_in_4bit=load_in_4bit, device=device)

    print("Calling run_pubmedqa.run(...) with external_llm (this may take time)...")
    results = run_pubmedqa.run(args, external_llm=llm)
    print("Run finished. Results summary (first rows):")
    try:
        # try printing a lightweight preview
        for r in (results[:5] if hasattr(results, '__len__') else [results]):
            print(r)
    except Exception:
        print(results)
    return results


if __name__ == '__main__':
    # Simple CLI flags via env vars for quick testing
    model_id = os.environ.get('HF_MODEL_ID', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
    use_4bit = os.environ.get('HF_LOAD_4BIT', '0') in ('1', 'true', 'True')
    device = os.environ.get('HF_DEVICE', None)

    print(f"Demo config: model_id={model_id}, load_in_4bit={use_4bit}, device={device}")
    try:
        demo_run(model_id=model_id, load_in_4bit=use_4bit, device=device)
    except Exception as e:
        print("Error during demo run:", type(e).__name__, e)
        print("If this is an attribute error from the model implementation, try upgrading transformers/accelerate/bitsandbytes or use load_in_4bit=False.")
import os
"""Demo: run CoTAgent using the Hugging Face Inference API.

Set environment variables:
  - HF_API_TOKEN: your Hugging Face inference API token
  - HF_MODEL_ID: e.g. meta-llama/Llama-3.1-8b-instruct

Then run from the repository root or from `hotpotqa_runs/`:
  python run_local_llm_demo.py
"""

try:
    # prefer package import when run from repo root
    from hotpotqa_runs.hf_inference_llm import HFInferenceLLM
except Exception:
    from hf_inference_llm import HFInferenceLLM

# Local transformers adapter is optional and loaded only when requested.
USE_TRANSFORMERS = os.environ.get('USE_TRANSFORMERS', '') in ['1', 'true', 'True']
if USE_TRANSFORMERS:
    try:
        from hotpotqa_runs.hf_transformers_llm import HFTransformersLLM
    except Exception:
        try:
            from hf_transformers_llm import HFTransformersLLM
        except Exception:
            HFTransformersLLM = None

try:
    from hotpotqa_runs.agents import CoTAgent, ReflexionStrategy
except Exception:
    from agents import CoTAgent, ReflexionStrategy


def main():
    # If user wants to use a local transformers model, set USE_TRANSFORMERS=1 and HF_LOCAL_MODEL
    if USE_TRANSFORMERS:
        if HFTransformersLLM is None:
            print('HFTransformersLLM adapter not available. Ensure `hotpotqa_runs/hf_transformers_llm.py` exists and dependencies are installed.')
            return
        local_model_id = os.environ.get('HF_LOCAL_MODEL', os.environ.get('HF_MODEL_ID'))
        if not local_model_id:
            print('Set HF_LOCAL_MODEL or HF_MODEL_ID to the local model id you want to load.')
            return

        try:
            # Pass `device=None` when HF_DEVICE is not set so adapter auto-detects
            device_env = os.environ.get('HF_DEVICE')
            llm = HFTransformersLLM(model_id=local_model_id, device=device_env, load_in_4bit=os.environ.get('HF_LOAD_4BIT','1') in ['1','true','True'])
        except Exception as e:
            print('Failed to construct HFTransformersLLM:', e)
            return

    else:
        hf_token = os.environ.get('HF_API_TOKEN')
        model_id = os.environ.get('HF_MODEL_ID', 'meta-llama/Llama-3.1-8b-instruct')
        if not hf_token:
            print('HF_API_TOKEN not set. Please export HF_API_TOKEN and HF_MODEL_ID to use the Hugging Face Inference API or set USE_TRANSFORMERS=1 to load models locally.')
            return

        llm = HFInferenceLLM(model_id=model_id, api_token=hf_token, temperature=0.0, max_new_tokens=256)

    agent = CoTAgent(
        question="Is aspirin recommended for reducing fever?",
        context="Abstract: ...",
        key="yes",
        self_reflect_llm=llm,
        action_llm=llm,
    )

    # Run one trial (run will call LLM when needed). This demo does not provide real context.
    try:
        agent.run(reflexion_strategy=ReflexionStrategy.REFLEXION)
    except Exception as e:
        print('Agent run failed:', e)
    print('Pred:', agent.answer)


if __name__ == '__main__':
    main()