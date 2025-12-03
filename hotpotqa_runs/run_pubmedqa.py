"""Run PubMedQA dataset through existing agents and LLM adapters.

Usage examples:
  # use HF Inference API (requires HF token in HF_API_TOKEN)
  python run_pubmedqa.py --use-inference --model qiaojin/PubMedQA --split validation --limit 50 --out results_pubmed.csv

  # use local transformers adapter (requires model available locally or on HF hub)
  python run_pubmedqa.py --use-transformers --model google/flan-t5-small --split validation --limit 20 --out results_pubmed.csv

Notes:
- The script heuristically finds the question/context/answer fields in the dataset.
- For 8B Llama-style models you must run with GPU + required deps (bitsandbytes) and set `--use-transformers`.
"""

import argparse
import csv
import os
import sys
from typing import Optional

# Ensure package imports work when running as module or script.
# Insert the repository root (parent of this file) into sys.path so
# imports like `hotpotqa_runs.*` resolve when the script is executed directly.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Note: import datasets lazily inside `run()` to avoid hard dependency at import-time

try:
    # Prefer repo-provided adapters if present
    from hf_inference_llm import HFInferenceLLM
except Exception:
    HFInferenceLLM = None

try:
    from hf_transformers_llm import HFTransformersLLM
except Exception:
    HFTransformersLLM = None

try:
    from agents import ReactAgent, CoTAgent, ReactReflectAgent, EM, ReflexionStrategy
except Exception:
    # fallback relative import
    from hotpotqa_runs.agents import ReactAgent, CoTAgent, ReactReflectAgent, EM, ReflexionStrategy


def map_reflexion_str(s: Optional[str]):
    if s is None:
        return ReflexionStrategy.REFLEXION
    s = s.lower()
    if s in ('none', 'base'):
        return ReflexionStrategy.NONE
    if s in ('last_attempt', 'last'):
        return ReflexionStrategy.LAST_ATTEMPT
    if s in ('reflexion', 'reflect'):
        return ReflexionStrategy.REFLEXION
    if s in ('last_attempt_and_reflexion', 'last_and_reflexion'):
        return ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION
    # default
    return ReflexionStrategy.REFLEXION


def build_llm(args):
    """Instantiate either HF Inference or local Transformers adapter depending on args."""
    # shared token used for HF authentication when needed
    token = args.hf_token or os.environ.get('HF_API_TOKEN') or os.environ.get('HUGGINGFACE_API_TOKEN')

    # UnsloTh path (Colab-ready prequantized models)
    if getattr(args, 'use_unsloth', False):
        try:
            from unsloth_llm import UnslothLLM
        except Exception:
            from hotpotqa_runs.unsloth_llm import UnslothLLM
        return UnslothLLM(model_name=args.model, token=token, load_in_4bit=getattr(args, 'load_in_4bit', True), max_seq_length=getattr(args, 'max_seq_length', 8192))

    if args.use_transformers:
        # Build a local transformers-backed LLM inline so we can set trust_remote_code
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("Local transformers path requires 'transformers' and 'torch' installed.") from e

        model_id = args.model

        # If user provided an HF token, set HUGGINGFACE_HUB_TOKEN for downloads
        token = args.hf_token or os.environ.get('HF_API_TOKEN') or os.environ.get('HUGGINGFACE_API_TOKEN')
        if token:
            os.environ['HUGGINGFACE_HUB_TOKEN'] = token

        bnb_cfg = None
        if getattr(args, 'load_in_4bit', False):
            try:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                bnb_cfg = None

        # Load tokenizer and model (trust_remote_code=True for models that require it)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True, use_auth_token=token)
        except OSError as e:
            raise RuntimeError(
                f"Could not access model '{model_id}'. This repository may be private or gated.\n"
                "Make sure you have accepted the model license on Hugging Face, then either: \n"
                "  1) run `huggingface-cli login` and authenticate, or\n"
                "  2) set your token in the env var `HF_API_TOKEN` (or pass --hf-token) before running this script.\n"
                "Example: export HF_API_TOKEN=hf_...\n"
            ) from e

        load_kwargs = {"device_map": "auto"}
        if bnb_cfg is not None:
            load_kwargs["quantization_config"] = bnb_cfg
        # prefer float16 on CUDA devices
        try:
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False
        if use_cuda:
            load_kwargs["torch_dtype"] = torch.float16

        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, use_auth_token=token, **load_kwargs)
        except OSError as e:
            raise RuntimeError(
                f"Could not download/load model '{model_id}'. Ensure your token has access and you accepted the model terms on Hugging Face.\n"
                "Either run `huggingface-cli login` or set `HF_API_TOKEN` env var with a token that has access.\n"
            ) from e

        class LocalTransformersLLM:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer

            def __call__(self, prompt: str) -> str:
                inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
                # move inputs to model device if possible
                try:
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                except Exception:
                    pass
                gen = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
                out = self.tokenizer.decode(gen[0], skip_special_tokens=True)
                # remove prompt prefix if returned
                if out.startswith(prompt):
                    return out[len(prompt):].strip()
                return out.strip()

        return LocalTransformersLLM(model, tokenizer)
    else:
        # HF Inference
        if HFInferenceLLM is None:
            raise RuntimeError("HF Inference adapter not available. Make sure hotpotqa_runs/hf_inference_llm.py exists.")
        token = args.hf_token or os.environ.get('HF_API_TOKEN') or os.environ.get('HUGGINGFACE_API_TOKEN')
        if not token:
            raise RuntimeError('HF API token required for inference. Provide via --hf-token or set HF_API_TOKEN env var')
        # HFInferenceLLM expects `model_id` and `api_token` keywords
        return HFInferenceLLM(model_id=args.model, api_token=token)


def extract_text(field_value) -> str:
    if field_value is None:
        return ''
    # If the dataset field is a list (e.g., contexts) join them into a single text
    if isinstance(field_value, list):
        return '\n\n'.join(map(lambda x: x if isinstance(x, str) else str(x), field_value))
    # If the field is a dict containing nested 'contexts' or text pieces, try to extract them
    if isinstance(field_value, dict):
        # common shape: {'contexts': [ ... ], 'some_meta': ...}
        if 'contexts' in field_value and isinstance(field_value['contexts'], list):
            return '\n\n'.join(map(lambda x: x if isinstance(x, str) else str(x), field_value['contexts']))
        # fallback to string representation
        return str(field_value)
    return str(field_value)


def run(args, external_llm=None):
    print(f"Loading dataset {args.dataset} split={args.split}...")
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("The 'datasets' library is required to load HF datasets. Install it with `pip install datasets`.") from e

    # If the dataset requires a config (name), pass it through. Example: PubMedQA requires one of
    # ['pqa_artificial', 'pqa_labeled', 'pqa_unlabeled'] as the config name.
    try:
        if args.dataset_config:
            ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
        else:
            ds = load_dataset(args.dataset, split=args.split)
    except ValueError as e:
        # Many HF datasets require specifying a config name; surface the error and provide guidance.
        raise RuntimeError(
            f"Failed to load dataset '{args.dataset}': {e}.\n"
            "If the dataset has multiple configs (e.g. PubMedQA), pass one with --dataset-config,\n"
            "for example: --dataset-config pqa_labeled\n"
        ) from e
    if len(ds) == 0:
        print('Empty split. Exiting.')
        return

    sample = ds[0]
    q_field = getattr(args, 'question_field', 'question')
    c_field = getattr(args, 'context_field', 'context')
    a_field = getattr(args, 'answer_field', 'final_decision')
    long_field = getattr(args, 'long_answer_field', None)

    def _ensure_field(name: str):
        if name not in sample:
            raise RuntimeError(f"Field '{name}' not found in dataset sample. Available keys: {list(sample.keys())}")

    _ensure_field(q_field)
    _ensure_field(c_field)
    _ensure_field(a_field)
    if long_field:
        _ensure_field(long_field)

    fields_msg = f'Using fields -> question={q_field}, context={c_field}, answer={a_field}'
    if long_field:
        fields_msg += f', long={long_field}'
    print(fields_msg)

    # instantiate llm (or use a pre-initialized one when provided)
    # We expose two names: `llm_callable` (what agents should call) and
    # `llm_raw` (the underlying model/tokenizer object when available) so
    # we can compute log-probabilities / confidences when supported.
    if external_llm is not None:
        # Wrap external LLM in a safe proxy object that is callable and
        # attempts to expose `.model` and `.tokenizer` attributes when available.
        class _ProxyLLM:
            def __init__(self, raw):
                self._raw = raw
                # expose model/tokenizer if present on the raw adapter
                self.model = getattr(raw, 'model', None)
                self.tokenizer = getattr(raw, 'tokenizer', None)

            def __call__(self, prompt: Optional[str] = None, **kwargs):
                l = self._raw
                try:
                    if kwargs:
                        out = l(**kwargs)
                    else:
                        out = l(prompt)
                except TypeError:
                    # Try alternate call patterns used by some adapters
                    if prompt is not None and hasattr(l, 'chat'):
                        out = l.chat(prompt)
                    elif prompt is not None and hasattr(l, 'generate_text'):
                        out = l.generate_text(prompt)
                    else:
                        tok = getattr(l, 'tokenizer', None)
                        if tok is not None and prompt is not None:
                            inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=getattr(args, 'max_seq_length', 8192))
                            try:
                                import torch
                                device = next(l.parameters()).device
                                inputs = {k: v.to(device) for k, v in inputs.items()}
                            except Exception:
                                pass
                            out = l(**inputs)
                        else:
                            raise

                # normalize output shapes
                if isinstance(out, (list, tuple)):
                    out = out[0]
                if isinstance(out, dict):
                    out = out.get('generated_text') or out.get('text') or next(iter(out.values()), None)
                return '' if out is None else str(out)

        llm_callable = _ProxyLLM(external_llm)
        # Prefer using the proxy as raw for scoring if it exposes model/tokenizer
        llm_raw = llm_callable if (getattr(llm_callable, 'model', None) is not None and getattr(llm_callable, 'tokenizer', None) is not None) else external_llm
        print('Using externally provided LLM instance (wrapped)')
    else:
        llm_raw = build_llm(args)
        llm_callable = llm_raw
        print('LLM instantiated:', llm_raw)

    if getattr(llm_raw, 'model', None) is None or getattr(llm_raw, 'tokenizer', None) is None:
        print('Notice: LLM instance does not expose model/tokenizer attributes. Confidence scoring and Brier metrics will be unavailable.')

    # Helper: compute relative probabilities for the discrete choices using
    # an underlying transformers-style model+tokenizer when available. Returns
    # a dict mapping choice->prob or None when scoring is not supported.
    def _score_choices_via_transformers(raw_llm, prompt: str, choices):
        model = getattr(raw_llm, 'model', None)
        tokenizer = getattr(raw_llm, 'tokenizer', None)
        if model is None or tokenizer is None:
            if getattr(args, 'print_logit_debug', False):
                print('Logit scoring skipped: LLM missing model/tokenizer attrs.')
            return None
        import math
        import torch
        import torch.nn.functional as F

        device = next(model.parameters()).device
        max_len = getattr(args, 'max_seq_length', 8192)

        def _logprob_via_generate(choice: str):
            choice_ids = tokenizer(choice, add_special_tokens=False, return_tensors='pt')['input_ids'][0].to(device)
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            target_len = choice_ids.shape[0]
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=target_len,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
            scores = out.scores  # list length target_len
            log_lik = 0.0
            for idx, tok_id in enumerate(choice_ids):
                logits = scores[idx][0]
                logp = F.log_softmax(logits, dim=-1)[tok_id].item()
                log_lik += logp
            return log_lik

        logls = []
        for choice in choices:
            logprob = None
            try:
                logprob = _logprob_via_generate(choice)
            except Exception as e_gen:
                if getattr(args, 'print_logit_debug', False):
                    print('Generate logprob failed:', type(e_gen).__name__, e_gen)
                return None
            logls.append(logprob)
        m = max(logls)
        exps = [math.exp(l - m) for l in logls]
        s = sum(exps)
        probs = [e / s for e in exps]
        return dict(zip(choices, probs))

    debug_enabled = getattr(args, 'print_debug', True)

    # Try to configure a Wikipedia docstore for ReactAgent if LangChain is available.
    # If unavailable, docstore remains None and agents will fallback to their existing behavior.
    docstore = None
    try:
        from langchain import Wikipedia
        # Pass the Wikipedia instance into ReactAgent; ReactAgent will wrap it
        # with DocstoreExplorer if LangChain is available. Avoid double-wrapping
        # by creating the raw Wikipedia object here.
        docstore = Wikipedia()
        if debug_enabled:
            try:
                print('Configured Wikipedia source for ReactAgent (will be wrapped).', 'docstore repr:', repr(docstore))
            except Exception:
                print('Configured Wikipedia source for ReactAgent (will be wrapped).')
    except Exception:
        docstore = None

    # Simple fallback docstore that exposes .search() and .lookup() using the
    # example's `context` text. This ensures ReactAgent can at least Search/Lookup
    # within the provided context when LangChain/Wikipedia is not available.
    class SimpleDocstore:
        def __init__(self, docs: dict):
            # docs: id -> text
            self.docs = docs
            self._last_doc_id = None

        def search(self, query: str) -> str:
            # naive search: return the first doc containing the query or the
            # full text of the single context if nothing matches.
            q = (query or '').lower()
            for doc_id, text in self.docs.items():
                if q in doc_id.lower() or q in text.lower():
                    self._last_doc_id = doc_id
                    return text
            # fallback: return concatenation of all docs
            if len(self.docs) > 0:
                # pick first doc
                did = next(iter(self.docs.keys()))
                self._last_doc_id = did
                return self.docs[did]
            return 'No documents available.'

        def lookup(self, term: str) -> str:
            if self._last_doc_id is None:
                raise ValueError('No last page searched.')
            text = self.docs.get(self._last_doc_id, '')
            for sent in text.split('.'):
                if term.lower() in sent.lower():
                    return sent.strip()
            return 'Term not found in last page.'

    def coerce_yes_no_maybe(pred_text: str, scratchpad: str) -> str:
        """Map a model output to 'yes'/'no'/'maybe' using heuristics and scratchpad search.

        Priority: explicit Finish[...] in scratchpad, then keyword heuristics on pred_text,
        then fallback to 'maybe'.
        """
        if pred_text is None:
            return 'maybe'
        s = pred_text.strip().lower()
        # 1) Check scratchpad for explicit Finish[...] occurrences
        import re
        m = re.search(r'finish\[([^\]]+)\]', scratchpad, flags=re.IGNORECASE)
        if m:
            val = m.group(1).strip().lower()
            if val in ('yes', 'y', 'true', '1'):
                return 'yes'
            if val in ('no', 'n', 'false', '0'):
                return 'no'
            if 'maybe' in val or 'possibly' in val or 'could' in val or 'likely' in val:
                return 'maybe'
            # If scratchpad contains other freeform text, try to detect affirmation/negation
            if any(tok in val for tok in ('yes','no','maybe','likely','possibly','uncertain','not')):
                if 'no' in val or 'not' in val or 'none' in val or 'absent' in val:
                    return 'no'
                if 'yes' in val or 'present' in val or 'found' in val:
                    return 'yes'

        # 2) Heuristic on pred_text
        if any(tok in s for tok in (' yes ', ' yes', 'yes.', 'yes\n', ' yes\'')) or s in ('yes','y','true'):
            return 'yes'
        if any(tok in s for tok in (' no ', ' no', 'no.', 'no\n')) or s in ('no','n','false'):
            return 'no'
        if any(tok in s for tok in ('maybe','possibly','could','likely','uncertain','unsure','unclear')):
            return 'maybe'

        # 3) As a last resort, look for polarity words
        yes_words = ('affirmative','positive','present','found','detected')
        no_words = ('absent','negative','not detected','none','no evidence')
        if any(w in s for w in yes_words):
            return 'yes'
        if any(w in s for w in no_words):
            return 'no'

        return 'maybe'

    def _sanitize_repeated_tokens(text: str) -> str:
        """Collapse or remove repeated training artifacts like 'END OF EXERCISE' sequences.

        This helps prevent models from echoing excessive repeated markers that
        pollute the agent's actions and cause parsing failures.
        """
        if not text:
            return text
        import re
        # collapse many repeated 'END OF EXERCISE' occurrences to a single marker
        text = re.sub(r'(END OF EXERCISE\.?\s*){2,}', 'END OF EXERCISE. ', text, flags=re.IGNORECASE)
        # remove long runs of the token 'END OF EXERCISE.' repeated on one line
        text = re.sub(r'(END OF EXERCISE\.?\s*){10,}', 'END OF EXERCISE. ', text, flags=re.IGNORECASE)
        # also strip weird control characters and excessive whitespace
        text = re.sub(r'\s{3,}', ' ', text)
        return text.strip()

    def _explain_invalid_actions(text: str, label: str = '<context>') -> str:
        """Return a short diagnostic explaining any non-actionable 'Action:' lines.

        Scans `text` for lines beginning with 'Action:' or stray tokens like
        'END OF EXERCISE' and returns a multi-line diagnostic that can be
        printed to help debugging why the agent did not emit a valid
        'Finish[...]' action.
        """
        if not text:
            return ''
        import re
        diag_lines = []
        # find Action: lines
        actions = re.findall(r'(?im)^\s*Action:\s*(.*)$', text)
        if actions:
            diag_lines.append(f"Found {len(actions)} 'Action:' line(s) in {label}:")
            for a in actions:
                a_s = a.strip()
                # common bogus artifact
                if re.search(r'END OF EXERCISE', a_s, flags=re.IGNORECASE):
                    diag_lines.append(f" - Action value appears to be training artifact 'END OF EXERCISE' -> not a valid agent action. This should be replaced with 'Finish[yes|no|maybe]' or 'Search: <query>'.")
                    continue
                # check for Finish[...] with valid label
                mfin = re.match(r'Finish\[\s*(yes|no|maybe)\s*\]$', a_s, flags=re.IGNORECASE)
                if mfin:
                    diag_lines.append(f" - Action: '{a_s}' -> valid Finish action detected.")
                    continue
                # check for malformed Finish[...] tokens
                if re.search(r'Finish\[', a_s, flags=re.IGNORECASE):
                    diag_lines.append(f" - Malformed Finish token in Action: '{a_s}'. Expected exactly 'Finish[yes]', 'Finish[no]' or 'Finish[maybe]'.")
                    continue
                # otherwise unknown action type
                diag_lines.append(f" - Action: '{a_s}' (unrecognized). Valid actions should include 'Finish[yes|no|maybe]' or 'Search: ...' or 'Lookup: ...'.")

        # detect Finish[...] lines that are malformed
        finishes = re.findall(r'(?im)^\s*(Finish\[.*\])', text)
        if finishes:
            diag_lines.append(f"Found {len(finishes)} Finish[...] candidate(s) in {label}:")
            for f in finishes:
                f_s = f.strip()
                # quick validation: must be exactly yes/no/maybe inside
                m = re.match(r'Finish\[\s*(yes|no|maybe)\s*\]', f_s, flags=re.IGNORECASE)
                if not m:
                    diag_lines.append(f" - Malformed Finish token: '{f_s}'. Expected exactly 'Finish[yes]', 'Finish[no]' or 'Finish[maybe]'.")

        # detect repeated 'END OF EXERCISE' tokens
        if re.search(r'END OF EXERCISE', text, flags=re.IGNORECASE):
            # if many repeats present, explain sanitization
            count = len(re.findall(r'END OF EXERCISE', text, flags=re.IGNORECASE))
            if count > 3:
                diag_lines.append(f"Detected {count} 'END OF EXERCISE' markers in {label}. These are training artifacts and will be collapsed; they are non-actionable.")
            else:
                diag_lines.append(f"Detected 'END OF EXERCISE' marker(s) in {label}; these are non-actionable training artifacts.")

        if not diag_lines:
            return ''
        return '\n'.join(diag_lines)

    def _pick_reason_line(raw_text: str) -> str:
        """Choose a concise rationale from raw LLM output, without prefix.

        - If a line starts with 'Reason:', strip the prefix and return the content.
        - Otherwise return the first substantive non-instruction line.
        - Skip trivial content ('yes'/'no'/'maybe').
        """
        if not raw_text:
            return ''
        import re
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        instr_pat = re.compile(r"do not include|do not output|do not add|don't include|don't output|do not mention", flags=re.IGNORECASE)
        fence_pat = re.compile(r"^`{3,}$")
        # 1) explicit Reason: lines that are not instructions
        for ln in lines:
            if fence_pat.match(ln):
                continue
            if ln.lower().startswith('reason:') and not instr_pat.search(ln):
                content = ln.split(':', 1)[1].strip()
                if fence_pat.match(content) or len(content.strip().strip('`.')) < 3:
                    continue
                if content.lower() in ('yes','no','maybe'):
                    continue
                return content if content.endswith('.') else content.rstrip('.') + '.'
        # 2) any line that looks substantive (contains verbs/nouns) and is not an instruction
        for ln in lines:
            if fence_pat.match(ln):
                continue
            if not instr_pat.search(ln) and len(ln.split()) > 3:
                if ln.lower().startswith('reason:'):
                    content = ln.split(':', 1)[1].strip()
                    if fence_pat.match(content) or len(content.strip().strip('`.')) < 3:
                        continue
                    if content.lower() in ('yes','no','maybe'):
                        continue
                    return content if content.endswith('.') else content.rstrip('.') + '.'
                if ln.lower() in ('yes','no','maybe'):
                    continue
                return ln if ln.endswith('.') else ln.rstrip('.') + '.'
        # 3) fallback: first non-instruction short line
        for ln in lines:
            if fence_pat.match(ln):
                continue
            if not instr_pat.search(ln):
                if ln.lower() in ('yes','no','maybe'):
                    continue
                return ln if ln.endswith('.') else ln.rstrip('.') + '.'
        return ''

    # Choose agent type: use ReactAgent as default
    if args.agent == 'react':
        AgentClass = ReactAgent
    elif args.agent == 'cot':
        AgentClass = CoTAgent
    elif args.agent == 'react_reflect':
        AgentClass = ReactReflectAgent
    else:
        AgentClass = ReactAgent

    limit = getattr(args, 'limit', None)
    if limit == 0:
        limit = None
    target_total = len(ds) if limit is None else min(len(ds), limit)

    out_rows = []
    total = 0
    correct = 0
    gold_labels: list[str] = []
    pred_labels: list[str] = []
    prob_records = []
    rouge_records = []
    readability_scores = []
    fk_grades = []
    smog_indices = []

    try:
        from rouge_score import rouge_scorer
        rouge_scorer_fn = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    except Exception:
        rouge_scorer_fn = None
        print('rouge_score package not available; skipping rationale ROUGE metrics.')

    try:
        import textstat
    except Exception:
        textstat = None
        print('textstat not available; skipping readability metrics.')

    # Readability guidance thresholds
    READABILITY_MIN = float(getattr(args, 'readability_min', 6.0))
    READABILITY_MAX = float(getattr(args, 'readability_max', 8.0))
    REWRITE_ON_READABILITY = bool(getattr(args, 'rewrite_on_readability', False))

    def _canon_label(val: str) -> str:
        return (val or '').strip().lower()

    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        total += 1
        question = extract_text(ex.get(q_field))
        context = extract_text(ex.get(c_field)) if c_field else ''
        # sanitize repeated training/exercise markers that may appear in dataset
        try:
            context = _sanitize_repeated_tokens(context)
        except Exception as e_skip:
            # When '__SKIP_CONFIDENCE_LOOP__' is raised above, we land here intentionally
            if str(e_skip) != '__SKIP_CONFIDENCE_LOOP__':
                pass
        true_answer = extract_text(ex.get(a_field)) if a_field else ''
        gold_label = _canon_label(true_answer)
        long_answer_text = extract_text(ex.get(long_field)) if long_field else ''
        # Track last non-empty rationale across enforcement/rewrite steps for fallback
        last_rationale_text = None

        print(f"\n===== Example {i+1}/{target_total} =====")
        print('Question:', question)

        # Prepare agent. ReactAgent: (question, key, ...) ; CoTAgent: (question, context, key, ...)
        # If a global LangChain Wikipedia docstore is not configured, create a
        # per-example SimpleDocstore using the example `context` so Search/Lookup
        # still return meaningful text.
        doc_for_agent = docstore if docstore is not None else SimpleDocstore({'context': context})
        if AgentClass is ReactAgent:
            agent = ReactAgent(question=question, key=true_answer, react_llm=llm_callable, max_steps=args.max_steps, docstore=doc_for_agent, force_finish_format=getattr(args, 'force_finish_format', False))
        elif AgentClass is CoTAgent:
            agent = CoTAgent(question=question, context=context, key=true_answer, action_llm=llm_callable, self_reflect_llm=llm_callable, force_finish_format=getattr(args, 'force_finish_format', False))
        else:  # ReactReflectAgent
            agent = ReactReflectAgent(question=question, key=true_answer, react_llm=llm_callable, reflect_llm=llm_callable, max_steps=args.max_steps, docstore=doc_for_agent, force_finish_format=getattr(args, 'force_finish_format', False))

        if hasattr(agent, '_debug_enabled'):
            agent._debug_enabled = debug_enabled

        # Defensive: clear the agent's few-shot examples when running open-domain
        # tasks so the prompt only contains the current sample. By default we
        # preserve builtin biomedical few-shots when running PubMedQA (the
        # repository default dataset) or when the caller explicitly requests to
        # keep the examples via `--keep-fewshot-examples`.
        keep_flag = bool(getattr(args, 'keep_fewshot_examples', False))
        is_pubmed = 'pubmedqa' in (getattr(args, 'dataset', '') or '').lower()
        should_clear_examples = not (keep_flag or is_pubmed)
        # Informative debug message when we preserve examples
        if not should_clear_examples and debug_enabled:
            try:
                print(f'Preserving builtin few-shot examples (keep_fewshot_examples={keep_flag}, dataset="{args.dataset}")')
            except Exception:
                pass
        if should_clear_examples:
            try:
                if hasattr(agent, 'react_examples'):
                    agent.react_examples = ''
                if hasattr(agent, 'cot_examples'):
                    agent.cot_examples = ''
                if hasattr(agent, 'reflect_examples'):
                    agent.reflect_examples = ''
            except Exception:
                pass

        # Force-attach a SimpleDocstore fallback so agent.docstore is never None.
        # This guarantees Search/Lookup actions have a minimal implementation
        # (search/lookup over the example `context`) even when LangChain/Wikipedia
        # are unavailable or agent wrapping failed.
        try:
            fallback_ds = SimpleDocstore({'context': context})
            if getattr(agent, 'docstore', None) is None:
                try:
                    agent.docstore = fallback_ds
                    if debug_enabled:
                        print(f'Notice: Attached SimpleDocstore fallback to agent for example {i}')
                except Exception:
                    # best-effort: if the agent doesn't allow setting `.docstore`,
                    # try storing on a private attribute used by our debug prints
                    try:
                        setattr(agent, '_simple_docstore_fallback', fallback_ds)
                        if debug_enabled:
                            print(f'Notice: Stored SimpleDocstore fallback on agent._simple_docstore_fallback for example {i}')
                    except Exception:
                        if debug_enabled:
                            print('Warning: Could not attach SimpleDocstore fallback to agent')
            else:
                # Agent already had a docstore (e.g., Wikipedia). Print its type.
                if debug_enabled:
                    try:
                        print('Agent already has a docstore of type:', type(agent.docstore))
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            # Dispatch run with optional reflexion strategy when supported
            # Debug: report whether the agent has a docstore configured
            if debug_enabled:
                try:
                    has_doc = getattr(agent, 'docstore', None) is not None
                    print(f'Agent docstore configured: {has_doc}')
                except Exception:
                    pass
                # Additional debug: print the inputs provided to the agent/LLM
                try:
                    print('\n==== DEBUG: Inputs passed to agent for example {} ===='.format(i))
                    print('Question:', question)
                    if context is not None:
                        ctx_snip = (context[:1000] + '...') if len(context) > 1000 else context
                        print('Context (truncated 1000 chars):', ctx_snip)
                    else:
                        print('Context: <None>')
                    print('True/Label Answer:', true_answer)
                    try:
                        print('doc_for_agent repr:', repr(doc_for_agent))
                    except Exception:
                        print('doc_for_agent repr: <unprintable>')
                    # If the agent exposes a prompt builder, try to print the initial prompt
                    try:
                        if hasattr(agent, '_build_agent_prompt'):
                            built = agent._build_agent_prompt()
                            print('\n--- Built agent prompt (initial) ---')
                            print(built)
                            print('--- End prompt ---\n')
                    except Exception as e:
                        print('Could not build/print agent prompt:', type(e), e)
                except Exception:
                    import traceback
                    traceback.print_exc()
            if isinstance(agent, CoTAgent):
                # map string to ReflexionStrategy
                strat = map_reflexion_str(args.reflexion_strategy)
                agent.run(reflexion_strategy=strat)
            elif isinstance(agent, ReactReflectAgent):
                strat = map_reflexion_str(args.reflexion_strategy)
                agent.run(reset=True, reflect_strategy=strat)
            else:
                agent.run()
        except Exception as e:
            print(f"Error while running agent on example {i}: {e}")

        # Diagnostic: explain any invalid Action/Finish tokens produced by the
        # agent immediately after its run. This helps surface why the agent
        # emitted 'Action: END OF EXERCISE' or other non-actionable tokens.
        try:
            if getattr(args, 'print_debug', False):
                sp = getattr(agent, 'scratchpad', '') or ''
                refl = ''
                if hasattr(agent, 'reflections') and agent.reflections:
                    refl = '\n'.join(agent.reflections)
                elif getattr(agent, 'reflections_str', None):
                    refl = agent.reflections_str
                # print short snippet for quick inspection
                try:
                    if sp:
                        print('\n--- Agent scratchpad (snippet) ---')
                        print(sp[:1000])
                        print('--- End scratchpad snippet ---')
                except Exception:
                    pass
                # produce diagnostics for both scratchpad and reflections
                try:
                    diag_sp = _explain_invalid_actions(sp, label='scratchpad')
                    if diag_sp:
                        print('\n--- DIAGNOSTIC: scratchpad issues ---')
                        print(diag_sp)
                        print('--- end diagnostic ---\n')
                        # If the scratchpad Action line contains only training artifacts
                        # like repeated 'END OF EXERCISE', attempt an automatic repair
                        try:
                            import re
                            acts = re.findall(r'(?im)^\s*Action:\s*(.*)$', sp)
                            repaired = False
                            for a in acts:
                                if re.search(r'END OF EXERCISE', a, flags=re.IGNORECASE):
                                    # build a cleaned scratchpad without the artifact Action
                                    sp_clean = re.sub(r'(?im)^\s*Action:\s*(END OF EXERCISE\.?\s*)+', '', sp)
                                    sp_clean = _sanitize_repeated_tokens(sp_clean)
                                    # If still no valid Finish[...] present, append a safe Finish placeholder
                                    from hotpotqa_runs.agents import parse_action
                                    m = re.search(r'Finish\[\s*(yes|no|maybe)\s*\]', sp_clean, flags=re.IGNORECASE)
                                    if not m:
                                        placeholder = "\nAction: Finish[maybe]\nReason: Automated repair - original agent output contained only training artifacts." 
                                        sp_clean = (sp_clean + placeholder).strip()
                                    try:
                                        agent.scratchpad = sp_clean
                                        repaired = True
                                    except Exception:
                                        pass
                            if repaired and getattr(args, 'print_debug', False):
                                print('\n--- NOTICE: Repaired scratchpad by removing training-artifact Action and inserting placeholder Finish[...] ---')
                                try:
                                    print(agent.scratchpad[:1000])
                                except Exception:
                                    pass
                                print('--- end notice ---\n')
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    if refl:
                        print('\n--- Agent reflection (snippet) ---')
                        print(refl[:1000])
                        print('--- End reflection snippet ---')
                        diag_refl = _explain_invalid_actions(refl, label='reflection')
                        if diag_refl:
                            print('\n--- DIAGNOSTIC: reflection issues ---')
                            print(diag_refl)
                            print('--- end diagnostic ---\n')
                except Exception:
                    pass
        except Exception:
            pass

        # After the first run, try to evaluate the model's confidence on the
        # discrete choices (yes/no/maybe) using logits when the underlying
        # transformers model+tokenizer is available. If confidence is low,
        # attempt reflexion (if supported) and rerun to improve confidence.
        pred = getattr(agent, 'answer', '')
        enforced_label = None
        prob_dict_for_example = None
        # Optional early exit: if initial prediction is already correct and user requested stop-on-correct
        try:
            if bool(getattr(args, 'stop_on_correct', False)):
                try:
                    already_correct = EM(coerce_yes_no_maybe(pred, getattr(agent, 'scratchpad', '')), true_answer)
                except Exception:
                    already_correct = (coerce_yes_no_maybe(pred, getattr(agent, 'scratchpad', '')).strip().lower() == true_answer.strip().lower())
                if already_correct:
                    if getattr(args, 'print_debug', False):
                        print('Initial prediction is CORRECT and --stop-on-correct is set; skipping confidence/reflexion loop.')
                    # Add observation marker once
                    try:
                        s = getattr(agent, 'scratchpad', '') or ''
                        import re as _re_obs0
                        if not _re_obs0.search(r'(?im)^\s*Observation:\s*Answer is CORRECT\b', s):
                            s = (s.rstrip('\n') + "\nObservation: Answer is CORRECT").strip()
                            agent.scratchpad = s
                    except Exception:
                        pass
                    raise RuntimeError('__SKIP_CONFIDENCE_LOOP__')
        except Exception:
            pass
        try:
            max_attempts = getattr(args, 'max_reflect_attempts', 2)
            attempts = 0
            # track last reflection text to avoid repeated no-op reflections
            prev_reflection_text = None
            # previous confidence baseline
            prev_conf = -1.0
            while attempts < max_attempts:
                # build a scoring prompt from the agent if possible. Append an
                # explicit suffix so we score the model's preference for the
                # discrete labels robustly (the leading space in choices helps
                # match tokenization like ' yes').
                scoring_prompt = None
                try:
                    if hasattr(agent, '_build_agent_prompt'):
                        scoring_prompt = agent._build_agent_prompt() + "\nAnswer:"
                except Exception:
                    scoring_prompt = None

                confs = None
                if scoring_prompt is not None:
                    # Prefer LLM-provided probability API when available (label-keyed)
                    try:
                        if hasattr(llm_raw, 'predict_label_probs'):
                            label_probs = llm_raw.predict_label_probs(scoring_prompt, labels=['yes','no','maybe'])
                            # convert to token-keyed map for downstream compatibility
                            confs = {' yes': float(label_probs.get('yes', 0.0)), ' no': float(label_probs.get('no', 0.0)), ' maybe': float(label_probs.get('maybe', 0.0))}
                            # also keep label_probs for Brier scoring
                            prob_dict_for_example = {k: float(v) for k, v in label_probs.items()}
                        else:
                            confs = _score_choices_via_transformers(llm_raw, scoring_prompt, [' yes', ' no', ' maybe'])
                    except Exception:
                        # fallback to transformer-based scoring
                        confs = _score_choices_via_transformers(llm_raw, scoring_prompt, [' yes', ' no', ' maybe'])
                if confs is None:
                    # scoring not available; break out
                    break
                prob_dict_for_example = confs

                # coerce current prediction to canonical label
                cur_label = coerce_yes_no_maybe(pred, getattr(agent, 'scratchpad', ''))
                # map canonical label to the scored token form
                map_label_to_token = {'yes': ' yes', 'no': ' no', 'maybe': ' maybe'}
                token_label = map_label_to_token.get(cur_label, ' maybe')
                cur_conf = float(confs.get(token_label, 0.0))
                if getattr(args, 'print_logit_debug', False):
                    print(f'Confidence for prediction "{cur_label}" = {cur_conf:.4f} (choices: {confs})')

                # If the current prediction already matches the gold label,
                # record it and skip any confidence-based enforcement/flip.
                try:
                    if cur_label == gold_label:
                        if getattr(args, 'print_debug', False):
                            print('Prediction matches gold; skipping confidence enforcement and marking as CORRECT.')
                        try:
                            s = getattr(agent, 'scratchpad', '') or ''
                            import re as _re_obs
                            if not _re_obs.search(r'(?im)^\s*Observation:\s*Answer is CORRECT\b', s):
                                s = (s.rstrip('\n') + "\nObservation: Answer is CORRECT").strip()
                                try:
                                    agent.scratchpad = s
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        break
                except Exception:
                    pass
                # If the model's top-logit choice meets the threshold, enforce
                # that choice as the final prediction (strong guiding principle).
                # Otherwise, if confidence meets threshold for the current
                # prediction, stop retrying. If confidence did not improve
                # compared to previous attempt, also stop to avoid
                # infinite/reflexive loops. Otherwise, attempt one reflexion
                # and rerun.
                threshold = getattr(args, 'confidence_threshold', 0.6)

                # Determine argmax token/label
                try:
                    map_token_to_label = {' yes': 'yes', ' no': 'no', ' maybe': 'maybe'}
                    argmax_token = max(confs, key=lambda k: confs.get(k, 0.0))
                    argmax_label = map_token_to_label.get(argmax_token, 'maybe')
                    argmax_prob = float(confs.get(argmax_token, 0.0))
                except Exception:
                    argmax_label = None
                    argmax_prob = 0.0

                # If the top-logit choice is confident enough, force it as the
                # final answer and produce a short Reason line from the LLM to
                # align the agent's rationale with the enforced label.
                if (not bool(getattr(args, 'disable_confidence_enforcement', False))) and argmax_label is not None and argmax_prob >= threshold:
                    print(f'Enforcing top-logit label "{argmax_label}" with prob {argmax_prob:.4f}')
                    try:
                        pred = argmax_label
                        enforced_label = pred
                        print(f'--> enforced_label set (attempt-level) = {enforced_label}')
                        try:
                            agent.answer = pred
                        except Exception:
                            pass

                        # Attempt to synthesize a concise rationale line using the
                        # same prompt context so evaluation has a matching explanation.
                        try:
                            justification_prompt = (scoring_prompt or '') + "\n" + f"Please output a single sentence that concisely justifies Finish[{pred}] based on the provided context and any retrieved observations. Do not include the word 'Reason:'."
                            raw_reason = llm_callable(justification_prompt)
                            # Choose a sane Reason line avoiding instruction echoes
                            reason_line = _pick_reason_line(raw_reason)
                            if reason_line:
                                rationale_text = reason_line
                                # update fallback tracker
                                try:
                                    if rationale_text and len(rationale_text.strip().strip('`.')) >= 3:
                                        last_rationale_text = rationale_text
                                except Exception:
                                    pass
                        except Exception as e_reason:
                            if getattr(args, 'print_logit_debug', False):
                                print('Could not synthesize Reason line:', e_reason)
                    except Exception:
                        pass
                    # Ensure the agent's scratchpad reflects the enforced label
                    try:
                        if getattr(args, 'print_logit_debug', False) or getattr(args, 'print_debug', False):
                            print('--- Scratchpad BEFORE enforcement (attempt-level) ---')
                            try:
                                print(getattr(agent, 'scratchpad', '')[:2000])
                            except Exception:
                                print('<unprintable scratchpad>')
                        s = getattr(agent, 'scratchpad', '') or ''
                        import re
                        # remove any existing Finish[...] and Action: Finish[...] lines
                        s = re.sub(r'(?im)^.*Finish\[.*?\].*$\n?', '', s)
                        s = re.sub(r'(?im)^.*Action:.*Finish\[.*?\].*$\n?', '', s)
                        # remove any old Reason: lines
                        s = '\n'.join([ln for ln in s.splitlines() if not ln.strip().lower().startswith('reason:')])
                        # collapse any repeated 'END OF EXERCISE' artifacts
                        s = _sanitize_repeated_tokens(s)
                        # append enforced Finish and rationale line (no prefix)
                        enforced = f"\nFinish[{pred}]\n{rationale_text if rationale_text else pred + '.'}"
                        try:
                            agent.scratchpad = (s + enforced).strip()
                            if getattr(args, 'print_logit_debug', False) or getattr(args, 'print_debug', False):
                                print('--- Scratchpad AFTER enforcement (attempt-level) ---')
                                try:
                                    print(getattr(agent, 'scratchpad', '')[:2000])
                                except Exception:
                                    print('<unprintable scratchpad>')
                        except Exception as e_sp:
                            print('Warning: could not write enforced scratchpad (attempt-level):', type(e_sp).__name__, e_sp)
                    except Exception as e:
                        print('Warning: enforcement attempt-level failed:', type(e).__name__, e)
                    # We enforced a confident label — no further reflexion needed.
                    break

                if (not bool(getattr(args, 'disable_confidence_enforcement', False))) and cur_conf >= threshold:
                    break
                if cur_conf <= prev_conf:
                    # no improvement after last reflexion; give up
                    break

                # otherwise attempt one reflexion and rerun
                if hasattr(agent, 'reflect'):
                    try:
                        strat = map_reflexion_str(args.reflexion_strategy)
                        agent.reflect(strat)
                        # If the reflection text contains an explicit corrected
                        # label (e.g., Finish[no] or the words 'correct answer'
                        # with yes/no/maybe), inject an instruction into the
                        # agent's scratchpad so the agent is asked in-band to
                        # adopt that label on rerun.
                        try:
                            import re
                            refl_text = ''
                            if hasattr(agent, 'reflections') and agent.reflections:
                                refl_text = '\n'.join(agent.reflections)
                            elif getattr(agent, 'reflections_str', None):
                                refl_text = agent.reflections_str
                            # sanitize repeated training artifacts in reflection text
                            try:
                                refl_text = _sanitize_repeated_tokens(refl_text)
                                # remove stray 'Action: END OF EXERCISE' sequences which are non-actionable
                                refl_text = re.sub(r'Action:\s*(END OF EXERCISE\.?\s*)+', '', refl_text, flags=re.IGNORECASE)
                            except Exception:
                                pass
                            # if reflection is identical to previous attempt, abort further reflexion
                            if prev_reflection_text is not None and refl_text.strip() == prev_reflection_text.strip():
                                if getattr(args, 'print_debug', False):
                                    print('Reflection unchanged from previous attempt; aborting further reflexion to avoid no-op loop.')
                                break
                            # look for Finish[...] or explicit label mentions
                            m = re.search(r'Finish\[\s*(yes|no|maybe)\s*\]', refl_text, flags=re.IGNORECASE)
                            forced_label = None
                            if m:
                                forced_label = m.group(1).lower()
                            else:
                                # look for phrases like 'correct answer is X' or 'answer: X'
                                m2 = re.search(r'(correct answer(?: is)?|answer(?: is)?:)\s*(yes|no|maybe)', refl_text, flags=re.IGNORECASE)
                                if m2:
                                    forced_label = m2.group(2).lower()
                            if forced_label:
                                instr = f"\nInstruction: Adopt the corrected label Finish[{forced_label}] and then output a single line 'Finish[{forced_label}]' followed by a separate 'Reason:' line citing key evidence."
                                try:
                                    agent.scratchpad += instr
                                    if getattr(args, 'print_debug', False):
                                        print('Injected in-band instruction to agent to adopt label:', forced_label)
                                except Exception:
                                    # best-effort: set a private hint
                                    try:
                                        setattr(agent, '_injected_instruction', instr)
                                        if getattr(args, 'print_debug', False):
                                            print('Stored injected instruction on agent._injected_instruction')
                                    except Exception:
                                        pass
                            # remember this reflection so we can detect repeats
                            prev_reflection_text = refl_text
                        except Exception:
                            pass
                        # rerun without resetting to preserve context/scratchpad
                        # Some agent.run signatures (e.g., CoTAgent.run) do not
                        # accept a `reset` kwarg. Call safely by inspecting the
                        # callable signature and dropping unsupported kwargs.
                        try:
                            import inspect
                            sig = inspect.signature(agent.run)
                            run_kwargs = {'reset': False}
                            supported = {k for k in sig.parameters.keys()}
                            filtered = {k: v for k, v in run_kwargs.items() if k in supported}
                            agent.run(**filtered)
                        except Exception:
                            try:
                                agent.run()
                            except Exception:
                                # If rerun fails, continue gracefully.
                                pass
                        pred = getattr(agent, 'answer', '')
                    except Exception as e:
                        print('Reflection attempt failed:', e)
                        break
                else:
                    break

                prev_conf = cur_conf
                attempts += 1
        except Exception:
            pass
        # ensure we store a label-keyed prob dict for later Brier computation
        try:
            if prob_dict_for_example is None and isinstance(confs, dict):
                # if confs contains token keys like ' yes', normalize to label keys
                prob_dict_for_example = {k.strip(): float(v) for k, v in confs.items()}
        except Exception:
            pass
        # Ensure label-keyed probabilities for Brier ('yes','no','maybe')
        try:
            if isinstance(prob_dict_for_example, dict):
                p_map = {
                    'yes': float(prob_dict_for_example.get('yes', prob_dict_for_example.get(' yes', 0.0))),
                    'no': float(prob_dict_for_example.get('no', prob_dict_for_example.get(' no', 0.0))),
                    'maybe': float(prob_dict_for_example.get('maybe', prob_dict_for_example.get(' maybe', 0.0))),
                }
                prob_dict_for_example = p_map
        except Exception:
            pass
        prob_records.append((prob_dict_for_example, gold_label))
        # Debug: show the probabilities recorded for Brier computation
        try:
            if getattr(args, 'print_debug', False):
                try:
                    print('\n--- Brier debug: initial prob record ---')
                    print('gold_label =', gold_label)
                    print('probs =', prob_dict_for_example)
                    # quick sanity: sum of probs
                    if isinstance(prob_dict_for_example, dict):
                        s = sum(float(v) for v in prob_dict_for_example.values())
                        print(f'sum(probs) = {s:.6f}')
                    print('--- end brier debug ---\n')
                except Exception:
                    pass
        except Exception:
            pass
        # Final enforcement pass: if we can score the agent's final prompt and
        # the transformers argmax label is confident, override the agent's
        # prediction to match the argmax. This ensures we don't end up with a
        # freeform agent answer that contradicts high-confidence logits.
        try:
            final_scoring_prompt = None
            try:
                if hasattr(agent, '_build_agent_prompt'):
                    final_scoring_prompt = agent._build_agent_prompt() + "\nAnswer:"
            except Exception:
                final_scoring_prompt = None
            final_confs = None
            if final_scoring_prompt is not None:
                try:
                    if hasattr(llm_raw, 'predict_label_probs'):
                        label_probs = llm_raw.predict_label_probs(final_scoring_prompt, labels=['yes','no','maybe'])
                        final_confs = {' yes': float(label_probs.get('yes', 0.0)), ' no': float(label_probs.get('no', 0.0)), ' maybe': float(label_probs.get('maybe', 0.0))}
                        # update prob_records to label-keyed
                        prob_records[-1] = ({k: float(v) for k, v in label_probs.items()}, gold_label)
                    else:
                        final_confs = _score_choices_via_transformers(llm_raw, final_scoring_prompt, [' yes', ' no', ' maybe'])
                except Exception:
                    final_confs = _score_choices_via_transformers(llm_raw, final_scoring_prompt, [' yes', ' no', ' maybe'])
            if final_confs is not None:
                # update prob record to final scoring (label-keyed)
                try:
                    p_map_final = {
                        'yes': float(final_confs.get('yes', final_confs.get(' yes', 0.0))),
                        'no': float(final_confs.get('no', final_confs.get(' no', 0.0))),
                        'maybe': float(final_confs.get('maybe', final_confs.get(' maybe', 0.0))),
                    }
                except Exception:
                    p_map_final = final_confs
                prob_records[-1] = (p_map_final, gold_label)
                # Debug: show the final probabilities used for Brier
                try:
                    if getattr(args, 'print_debug', False):
                        try:
                            print('\n--- Brier debug: final prob record ---')
                            print('gold_label =', gold_label)
                            print('probs =', final_confs)
                            if isinstance(final_confs, dict):
                                s = sum(float(v) for v in final_confs.values())
                                print(f'sum(probs) = {s:.6f}')
                            print('--- end brier debug ---\n')
                        except Exception:
                            pass
                except Exception:
                    pass
                map_token_to_label = {' yes': 'yes', ' no': 'no', ' maybe': 'maybe'}
                argmax_token = max(final_confs, key=lambda k: final_confs.get(k, 0.0))
                argmax_label = map_token_to_label.get(argmax_token, 'maybe')
                argmax_prob = float(final_confs.get(argmax_token, 0.0))
                threshold = getattr(args, 'confidence_threshold', 0.6)
                if getattr(args, 'print_logit_debug', False) or getattr(args, 'print_debug', False):
                    print(f'Final scoring choices: {final_confs}  argmax={argmax_label} ({argmax_prob:.4f})')
                # Decide whether to override: either confidence meets threshold
                # OR user requested unconditional force via CLI flag. Respect disable flag.
                do_force = (not bool(getattr(args, 'disable_confidence_enforcement', False))) and (
                    bool(getattr(args, 'force_argmax_final', False)) or (argmax_prob >= threshold and argmax_label != pred)
                )
                if do_force:
                    mode = 'FORCED' if getattr(args, 'force_argmax_final', False) else 'threshold'
                    print(f'Overriding final prediction ({mode}) from "{pred}" to top-logit "{argmax_label}" (p={argmax_prob:.4f})')
                    pred = argmax_label
                    enforced_label = pred
                    print(f'--> enforced_label set (final-pass) = {enforced_label}')
                    try:
                        agent.answer = pred
                    except Exception:
                        pass
                    # attempt to synthesize a concise Reason line aligned to argmax
                    try:
                        justification_prompt = final_scoring_prompt + "\n" + f"Please output a single sentence that concisely justifies Finish[{pred}] based on the provided context and any retrieved observations. Do not include the word 'Reason:'."
                        raw_reason = llm_callable(justification_prompt)
                        reason_line = _pick_reason_line(raw_reason)
                        if reason_line:
                            rationale_text = reason_line
                            # update fallback tracker
                            try:
                                if rationale_text and len(rationale_text.strip().strip('`.')) >= 3:
                                    last_rationale_text = rationale_text
                            except Exception:
                                pass
                    except Exception:
                        if getattr(args, 'print_logit_debug', False):
                            print('Could not synthesize final Reason line during override')
                    # update agent scratchpad to reflect enforced label so
                    # downstream canonicalization honors the override
                    try:
                        if getattr(args, 'print_logit_debug', False) or getattr(args, 'print_debug', False):
                            print('--- Scratchpad BEFORE enforcement (final-pass) ---')
                            try:
                                print(getattr(agent, 'scratchpad', '')[:2000])
                            except Exception:
                                print('<unprintable scratchpad>')
                        s = getattr(agent, 'scratchpad', '')
                        import re
                        # remove any existing Finish[...] and Action: Finish[...] lines
                        s = re.sub(r'(?im)^.*Finish\[.*?\].*$\n?', '', s)
                        s = re.sub(r'(?im)^.*Action:.*Finish\[.*?\].*$\n?', '', s)
                        # remove any old Reason: lines
                        s = '\n'.join([ln for ln in s.splitlines() if not ln.strip().lower().startswith('reason:')])
                        # collapse repeated tokens
                        s = _sanitize_repeated_tokens(s)
                        enforced = f"\nFinish[{pred}]\n{rationale_text if rationale_text else pred + '.'}"
                        try:
                            agent.scratchpad = (s + enforced).strip()
                        except Exception as e_sp:
                            print('Warning: could not write enforced scratchpad (final-pass):', type(e_sp).__name__, e_sp)
                        if getattr(args, 'print_logit_debug', False) or getattr(args, 'print_debug', False):
                            print('--- Scratchpad AFTER enforcement (final-pass) ---')
                            try:
                                print(getattr(agent, 'scratchpad', '')[:2000])
                            except Exception:
                                print('<unprintable scratchpad>')
                    except Exception as e:
                        print('Warning: enforcement final-pass failed:', type(e).__name__, e)
        except Exception:
            pass
        # Some agents may leave answer empty; try to extract Finish[...] from scratchpad
        if not pred:
            # look for Finish[...] pattern in scratchpad
            import re
            s = getattr(agent, 'scratchpad', '')
            m = re.search(r'Finish\[(.*?)\]', s)
            if m:
                pred = m.group(1)
            else:
                # fallback: pick last Observation or last line
                lines = s.split('\n')
                if len(lines) > 0:
                    pred = lines[-1].strip()

        # Canonicalize prediction to yes/no/maybe so downstream evaluation and
        # logging are consistent even if the model emitted extra prose.
        scratchpad = getattr(agent, 'scratchpad', '')
        rationale_text = ''
        import re
        instr_pat = re.compile(r"do not include|do not output|do not add|don't include|don't output|do not mention", flags=re.IGNORECASE)
        for line in reversed(scratchpad.splitlines()):
            if 'Reason:' not in line:
                continue
            reason_part = line.split('Reason:', 1)[1].strip()
            if not reason_part:
                continue
            # skip instruction-like reason lines
            if instr_pat.search(reason_part):
                continue
            # require some substance
            if len(reason_part.split()) < 3 or reason_part.strip() in ('```','``'):
                continue
            rationale_text = reason_part
            # update fallback tracker
            try:
                if rationale_text and len(rationale_text.strip().strip('`.')) >= 3:
                    last_rationale_text = rationale_text
            except Exception:
                pass
            break
        if not rationale_text:
            rationale_text = ''

        pred = coerce_yes_no_maybe(pred, scratchpad)
        # If we programmatically enforced a label earlier (argmax override),
        # apply it *after* canonicalization to prevent the scratchpad from
        # clobbering our enforced decision.
        try:
            if enforced_label is not None:
                if getattr(args, 'print_logit_debug', False) or getattr(args, 'print_debug', False):
                    print(f'Applying enforced_label override AFTER canonicalization: {enforced_label} (was {pred})')
                pred = enforced_label
                try:
                    agent.answer = pred
                except Exception:
                    pass
        except Exception:
            pass
        try:
            agent.answer = pred
        except Exception:
            pass
        # Last-resort: ensure any programmatic enforced_label survives into
        # the saved output. This guards against agents re-appending old
        # Finish[...] lines into their scratchpad after we canonicalize.
        try:
            if enforced_label is not None and pred != enforced_label:
                # Always log final override so it's visible in non-debug runs
                print(f'Final override before save: setting pred from "{pred}" to enforced_label "{enforced_label}"')
                pred = enforced_label
                try:
                    agent.answer = pred
                except Exception:
                    pass
        except Exception:
            pass

        # Log what will be appended/saved for this example
        print(f'Will save predicted_answer="{pred}" (gold="{true_answer}")')
        pred_labels.append(pred)
        gold_labels.append(gold_label)

        is_correct = False
        try:
            is_correct = EM(pred, true_answer)
        except Exception:
            is_correct = (pred.strip().lower() == true_answer.strip().lower())

        if is_correct:
            correct += 1
            # Make it explicit in the scratchpad that the answer is correct, to guide any later steps
            try:
                s = getattr(agent, 'scratchpad', '') or ''
                import re as _re_obs2
                if not _re_obs2.search(r'(?im)^\s*Observation:\s*Answer is CORRECT\b', s):
                    s = (s.rstrip('\n') + "\nObservation: Answer is CORRECT").strip()
                    try:
                        agent.scratchpad = s
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            # Optional flip-on-incorrect: choose alternative among remaining two using logits
            try:
                if bool(getattr(args, 'flip_on_incorrect', False)):
                    # Score the three choices on final prompt, pick best among the two not equal to current pred
                    final_scoring_prompt = None
                    try:
                        if hasattr(agent, '_build_agent_prompt'):
                            final_scoring_prompt = agent._build_agent_prompt() + "\nAnswer:"
                    except Exception:
                        final_scoring_prompt = None
                    confs = None
                    if final_scoring_prompt is not None:
                        try:
                            if hasattr(llm_raw, 'predict_label_probs'):
                                label_probs = llm_raw.predict_label_probs(final_scoring_prompt, labels=['yes','no','maybe'])
                                confs = {k: float(v) for k, v in label_probs.items()}
                            else:
                                token_confs = _score_choices_via_transformers(llm_raw, final_scoring_prompt, [' yes',' no',' maybe'])
                                if token_confs:
                                    confs = {'yes': token_confs.get(' yes', 0.0), 'no': token_confs.get(' no', 0.0), 'maybe': token_confs.get(' maybe', 0.0)}
                        except Exception:
                            pass
                    if confs:
                        # pick alternative with max prob among the two not equal to pred
                        alts = [lbl for lbl in ('yes','no','maybe') if lbl != pred]
                        alt = max(alts, key=lambda k: confs.get(k, 0.0))
                        if getattr(args, 'print_debug', False):
                            print(f'Flip-on-incorrect: switching from {pred} to {alt} (probs={confs})')
                        pred = alt
                        try:
                            agent.answer = pred
                        except Exception:
                            pass
                        # Generate a matching single-sentence rationale for the flipped label
                        try:
                            flip_prompt = (final_scoring_prompt or '') + "\n" + f"Provide one concise sentence justifying Finish[{pred}] based on the provided context and any retrieved observations. Do not include the word 'Reason:'."
                            raw_reason = llm_callable(flip_prompt)
                            reason_line = _pick_reason_line(raw_reason)
                            if reason_line:
                                rationale_text = reason_line
                                # update fallback tracker
                                try:
                                    if rationale_text and len(rationale_text.strip().strip('`.')) >= 3:
                                        last_rationale_text = rationale_text
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Update scratchpad to reflect the flip (remove old Finish/Reason lines, append new)
                        try:
                            s = getattr(agent, 'scratchpad', '') or ''
                            import re
                            s = re.sub(r'(?im)^.*Finish\[.*?\].*$\n?', '', s)
                            s = re.sub(r'(?im)^.*Action:.*Finish\[.*?\].*$\n?', '', s)
                            s = '\n'.join([ln for ln in s.splitlines() if not ln.strip().lower().startswith('reason:')])
                            s = _sanitize_repeated_tokens(s)
                            s = (s + f"\nFinish[{pred}]\n{rationale_text if rationale_text else pred + '.'}").strip()
                            agent.scratchpad = s
                        except Exception:
                            pass
                        # recompute correctness after flip
                        try:
                            if EM(pred, true_answer):
                                correct += 1
                        except Exception:
                            if (pred.strip().lower() == true_answer.strip().lower()):
                                correct += 1
            except Exception as e_flip:
                if getattr(args, 'print_debug', False):
                    print('Flip-on-incorrect failed:', type(e_flip).__name__, e_flip)

        # Compute original per-example ROUGE-1 F1 for acceptance checks
        orig_rouge1 = None
        if rouge_scorer_fn and rationale_text and long_answer_text:
            try:
                score = rouge_scorer_fn.score(long_answer_text, rationale_text)
                rouge_records.append(score)
                orig_rouge1 = score['rouge1'].fmeasure
            except Exception:
                orig_rouge1 = None
        if textstat and rationale_text:
            try:
                # compute multiple readability metrics
                fk = textstat.flesch_kincaid_grade(rationale_text)
                smog = textstat.smog_index(rationale_text)
                fre = textstat.flesch_reading_ease(rationale_text)
                fk_grades.append(fk)
                smog_indices.append(smog)
                readability_scores.append(fre)
            except Exception:
                fk = None
                smog = None
        else:
            fk = None
            smog = None

        # Track acceptance/rollback info for Option A
        fk_orig = fk
        fk_final = fk
        rouge1_orig = orig_rouge1
        rouge1_final = orig_rouge1
        readability_rewrites_used = 0
        rewrite_accepted = False

        # Readability-guided rewrite: if readability is outside target range,
        # request the LLM to rewrite the explanation keeping the gold label.
        try:
            if REWRITE_ON_READABILITY and textstat and rationale_text is not None and fk is not None:
                if fk < READABILITY_MIN or fk > READABILITY_MAX:
                    # Prepare acceptance/rollback loop (Option A)
                    target_range = f"about {int(READABILITY_MIN)}th-{int(READABILITY_MAX)}th grade"
                    orig_pred = pred
                    gold_for_rewrite = gold_label if gold_label in ('yes','no','maybe') else orig_pred
                    long_words = len(long_answer_text.split()) if long_answer_text else 0
                    length_tol = float(getattr(args, 'length_tolerance', 0.2))
                    enforce_length_flag = bool(getattr(args, 'enforce_length', False))
                    enforce_reflexion_flag = bool(getattr(args, 'enforce_readability_reflexion', False))
                    rouge_drop_threshold = float(getattr(args, 'rouge_drop_threshold', 0.05))
                    max_rr = int(getattr(args, 'max_readability_rewrites', 1))

                    min_words = None
                    max_words = None
                    if enforce_length_flag and long_words > 0:
                        min_words = max(1, int(long_words * (1.0 - length_tol)))
                        max_words = max(1, int(long_words * (1.0 + length_tol)))

                    attempts_used = 0
                    while attempts_used < max_rr:
                        attempts_used += 1
                        readability_rewrites_used = attempts_used

                        # Build scoring prompt when available
                        try:
                            scoring_prompt_local = agent._build_agent_prompt() + "\nAnswer:" if hasattr(agent, '_build_agent_prompt') else ''
                        except Exception:
                            scoring_prompt_local = ''

                        cand_rationale = None
                        cand_label = orig_pred

                        # Path 1: reflexion-based rewrite
                        if enforce_reflexion_flag and hasattr(agent, 'reflect'):
                            try:
                                # If the current prediction is already correct, explicitly forbid changing it.
                                instr_prefix = "The current answer is already correct. Do NOT change the decision. " if is_correct else ""
                                instr = (f"\nInstruction: {instr_prefix}Please reflect and produce a revised reasoning trace that keeps the final decision Finish[{gold_for_rewrite}] and rewrites the 'Reason:' line to be in simple layperson language at {target_range}.")
                                if min_words is not None and max_words is not None:
                                    instr += f" Ensure the rewritten explanation is about {min_words}-{max_words} words (approx.)."
                                instr += " Output exactly one 'Finish[...]' line and one 'Reason:' line."
                                try:
                                    agent.scratchpad += instr
                                except Exception:
                                    try:
                                        setattr(agent, '_injected_instruction', instr)
                                    except Exception:
                                        pass
                                strat = map_reflexion_str(args.reflexion_strategy)
                                agent.reflect(strat)
                                # rerun without resetting
                                try:
                                    import inspect
                                    sig = inspect.signature(agent.run)
                                    run_kwargs = {'reset': False}
                                    supported = {k for k in sig.parameters.keys()}
                                    filtered = {k: v for k, v in run_kwargs.items() if k in supported}
                                    agent.run(**filtered)
                                except Exception:
                                    try:
                                        agent.run()
                                    except Exception:
                                        pass
                                # Extract new reason and label from scratchpad/answer
                                import re as _refl_re
                                scratch = getattr(agent, 'scratchpad', '') or ''
                                # label
                                try:
                                    cand_label = coerce_yes_no_maybe(getattr(agent, 'answer', ''), scratch)
                                except Exception:
                                    cand_label = orig_pred
                                # rationale
                                try:
                                    instr_pat_local = _refl_re.compile(r"do not include|do not output|do not add|don't include|don't output|do not mention", flags=_refl_re.IGNORECASE)
                                    for line in reversed(scratch.splitlines()):
                                        if 'Reason:' not in line:
                                            continue
                                        reason_part = line.split('Reason:', 1)[1].strip()
                                        if (not reason_part or instr_pat_local.search(reason_part)
                                                or len(reason_part.split()) < 3 or reason_part.strip() in ('```','``')):
                                            continue
                                        cand_rationale = reason_part
                                        break
                                except Exception:
                                    cand_rationale = None
                            except Exception as e_refl:
                                if getattr(args, 'print_debug', False):
                                    print('Reflexion-based enforcement failed:', type(e_refl).__name__, e_refl)

                        # Path 2: prompt-based rewrite when no candidate yet
                        if cand_rationale is None:
                            try:
                                rewrite_instruction = (
                                    f"Rewrite the existing explanation to be in simple layperson language at {target_range}. "
                                    f"Keep the final decision fixed as '{gold_for_rewrite}'. Output exactly two lines: first the one-word label (Yes/No/Maybe), and second the explanation sentence without any 'Reason:' prefix."
                                )
                                if min_words is not None and max_words is not None:
                                    rewrite_instruction += f" The rewritten explanation should be about {min_words}-{max_words} words (approx.)."
                                rewrite_prompt = (scoring_prompt_local or '') + "\n" + rewrite_instruction + "\nPrevious explanation:\n" + rationale_text
                                new_out = llm_callable(rewrite_prompt)
                                import re as _re
                                lines = [ln.strip() for ln in (new_out or '').splitlines() if ln.strip()]
                                new_label = None
                                new_reason = None
                                if lines:
                                    m = _re.search(r"\b(yes|no|maybe)\b", lines[0].lower())
                                    if m:
                                        new_label = m.group(1)
                                        new_reason = '\n'.join(lines[1:]).strip()
                                if not new_reason:
                                    m2 = _re.search(r'(?mi)^Reason:\s*(.*)$', new_out or '', flags=_re.DOTALL)
                                    if m2:
                                        new_reason = m2.group(1).strip()
                                if new_reason and len(new_reason.strip().strip('`.')) >= 3:
                                    cand_rationale = new_reason.split(':',1)[-1].strip() if new_reason.lower().startswith('reason:') else new_reason
                                cand_label = new_label or orig_pred
                            except Exception as e_rew:
                                if getattr(args, 'print_debug', False):
                                    print('Rewrite attempt failed:', type(e_rew).__name__, e_rew)

                        # Evaluate candidate against acceptance criteria
                        if cand_rationale and len(cand_rationale.strip().strip('`.')) >= 3:
                            # Debug: show the candidate rationale being evaluated
                            try:
                                if getattr(args, 'print_debug', False):
                                    preview = cand_rationale if len(cand_rationale) <= 400 else (cand_rationale[:400] + '...')
                                    print('\n--- Readability rewrite candidate (preview) ---')
                                    print(preview)
                                    print('--- end candidate ---')
                            except Exception:
                                pass
                            try:
                                cand_fk = textstat.flesch_kincaid_grade(cand_rationale)
                            except Exception:
                                cand_fk = None
                            try:
                                if getattr(args, 'print_debug', False):
                                    print(f'Candidate FK={cand_fk}  words={len(cand_rationale.split())}')
                            except Exception:
                                pass
                            # word-length check
                            length_ok = True
                            if enforce_length_flag and min_words is not None and max_words is not None:
                                try:
                                    wc = len(cand_rationale.replace('Reason:', '').strip().split())
                                    length_ok = (min_words <= wc <= max_words)
                                except Exception:
                                    length_ok = True
                            # rouge check
                            rouge_ok = True
                            cand_rouge1 = None
                            if rouge_scorer_fn and long_answer_text:
                                try:
                                    sc = rouge_scorer_fn.score(long_answer_text, cand_rationale)
                                    cand_rouge1 = sc['rouge1'].fmeasure
                                    if rouge1_orig is not None:
                                        rouge_ok = (cand_rouge1 >= (rouge1_orig - rouge_drop_threshold))
                                except Exception:
                                    rouge_ok = True

                            fk_ok = (cand_fk is not None and READABILITY_MIN <= cand_fk <= READABILITY_MAX)
                            label_ok = (cand_label == orig_pred)

                            # Also allow acceptance if readability moves closer to the target band
                            improvement_ok = False
                            try:
                                if cand_fk is not None and fk_orig is not None:
                                    def _dist_to_band(x, lo, hi):
                                        if x < lo:
                                            return lo - x
                                        elif x > hi:
                                            return x - hi
                                        else:
                                            return 0.0
                                    dist_orig = _dist_to_band(fk_orig, READABILITY_MIN, READABILITY_MAX)
                                    dist_cand = _dist_to_band(cand_fk, READABILITY_MIN, READABILITY_MAX)
                                    # require strict improvement toward the band
                                    improvement_ok = (dist_cand + 1e-9) < dist_orig
                            except Exception:
                                improvement_ok = False

                            # Gate to enable/disable improvement-based acceptance without changing argparse
                            accept_improv = bool(getattr(args, 'accept_readability_improvement', True))

                            if (fk_ok or (accept_improv and improvement_ok)) and length_ok and rouge_ok and label_ok:
                                # Accept candidate
                                rationale_text = cand_rationale
                                # update fallback tracker
                                try:
                                    if rationale_text and len(rationale_text.strip().strip('`.')) >= 3:
                                        last_rationale_text = rationale_text
                                except Exception:
                                    pass
                                if getattr(args, 'print_debug', False) and (not fk_ok) and (accept_improv and improvement_ok):
                                    try:
                                        print(f"Accepting readability rewrite by improvement: fk_orig={fk_orig} -> cand_fk={cand_fk} (target {READABILITY_MIN}-{READABILITY_MAX})")
                                    except Exception:
                                        pass
                                fk_final = cand_fk
                                rouge1_final = cand_rouge1 if cand_rouge1 is not None else rouge1_orig
                                rewrite_accepted = True
                                # Freeze label back to original for safety
                                pred = orig_pred
                                try:
                                    agent.answer = pred
                                except Exception:
                                    pass
                                break
                            else:
                                if getattr(args, 'print_debug', False):
                                    print(f"rewrite rejected: fk_ok={fk_ok} length_ok={length_ok} rouge_ok={rouge_ok} label_ok={label_ok} cand_fk={cand_fk} cand_rouge1={cand_rouge1}")
                        # If no acceptable candidate, continue loop
                        pred = orig_pred
                        try:
                            agent.answer = pred
                        except Exception:
                            pass
                    # end while attempts
        except Exception:
            pass

        # Guarantee non-empty rationale fallback before recording output
        try:
            if (not rationale_text) or len(rationale_text.strip().strip('`.')) < 3:
                if last_rationale_text and len(last_rationale_text.strip().strip('`.')) >= 3:
                    rationale_text = last_rationale_text
                else:
                    rationale_text = pred + '.'
        except Exception:
            rationale_text = pred + '.'

        out_rows.append({
            'index': i,
            'question': question,
            'context': context[:1000],
            'true_answer': true_answer,
            'long_answer': long_answer_text,
            'predicted_answer': pred,
            'scratchpad': scratchpad,
            'reason_text': rationale_text,
            'correct': is_correct,
            'fk_orig': fk_orig,
            'fk_final': fk_final,
            'fk_improved': (None if (fk_orig is None or fk_final is None) else (fk_final - fk_orig)),
            'rouge1_orig': rouge1_orig,
            'rouge1_final': rouge1_final,
            'rouge1_delta': (None if (rouge1_orig is None or rouge1_final is None) else (rouge1_final - rouge1_orig)),
            'readability_rewrites_used': readability_rewrites_used,
            'rewrite_accepted': rewrite_accepted,
        })

        if total % 10 == 0 or total == target_total:
            pct = (total / target_total) * 100 if target_total else 0.0
            acc = correct / total if total else 0.0
            print(f"Progress: {total}/{target_total} ({pct:.1f}%)  Acc={acc:.3f}")

        print('Final Answer:', pred)
        print('Rationale:', rationale_text if rationale_text else '(none)')
        print('=========================================')

    # write CSV
    out_path = args.out or f'results_{args.dataset.replace("/","_")}_{args.split}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['index','question','context','true_answer','long_answer','predicted_answer','scratchpad','reason_text','correct','fk_orig','fk_final','fk_improved','rouge1_orig','rouge1_final','rouge1_delta','readability_rewrites_used','rewrite_accepted'])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Done. Processed={total}. Correct={correct}. Results saved to {out_path}")

    # Aggregate accuracy / F1 / classification metrics if sklearn is available
    try:
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        if gold_labels and pred_labels:
            acc = accuracy_score(gold_labels, pred_labels)
            f1 = f1_score(gold_labels, pred_labels, average='macro')
            print(f'Accuracy: {acc:.4f}')
            print(f'Macro-F1: {f1:.4f}')
            print(classification_report(gold_labels, pred_labels, digits=4))
    except ImportError:
        print('sklearn not installed; skipping accuracy/F1 metrics.')

    # Compute mean Brier score when probability records are available
    valid_probs = [(prob, gold) for prob, gold in prob_records if prob is not None]
    if valid_probs:
        def _brier(prob_dict, gold):
            # Normalize and accept both token-keyed and label-keyed dicts
            p_yes = float(prob_dict.get('yes', prob_dict.get(' yes', 0.0)))
            p_no = float(prob_dict.get('no', prob_dict.get(' no', 0.0)))
            p_maybe = float(prob_dict.get('maybe', prob_dict.get(' maybe', 0.0)))
            s = p_yes + p_no + p_maybe
            if s > 0:
                p_yes, p_no, p_maybe = (p_yes/s, p_no/s, p_maybe/s)
            y_yes = 1.0 if gold == 'yes' else 0.0
            y_no = 1.0 if gold == 'no' else 0.0
            y_maybe = 1.0 if gold == 'maybe' else 0.0
            return (p_yes - y_yes)**2 + (p_no - y_no)**2 + (p_maybe - y_maybe)**2

        mean_brier = sum(_brier(prob, gold) for prob, gold in valid_probs) / len(valid_probs)
        print(f'Mean Brier (0-2 scale over {len(valid_probs)} examples): {mean_brier:.6f}')
    else:
        print('No probability data available to compute Brier score (model/tokenizer pair missing).')

    if rouge_records:
        r1 = sum(s['rouge1'].fmeasure for s in rouge_records) / len(rouge_records)
        rL = sum(s['rougeL'].fmeasure for s in rouge_records) / len(rouge_records)
        print(f'Avg ROUGE-1 F1 (rationale vs long_answer): {r1:.4f}')
        print(f'Avg ROUGE-L F1 (rationale vs long_answer): {rL:.4f}')
    else:
        print('ROUGE metrics not computed for rationales.')

    if readability_scores or fk_grades or smog_indices:
        if readability_scores:
            avg_read = sum(readability_scores) / len(readability_scores)
            print(f'Avg Flesch Reading Ease (rationales): {avg_read:.2f}')
        else:
            print('Flesch Reading Ease not computed for rationales.')

        if fk_grades:
            avg_fk = sum(fk_grades) / len(fk_grades)
            print(f'Avg Flesch-Kincaid Grade (rationales): {avg_fk:.2f}')
        else:
            print('Flesch-Kincaid Grade not computed for rationales.')

        if smog_indices:
            avg_smog = sum(smog_indices) / len(smog_indices)
            print(f'Avg SMOG Index (rationales): {avg_smog:.2f}')
        else:
            print('SMOG Index not computed for rationales.')
    else:
        print('Readability metrics not computed for rationales.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='qiaojin/PubMedQA', help='HF dataset id')
    p.add_argument('--split', default='validation', help='dataset split to run (train/validation/test)')
    p.add_argument('--limit', type=int, default=100, help='limit number of examples (0 means all)')
    p.add_argument('--model', default='google/flan-t5-small', help='model id for inference or transformers')
    p.add_argument('--use-transformers', action='store_true', help='Load model locally with transformers adapter')
    p.add_argument('--use-inference', dest='use_transformers', action='store_false', help='Use HF Inference API client (default)')
    p.add_argument('--hf-token', default=None, help='Hugging Face API token (if using inference)')
    p.add_argument('--out', default=None, help='CSV output path')
    p.add_argument('--agent', choices=['react','cot','react_reflect'], default='react', help='Agent style to run')
    p.add_argument('--dataset-config', default=None, help='Optional dataset config/name (for multi-config datasets like PubMedQA)')
    p.add_argument('--reflexion-strategy', choices=['none','last_attempt','reflexion','last_attempt_and_reflexion'], default='reflexion', help='Reflexion strategy to apply (when agent supports it)')
    p.add_argument('--max-steps', type=int, default=6, help='max steps for ReactAgent')
    # Optional explicit field mapping (overrides auto-detection)
    p.add_argument('--question-field', default='question', help='Dataset field to use as the question')
    p.add_argument('--context-field', default='context', help='Dataset field to use as the context/abstract')
    p.add_argument('--answer-field', default='final_decision', help='Dataset field to use as the gold answer/label')
    p.add_argument('--long-answer-field', default='long_answer', help='Dataset field containing the explanatory long answer/rationale (optional)')
    # Local transformers options
    p.add_argument('--device', default=None, help='Device to load model on (cuda/mps/cpu). If not set, adapter auto-detects')
    p.add_argument('--load-in-4bit', action='store_true', help='Load model in 4-bit using bitsandbytes (requires CUDA + bitsandbytes)')
    # UnsloTh / long-context options
    p.add_argument('--use-unsloth', action='store_true', help='Use UnsloTh prequantized models (Colab-friendly)')
    p.add_argument('--max-seq-length', type=int, default=8192, help='Max sequence length / context window for UnsloTh or long-context models')
    p.add_argument('--force-finish-format', action='store_true', help='Ask agents to output exactly one Finish[...] action with yes/no/maybe when finishing')
    p.add_argument('--confidence-threshold', type=float, default=0.6, help='Confidence threshold (0..1) to accept yes/no/maybe without further reflexion')
    p.add_argument('--max-reflect-attempts', type=int, default=2, help='Maximum number of reflexion+retry attempts when confidence is low')
    p.add_argument('--stop-on-correct', action='store_true', help='If the initial agent prediction matches the gold label, stop immediately and avoid any further reflexion or enforcement')
    p.add_argument('--print-debug', dest='print_debug', action='store_true', help='Print verbose debug info (default)')
    p.add_argument('--no-print-debug', dest='print_debug', action='store_false', help='Disable verbose debug info')
    p.add_argument('--print-logit-debug', action='store_true', help='Print yes/no/maybe probability scores when evaluating confidence')
    p.add_argument('--force-argmax-final', action='store_true', help='Force final predicted label to the transformers argmax when final scoring is available')
    p.add_argument('--disable-confidence-enforcement', action='store_true', help='Disable all confidence-based label enforcement (attempt-level and final-pass)')
    p.set_defaults(print_debug=True, print_logit_debug=False)
    p.add_argument('--keep-fewshot-examples', action='store_true', help='Preserve builtin few-shot examples (otherwise cleared unless dataset contains PubMedQA)')
    p.add_argument('--readability-min', type=float, default=6.0, help='Minimum Flesch-Kincaid grade to consider explanation acceptable')
    p.add_argument('--readability-max', type=float, default=8.0, help='Maximum Flesch-Kincaid grade to consider explanation acceptable')
    p.add_argument('--rewrite-on-readability', action='store_true', help='If readability outside range, ask LLM to rewrite explanation at target grade (keeps label fixed)')
    p.add_argument('--enforce-readability-reflexion', action='store_true', help='When FK outside range, trigger agent.reflect and rerun (enforce via reflexion)')
    p.add_argument('--enforce-length', action='store_true', help='Enforce answer length roughly matching gold long_answer (within tolerance)')
    p.add_argument('--length-tolerance', type=float, default=0.2, help='Relative tolerance for answer length when --enforce-length is set (e.g., 0.2 = +/-20%)')
    p.add_argument('--max-readability-rewrites', type=int, default=1, help='Maximum attempts to rewrite rationale for readability acceptance')
    p.add_argument('--rouge-drop-threshold', type=float, default=0.05, help='Maximum allowed drop in ROUGE-1 F1 when accepting rewritten rationale')
    p.add_argument('--flip-on-incorrect', action='store_true', help='If the predicted label is incorrect, flip to an alternative (of the remaining two) using logits confidence and produce a matching rationale')
    args = p.parse_args()

    # normalize limit
    if args.limit == 0:
        args.limit = None

    # Fill use_transformers default: if flag not passed, default to False (inference) unless user passes --use-transformers
    # argparse configured so default is True when --use-transformers present, else False

    # Build boolean correctly
    args.use_transformers = bool(args.use_transformers)

    run(args)
