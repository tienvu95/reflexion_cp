"""Local Hugging Face `transformers` adapter for generation.

This adapter supports loading models via `transformers` and (optionally)
loading in 4-bit with `bitsandbytes` for large models.

Usage examples:
  # simple: let adapter load tokenizer+model by model_id
  llm = HFTransformersLLM(model_id="stanford-crfm/BioMedLM", device="cuda", load_in_4bit=True)
  text = llm("Summarize: ...")

  # advanced: pass preloaded tokenizer+model
  from transformers import AutoTokenizer, AutoModelForCausalLM
  tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
  llm = HFTransformersLLM(model=model, tokenizer=tok, device="cuda")
"""
from typing import Optional
import os

# Recognized seq2seq model type identifiers
SEQ2SEQ_TYPES = {'t5', 'mt5', 'mbart', 'marian', 'pegasus', 'flan-t5'}


class HFTransformersLLM:
    def __init__(self,
                 model_id: Optional[str] = None,
                 model: Optional[object] = None,
                 tokenizer: Optional[object] = None,
                 device: Optional[str] = None,
                 load_in_4bit: bool = False,
                 bnb_config: Optional[object] = None,
                 temperature: float = 0.0,
                 max_new_tokens: int = 256,
                 ) -> None:
        self.model_id = model_id
        self._model = model
        self._tokenizer = tokenizer
        self.device = device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_DEVICE_ORDER") else "cpu")
        self.load_in_4bit = load_in_4bit
        self.bnb_config = bnb_config
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # Lazy imports: don't require heavy deps until actually loading model
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            import torch
        except Exception:
            raise RuntimeError('`torch` is required for HFTransformersLLM. Install PyTorch appropriate for your platform.')

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from transformers import AutoConfig, AutoModelForSeq2SeqLM
        except Exception as e:
            raise RuntimeError("`transformers` required for HFTransformersLLM. Install with `pip install transformers accelerate`") from e

        # Optional bitsandbytes config for 4-bit loading
        bnb_cfg = None
        if self.load_in_4bit:
            try:
                # local import may fail if bitsandbytes not installed
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(**(self.bnb_config or {})) if self.bnb_config is None else self.bnb_config
            except Exception:
                # If bitsandbytes missing, surface an explicit error
                raise RuntimeError("`bitsandbytes`/4-bit loading requested but not available. Install `bitsandbytes` and set up proper CUDA environment or disable `load_in_4bit`.")

        # Load tokenizer if needed
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

        # Determine torch device and preferred dtype
        use_cuda = torch.cuda.is_available()
        use_mps = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
        if self.device is None:
            if use_cuda:
                self.device = 'cuda'
            elif use_mps:
                self.device = 'mps'
            else:
                self.device = 'cpu'

        # If running on CPU only, loading very large models (8B+) will attempt
        # to offload parameters to disk which is fragile and often fails with
        # confusing errors. Surface a clear error and guidance instead of
        # letting transformers/accelerate try disk offload automatically.
        if self.device == 'cpu' and not use_cuda:
            raise RuntimeError(
                "Local CPU-only loading of large HF models is not supported in this demo. "
                "Attempting to load a multi-gigabyte model on CPU will trigger disk offload and likely fail. "
                "Options:\n"
                "  1) Use a GPU (set device='cuda' and ensure CUDA + bitsandbytes are installed).\n"
                "  2) Use the Hugging Face Inference API: set HF_API_TOKEN/HUGGINGFACE_HUB_TOKEN and use the HFInferenceLLM adapter.\n"
                "  3) Use a much smaller model (e.g., 'gpt2' or other small HF models) for local CPU runs.\n"
                "If you really want disk offload, use the transformers/accelerate disk_offload utilities directly, but that's outside this helper."
            )

        # If 4-bit requested but CUDA not available, disable and warn
        if self.load_in_4bit and not use_cuda:
            # bitsandbytes quantization only supported on CUDA; unset load_in_4bit
            self.load_in_4bit = False

        # Preferred torch dtype
        if self.device == 'cuda':
            preferred_dtype = torch.float16
        else:
            # MPS and CPU use float32 for better compatibility
            preferred_dtype = torch.float32

        # Debug logging: inform which device and dtype will be used
        try:
            print(f"HFTransformersLLM: device={self.device}, preferred_dtype={preferred_dtype}, load_in_4bit={self.load_in_4bit}")
        except Exception:
            pass

        # Load model if needed
        if self._model is None:
            # Determine model config to choose causal vs seq2seq class
            config = AutoConfig.from_pretrained(self.model_id)
            model_type = getattr(config, 'model_type', None)

            # Use device_map auto so transformers/accelerate picks up available devices
            load_kwargs = {}
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig as _BNBC
                    load_kwargs.update({
                        'device_map': 'auto',
                        'quantization_config': bnb_cfg,
                    })
                except Exception:
                    load_kwargs.update({'device_map': 'auto', 'load_in_4bit': True})
            else:
                load_kwargs.update({'device_map': 'auto'})

            # For encoder-decoder models (e.g., t5/flan-t5) use AutoModelForSeq2SeqLM
            if model_type in SEQ2SEQ_TYPES or (self.model_id and 't5' in (self.model_id.lower())):
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, torch_dtype=preferred_dtype if preferred_dtype is not None else None, **load_kwargs)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=preferred_dtype if preferred_dtype is not None else None, **load_kwargs)

            try:
                print(f"HFTransformersLLM: loaded model {self.model_id} with config.model_type={model_type}")
            except Exception:
                pass

            # If device_map was not used (e.g., mps/cpu), move model to selected device
            try:
                if load_kwargs.get('device_map') in (None, 'none') or self.device in ('mps', 'cpu'):
                    # move model to device
                    dev = torch.device(self.device)
                    self._model.to(dev)
            except Exception:
                # ignore device move errors and hope transformers handled it
                pass

        self._model.eval()
        self._loaded = True

    def __call__(self, prompt: str) -> str:
        # Lazy-load heavy deps and model
        self._ensure_loaded()

        # perform generation
        try:
            import torch
        except Exception:
            raise RuntimeError('`torch` required for HFTransformersLLM. Install PyTorch appropriate for your platform.')

        tok = self._tokenizer
        model = self._model

        # Tokenize with truncation to avoid model indexing errors when prompt is too long
        try:
            max_len = getattr(tok, 'model_max_length', None)
            if max_len is None or max_len <= 0:
                max_len = getattr(self._model.config, 'n_positions', None) or getattr(self._model.config, 'max_position_embeddings', None) or 1024
            # leave a small margin
            max_len = max(32, int(max_len) - 16)
            inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=max_len)
        except Exception:
            # fallback to default tokenization without truncation
            inputs = tok(prompt, return_tensors='pt')

        # Move input tensors to the same device as the model's parameters (best effort)
        try:
            model_device = next(self._model.parameters()).device
        except StopIteration:
            model_device = None

        if model_device is not None:
            try:
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            except Exception:
                # fallback: try moving to MPS/CUDA/CPU depending on availability
                try:
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
                        inputs = {k: v.to('mps') for k, v in inputs.items()}
                    else:
                        inputs = {k: v.to('cpu') for k, v in inputs.items()}
                except Exception:
                    pass

        # note: generation params can be tuned. For seq2seq models, generation
        # expects encoder inputs; for causal models, the prompt is in input_ids.
        gen = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=(self.temperature>0.0), temperature=self.temperature)
        out = tok.decode(gen[0], skip_special_tokens=True)

        # For causal models the output often contains the prompt as prefix; try
        # to remove it when present. For seq2seq, the model returns only the
        # generated continuation, so we can return it directly.
        try:
            model_config = getattr(model, 'config', None)
            model_type = getattr(model_config, 'model_type', None)
        except Exception:
            model_type = None

        if model_type not in SEQ2SEQ_TYPES and out.startswith(prompt):
            return out[len(prompt):].strip()
        return out.strip()

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def predict_label_probs(self, prompt: str, labels=None) -> dict:
        """Return probability distribution over discrete labels (e.g., ['yes','no','maybe']).

        This method computes next-token logits for the given prompt and converts
        the logits for the token IDs of the provided labels into a normalized
        probability distribution. It assumes each label maps to a single token
        in the tokenizer (common for words like 'yes','no','maybe').

        Returns a dict mapping label -> probability (float).
        """
        self._ensure_loaded()
        if labels is None:
            labels = ["yes", "no", "maybe"]

        tok = self._tokenizer
        model = self._model

        # Prepare inputs similar to __call__ path
        try:
            max_len = getattr(tok, 'model_max_length', None) or 2048
            max_len = max(32, int(max_len) - 16)
            inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=max_len)
        except Exception:
            inputs = tok(prompt, return_tensors='pt')

        # move inputs to model device
        try:
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except Exception:
            pass

        # Run model in eval mode to get logits
        with __import__('torch').no_grad():
            out = model(**inputs, return_dict=True)
            logits = getattr(out, 'logits', None)
            if logits is None:
                raise RuntimeError('Model did not return logits for probability prediction.')

        # We care about the last token logits
        last_logits = logits[0, -1, :]

        # Map label -> token id (require single-token labels)
        label_token_ids = {}
        for lab in labels:
            toks = tok(lab, add_special_tokens=False, return_tensors='pt')
            ids = toks['input_ids'][0].tolist()
            if len(ids) != 1:
                # If label tokenizes to multiple tokens, we warn and use the first token
                label_token_ids[lab] = ids[0]
            else:
                label_token_ids[lab] = ids[0]

        import math
        import torch as _torch

        # Extract logits for the candidate token ids
        cand_ids = list(label_token_ids.values())
        cand_logits = _torch.stack([last_logits[_torch.tensor(i, device=last_logits.device)] for i in cand_ids])
        probs = _torch.softmax(cand_logits, dim=0).cpu().tolist()

        return {lab: float(p) for lab, p in zip(labels, probs)}
