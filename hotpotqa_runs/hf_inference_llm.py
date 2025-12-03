"""Simple Hugging Face Inference API adapter for text generation.

Usage:
  export HF_API_TOKEN=hf_xxx
  export HF_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
  from hf_inference_llm import HFInferenceLLM
  llm = HFInferenceLLM(model_id=os.environ['HF_MODEL_ID'])
  text = llm(prompt)
"""
import os
import json
from typing import Optional
try:
    import requests
except Exception:
    requests = None

class HFInferenceLLM:
    def __init__(self,
                 model_id: str,
                 api_token: Optional[str] = None,
                 temperature: float = 0.0,
                 max_new_tokens: int = 256,
                 ) -> None:
        self.model_id = model_id
        self.api_token = api_token or os.environ.get('HF_API_TOKEN')
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        if requests is None:
            raise RuntimeError('`requests` library is required for HFInferenceLLM. Install with `pip install requests`.')

        if not self.api_token:
            raise RuntimeError('Hugging Face API token not found. Set `HF_API_TOKEN` env var or pass `api_token`.')

        # New Hugging Face routing endpoint is preferred; older api-inference host
        # may return 410 instructing to use router.huggingface.co.
        self.url = f'https://router.huggingface.co/models/{self.model_id}'
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }

    def __call__(self, prompt: str) -> str:
        payload = {
            'inputs': prompt,
            'parameters': {
                'temperature': float(self.temperature),
                'max_new_tokens': int(self.max_new_tokens),
            },
            'options': {
                'wait_for_model': True,
            }
        }

        # Try the primary router url; if server indicates the older host is required
        # or vice-versa, surface the error. We also retry once against the
        # legacy api-inference host if we get an instructive 410 response.
        resp = requests.post(self.url, headers=self.headers, data=json.dumps(payload), timeout=120)
        # If router indicates the model isn't available (410) or not found (404),
        # try the legacy api-inference endpoint as a fallback.
        if resp.status_code in (410, 404):
            legacy = f'https://api-inference.huggingface.co/models/{self.model_id}'
            resp = requests.post(legacy, headers=self.headers, data=json.dumps(payload), timeout=120)

        if resp.status_code != 200:
            # try to surface helpful error: include status, content-type and a
            # snippet of the response body so callers can debug 404/403/HTML errors.
            content_type = resp.headers.get('content-type', '')
            body = ''
            try:
                # Try to parse JSON first for structured error info
                body_json = resp.json()
                body = json.dumps(body_json)
            except Exception:
                try:
                    body = resp.text
                except Exception:
                    body = '<unreadable response body>'

            snippet = body[:2000]
            raise RuntimeError(f"HF inference API error: {resp.status_code} - content-type={content_type} - body_snippet={snippet}")

        data = resp.json()
        # Model outputs vary. Common shapes:
        # - list of dicts with 'generated_text'
        # - dict with 'error'
        # - string
        if isinstance(data, list):
            # join generated_text fields
            parts = []
            for item in data:
                if isinstance(item, dict) and 'generated_text' in item:
                    parts.append(item['generated_text'])
                elif isinstance(item, dict) and 'text' in item:
                    parts.append(item['text'])
                else:
                    parts.append(str(item))
            return ' '.join(parts).strip()

        if isinstance(data, dict):
            if 'generated_text' in data:
                return data['generated_text'].strip()
            if 'error' in data:
                raise RuntimeError(f"HF inference API returned error: {data['error']}")
            # fallback: return json dump
            return json.dumps(data)

        # otherwise fallback to text
        return str(data).strip()

    def predict_label_probs(self, prompt: str, labels=None) -> dict:
        """Ask the hosted model to provide probabilities for discrete labels.

        This is a heuristic: the inference API does not expose logits, so we
        prompt the model to return probabilities in JSON. The output is parsed
        for a JSON object or float pairs. Returns a dict mapping label->prob.
        """
        if labels is None:
            labels = ['yes', 'no', 'maybe']

        # Build a small instruction prompt asking for JSON probabilities
        instruct = (
            "For the following input, return a JSON object with probabilities for keys 'yes','no','maybe'."
            " Example: {\"yes\":0.7, \"no\":0.2, \"maybe\":0.1}. Do not include extra text.\n\n"
        )
        full = instruct + prompt
        text = self.__call__(full)

        # Try to parse JSON from the response
        try:
            # find first JSON object in text
            import re, json
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                obj = json.loads(m.group(0))
                # normalize keys and values
                out = {}
                for k, v in obj.items():
                    kk = str(k).strip().lower()
                    try:
                        out[kk] = float(v)
                    except Exception:
                        try:
                            out[kk] = float(str(v))
                        except Exception:
                            out[kk] = 0.0
                # ensure all requested labels present
                for lab in labels:
                    if lab not in out:
                        out[lab] = 0.0
                # normalize to sum 1 if possible
                s = sum(out.values())
                if s > 0:
                    for k in out:
                        out[k] = out[k] / s
                return out
        except Exception:
            pass

        # Fallback: extract floats in order if JSON not found
        try:
            import re
            nums = re.findall(r"([0-9]*\.?[0-9]+)", text)
            probs = [float(n) for n in nums]
            if len(probs) >= len(labels):
                out = {lab: float(probs[i]) for i, lab in enumerate(labels)}
                s = sum(out.values())
                if s > 0:
                    for k in out:
                        out[k] = out[k] / s
                return out
        except Exception:
            pass

        # Last resort: soft uniform fallback
        return {lab: 1.0/len(labels) for lab in labels}
