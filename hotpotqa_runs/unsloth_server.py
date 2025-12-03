"""Simple FastAPI server that loads an UnsloTh model once and serves /generate.

Usage:
  pip install fastapi uvicorn
  python -m uvicorn hotpotqa_runs.unsloth_server:app --host 127.0.0.1 --port 8000 --workers 1

Then POST JSON {"prompt": "..."} to http://127.0.0.1:8000/generate

This keeps the model loaded in memory across requests so you don't reload for each run.
"""
from typing import Optional
import os
from pydantic import BaseModel

try:
    from fastapi import FastAPI, HTTPException
except Exception:
    raise RuntimeError("Install fastapi: pip install fastapi uvicorn")

try:
    from hotpotqa_runs.unsloth_llm import UnslothLLM
except Exception:
    # support running from repo root
    from unsloth_llm import UnslothLLM


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 256


app = FastAPI(title="Unsloth LLM server")

# Configure these via env or edit before starting
MODEL_NAME = os.environ.get('UNSLOTH_MODEL', 'unsloth/Meta-Llama-3.1-8B-bnb-4bit')
HF_TOKEN = os.environ.get('HF_API_TOKEN') or os.environ.get('HUGGINGFACE_API_TOKEN')
MAX_SEQ = int(os.environ.get('UNSLOTH_MAX_SEQ', '8192'))
LOAD_4BIT = os.environ.get('UNSLOTH_4BIT', '1') in ('1', 'true', 'True')

# Load model once on import
print(f"Loading UnsloTh model {MODEL_NAME} (this may take a while)...")
llm = UnslothLLM(model_name=MODEL_NAME, token=HF_TOKEN, load_in_4bit=LOAD_4BIT, max_seq_length=MAX_SEQ)
print("Model loaded.")


@app.post('/generate')
def generate(req: GenerateRequest):
    try:
        # Some adapters ignore max_new_tokens; pass via prompt if needed
        out = llm(req.prompt)
        return {"text": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
