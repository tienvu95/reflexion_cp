"""Evaluate PubMedQA with HF Transformers / Inference LLM and optionally refine explanations.

This script does not fine-tune the model. It runs the model (local HF or HF Inference),
computes Flesch-Kincaid grade and SMOG index for explanations, computes Brier score for
the yes/no/maybe prediction probabilities, and optionally asks the LLM to rewrite
explanations to meet a target readability range.

Usage:
  export HF_API_TOKEN=hf_xxx   # if using HF inference
  export HF_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
  python -m hotpotqa_runs.eval_refine_hf

Note: this script expects `hotpotqa_runs.hf_transformers_llm.HFTransformersLLM` or
`hotpotqa_runs.hf_inference_llm.HFInferenceLLM` to be available.
"""
import os
import math
import traceback
from typing import List

from datasets import load_dataset

import textstat
from statistics import mean

from hotpotqa_runs.hf_transformers_llm import HFTransformersLLM
from hotpotqa_runs.hf_inference_llm import HFInferenceLLM


def build_llm_from_env():
    model_id = os.environ.get('HF_MODEL_ID', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
    hf_token = os.environ.get('HF_API_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')

    # Prefer local HFTransformersLLM if CUDA available and adapter exists
    try:
        import torch
        if torch.cuda.is_available():
            try:
                return HFTransformersLLM(model_id=model_id, load_in_4bit=False, device='cuda')
            except Exception:
                traceback.print_exc()
                print('Local HFTransformersLLM failed; falling back to HF Inference if token present')
    except Exception:
        pass

    # Fallback to HF Inference API
    if hf_token:
        return HFInferenceLLM(model_id=model_id, api_token=hf_token, temperature=0.0, max_new_tokens=256)

    raise RuntimeError('No LLM available: provide HF_API_TOKEN for hosted inference or run on a CUDA machine for local load')


def make_prompt(question: str, context: str) -> str:
    system = (
        "You are a careful medical assistant. First output a single word: 'yes' or 'no' or 'maybe' on the first line. "
        "On the second line, start with 'Reason:' and give a concise justification."
    )
    return system + "\n\nQuestion: " + question + "\n\nAbstract:\n" + (context or "") + "\n\n"


def extract_label_and_reason(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    label = None
    reason = ''
    if lines:
        # first non-empty token that matches yes/no/maybe
        import re
        m = re.search(r'\b(yes|no|maybe)\b', lines[0].lower())
        if m:
            label = m.group(1)
            reason = '\n'.join(lines[1:]).strip()
        else:
            # try scanning later lines
            for ln in lines:
                m = re.search(r'\b(yes|no|maybe)\b', ln.lower())
                if m:
                    label = m.group(1)
                    break
            reason = '\n'.join(lines[1:]).strip() if len(lines) > 1 else '\n'.join(lines).strip()
    return label, reason


def run_eval(limit=100, readability_target=(6.0, 8.0), rewrite_on_readability=True):
    print('Building LLM...')
    llm = build_llm_from_env()
    print('LLM ready:', type(llm))

    ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled')
    split = ds['train']
    N = min(len(split), limit)

    brier_sum = 0.0
    brier_n = 0

    fk_list = []
    smog_list = []

    for i in range(N):
        ex = split[i]
        q = ex.get('question', '')
        ctx = ex.get('context', '')
        gold = (ex.get('final_decision') or '').strip().lower()
        prompt = make_prompt(q, ctx)

        # get label probs if available
        probs = None
        try:
            if hasattr(llm, 'predict_label_probs'):
                probs = llm.predict_label_probs(prompt, labels=['yes','no','maybe'])
            else:
                # fallback: ask the model to output probabilities explicitly (slower)
                pass
        except Exception:
            traceback.print_exc()

        # get full generation (label + reason)
        try:
            out = llm(prompt)
        except Exception:
            traceback.print_exc()
            out = ''

        pred_label, reason = extract_label_and_reason(out)

        # compute readabilities
        fk = textstat.flesch_kincaid_grade(reason) if reason else 0.0
        smog = textstat.smog_index(reason) if reason else 0.0
        fk_list.append(fk)
        smog_list.append(smog)

        # compute brier: need probability for gold label
        if probs and gold in {'yes','no','maybe'}:
            p_true = float(probs.get(gold, 0.0))
            y = 1.0 if pred_label == gold else 0.0
            # Brier for binary is (p - y)^2; here we use one-vs-all for true class
            brier_sum += (p_true - y) ** 2
            brier_n += 1

        # readability-guided rewrite
        if rewrite_on_readability and reason:
            low, high = readability_target
            if fk < low or fk > high:
                # ask model to rewrite explanation at target grade
                rewrite_instr = f"Rewrite the explanation in simple layperson language at about {int(low)}th-{int(high)}th grade level. Keep the short yes/no/maybe label unchanged."
                new_prompt = prompt + "\n" + rewrite_instr + "\n\nPrevious explanation:\n" + reason
                try:
                    new_out = llm(new_prompt)
                    new_label, new_reason = extract_label_and_reason(new_out)
                    if new_reason:
                        reason = new_reason
                        # update readabilities
                        fk = textstat.flesch_kincaid_grade(reason)
                        smog = textstat.smog_index(reason)
                except Exception:
                    traceback.print_exc()

        # print progress for first few
        if i < 5:
            print(f"[{i}] Q: {q[:120]}")
            print('  gold:', gold, 'pred:', pred_label)
            print(f'  FK={fk:.2f} SMOG={smog:.2f}')
            print('  reason (first 200ch):', (reason[:200] + '...') if len(reason) > 200 else reason)

    avg_fk = mean(fk_list) if fk_list else 0.0
    avg_smog = mean(smog_list) if smog_list else 0.0
    brier = (brier_sum / brier_n) if brier_n > 0 else None

    print('\nSummary:')
    print(f'  Examples processed: {N}')
    print(f'  Avg Flesch-Kincaid grade: {avg_fk:.2f}')
    print(f'  Avg SMOG index: {avg_smog:.2f}')
    if brier is not None:
        print(f'  Brier score (one-vs-all on true class): {brier:.6f}  (N={brier_n})')
    else:
        print('  Brier score: not computed (no probability outputs from LLM)')


if __name__ == '__main__':
    # quick CLI via env vars
    LIM = int(os.environ.get('EVAL_LIMIT', '50'))
    run_eval(limit=LIM, readability_target=(6.0, 8.0), rewrite_on_readability=True)
