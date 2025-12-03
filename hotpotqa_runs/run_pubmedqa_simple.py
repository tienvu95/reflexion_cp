"""Minimal PubMedQA runner (no agents, no logits, optional reflexion).

Behavior per example:
- Query the local LLM (UnsloTh adapter) with the question + context.
- LLM must reply with exactly two lines:
  1) A single word: `yes` / `no` / `maybe` (case-insensitive)
  2) A line starting with `Reason:` followed by the justification text.
- If `--reflexion` is provided, the script will run one simple reflection pass:
  feed the previous answer back to the model and ask it to revise once.
- The prompt asks the model to make the `Reason:` about the dataset `long_answer`
  length (words) ±10% (best-effort).

This file is intentionally tiny and uses only `datasets`, `unsloth` adapter,
and optional metric libraries (`sklearn`, `rouge_score`).
"""

from typing import Optional
import os
import sys

# keep imports lazy and robust against different invocation CWDs
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def build_llm(model_name: Optional[str] = None):
    try:
        from hotpotqa_runs.unsloth_llm import UnslothLLM
    except Exception:
        try:
            from unsloth_llm import UnslothLLM
        except Exception:
            UnslothLLM = None
    if UnslothLLM is None:
        raise RuntimeError('UnsloTh adapter not found. Add hotpotqa_runs/unsloth_llm.py or install adapter')
    model = model_name or os.environ.get('HF_LOCAL_MODEL', '')
    return UnslothLLM(model_name=model)


def extract_label_and_reason(text: str):
    if not text:
        return 'maybe', 'Reason: '
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return 'maybe', 'Reason: '
    first = lines[0].strip().lower()
    # normalize first token to yes/no/maybe
    if 'yes' in first.split():
        label = 'yes'
    elif 'no' in first.split():
        label = 'no'
    elif 'maybe' in first.split() or 'uncertain' in first:
        label = 'maybe'
    else:
        # if first line isn't a clear word, try first token
        tok = first.split()[0] if first.split() else 'maybe'
        label = {'yes':'yes','no':'no'}.get(tok, 'maybe')

    # find a line that begins with 'reason:' or use second line
    reason = ''
    for ln in lines[1:]:
        if ln.lower().startswith('reason:'):
            reason = ln
            break
    if not reason and len(lines) >= 2:
        reason = 'Reason: ' + lines[1]
    if not reason:
        reason = 'Reason: '
    return label, reason


def make_prompt(question: str, context: str, target_words: int):
    return (
        "You are a helpful medical assistant. Read the question and the context, "
        "and then answer with exactly two lines.\n\n"
        "Line 1: a single word answer: 'yes' or 'no' or 'maybe' (nothing else).\n"
        "Line 2: start with 'Reason:' then a short justification that is about "
        f"{target_words} words (plus or minus 10%). Keep it factual and concise.\n\n"
        "Now the question and context:\n\nQuestion: " + question + "\n\nContext: " + (context or '') + "\n\nOutput:\n"
    )


def main():
    import argparse
    import csv

    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None, help='UnsloTh model id or set HF_LOCAL_MODEL')
    p.add_argument('--dataset', default='qiaojin/PubMedQA')
    p.add_argument('--dataset-config', default='pqa_labeled')
    p.add_argument('--split', default='validation')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--reflexion', action='store_true', help='Do one reflect/revise pass')
    p.add_argument('--out', default='pubmedqa_simple_results.csv')
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except Exception:
        raise RuntimeError('Please install the `datasets` package')

    ds = load_dataset(args.dataset, args.dataset_config)
    if args.split not in ds:
        raise RuntimeError(f"Split {args.split} not in dataset; available: {list(ds.keys())}")
    part = ds[args.split]
    N = len(part) if args.limit <= 0 else min(len(part), args.limit)

    # Use run() so callers can pass an external LLM instance
    run(args, external_llm=None)


def run(args, external_llm=None):
    """Run the simple PubMedQA pipeline.

    Args:
        args: object with fields similar to the original runner (dataset, split, limit, model, reflexion, out)
        external_llm: if provided, a callable LLM instance (llm(prompt) -> text). If None, the local adapter is built.

    Returns:
        results: list of dicts with keys index, question, true, pred, reason, gold_long
    """
    # tolerate argparse.Namespace or SimpleNamespace style args
    dataset = getattr(args, 'dataset', 'qiaojin/PubMedQA')
    dataset_config = getattr(args, 'dataset_config', getattr(args, 'dataset_config', 'pqa_labeled'))
    split = getattr(args, 'split', 'validation')
    limit = int(getattr(args, 'limit', 0))
    reflexion = bool(getattr(args, 'reflexion', False))
    out_fn = getattr(args, 'out', 'pubmedqa_simple_results.csv')
    model_arg = getattr(args, 'model', None)

    try:
        from datasets import load_dataset
    except Exception:
        raise RuntimeError('Please install the `datasets` package')

    ds = load_dataset(dataset, dataset_config)
    if split not in ds:
        raise RuntimeError(f"Split {split} not in dataset; available: {list(ds.keys())}")
    part = ds[split]
    N = len(part) if limit <= 0 else min(len(part), limit)

    llm = external_llm if external_llm is not None else build_llm(model_arg)

    results = []

    for i in range(N):
        ex = part[i]
        q = ex.get('question') or ex.get('Question') or ''
        ctx = ex.get('context') or ex.get('abstract') or ''
        gold = (ex.get('final_decision') or ex.get('answer') or '').strip().lower()
        gold_long = (ex.get('long_answer') or '').strip()
        gold_words = len(gold_long.split()) if gold_long else 40
        target = max(5, int(round(gold_words)))

        prompt = make_prompt(q, ctx, target)
        try:
            out = llm(prompt)
        except Exception as e:
            out = ''
            print(f'LLM call failed for idx {i}:', e)

        label, reason = extract_label_and_reason(out)

        if reflexion:
            reflect_prompt = (
                "You previously answered the question. Here is your previous output:\n\n"
                f"Answer: {label}\n{reason}\n\nPlease reflect briefly and, if you think a change is needed, provide a revised answer. "
                "Output again exactly two lines: Line1 single word yes/no/maybe, Line2 starting with 'Reason:' with a justification about "
                f"{target} words (±10%). Otherwise, repeat the same two-line form.\n\nQuestion: " + q + "\nContext: " + (ctx or '') + "\n\nOutput:\n"
            )
            try:
                out2 = llm(reflect_prompt)
                label2, reason2 = extract_label_and_reason(out2)
                label, reason = label2, reason2
            except Exception:
                pass

        results.append({'index': i, 'question': q, 'true': gold, 'pred': label, 'reason': reason, 'gold_long': gold_long})

        if (i+1) % 20 == 0 or (i+1) == N:
            print(f'Progress {i+1}/{N}')

    # write CSV
    keys = ['index', 'question', 'true', 'pred', 'reason']
    with open(out_fn, 'w', newline='', encoding='utf-8') as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, '') for k in keys})

    # basic metrics
    try:
        from sklearn.metrics import accuracy_score, f1_score
        golds = [r['true'] for r in results]
        preds = [r['pred'] for r in results]
        print('Accuracy:', accuracy_score(golds, preds))
        print('Macro-F1:', f1_score(golds, preds, average='macro'))
    except Exception:
        print('sklearn not available; skipping accuracy/f1')

    # rouge on reasons vs gold_long
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
        r1s = []
        rLs = []
        for r in results:
            ref = (r['gold_long'] or '').strip()
            hyp = r['reason'].split(':',1)[1] if ':' in r['reason'] else r['reason']
            if not ref:
                continue
            sc = scorer.score(ref, hyp)
            r1s.append(sc['rouge1'].fmeasure)
            rLs.append(sc['rougeL'].fmeasure)
        if r1s:
            print('Avg ROUGE-1 F1:', sum(r1s)/len(r1s))
            print('Avg ROUGE-L F1:', sum(rLs)/len(rLs))
    except Exception:
        print('rouge_score not available; skipping ROUGE')

    print('Results saved to', out_fn)
    return results


