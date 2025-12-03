"""Lightweight runner: UnsloTh + CoTAgent + Reflexion

This file mirrors the simple notebook flow (no heavy scoring or
sanitization). It intentionally keeps imports lazy and minimal so
the module can be imported quickly in the editor.

Usage (example):
  python run_pubmedqa_lite.py --model <unsloth-model-id> --num-trials 3

Set `HF_LOCAL_MODEL` env var or pass `--model` to choose a local UnsloTh model.
"""

import os
import sys

# Keep import-time overhead tiny for editor / linting: append repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def build_unsloth_or_die(model_name=None, load_in_4bit=True, max_seq_length=8192):
    """Instantiate UnsloTh adapter present in this repo or raise helpful error."""
    try:
        from hotpotqa_runs.unsloth_llm import UnslothLLM
    except Exception:
        try:
            from unsloth_llm import UnslothLLM
        except Exception:
            UnslothLLM = None
    if UnslothLLM is None:
        raise RuntimeError("UnsloTh adapter not found. Ensure `unsloth_llm.py` is available in the repo.")
    model = model_name or os.environ.get('HF_LOCAL_MODEL') or ''
    return UnslothLLM(model_name=model, load_in_4bit=load_in_4bit, max_seq_length=max_seq_length)


def main():
    import argparse
    import joblib

    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None, help='Local UnsloTh model id (or set HF_LOCAL_MODEL env var)')
    p.add_argument('--num-trials', type=int, default=5)
    p.add_argument('--out-root', default='../root/CoT/unsloth', help='Where to save logs/agents')
    p.add_argument('--dataset', default='qiaojin/PubMedQA', help='HF dataset id to evaluate (default: qiaojin/PubMedQA)')
    p.add_argument('--dataset-config', default='pqa_labeled', help='Dataset config (PubMedQA requires pqa_labeled)')
    p.add_argument('--split', default='train', help='Which split to evaluate (train/validation/test)')
    p.add_argument('--mode', choices=['short','long','both'], default='both', help='Which evaluation to run: short=label/Brier, long=rationale ROUGE/readability')
    p.add_argument('--limit', type=int, default=0, help='Limit number of examples (0 means all)')
    args = p.parse_args()

    # Lazy import of project agent classes / prompts so this module imports fast
    from agents import CoTAgent, ReflexionStrategy
    # prefer project-local prompts/fewshots
    try:
        from fewshots import COTQA_SIMPLE6 as COT, COT_SIMPLE_REFLECTION as COT_REFLECT
    except Exception:
        from hotpotqa_runs.fewshots import COTQA_SIMPLE6 as COT, COT_SIMPLE_REFLECTION as COT_REFLECT
    try:
        from prompts import cot_simple_reflect_agent_prompt as AGENT_PROMPT, cot_simple_reflect_prompt as REFLECT_PROMPT
    except Exception:
        from hotpotqa_runs.prompts import cot_simple_reflect_agent_prompt as AGENT_PROMPT, cot_simple_reflect_prompt as REFLECT_PROMPT

    # If user asked for the simple CoT notebook-like flow, run that quickly
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    if args.mode in ('short','long','both'):
        # Load PubMedQA from HF datasets
        try:
            from datasets import load_dataset
        except Exception:
            raise RuntimeError('Please install `datasets` to run PubMedQA evaluation (pip install datasets)')
        ds = load_dataset(args.dataset, args.dataset_config)
        split_name = args.split
        if split_name not in ds:
            # some datasets provide 'train'/'validation' etc.
            available = list(ds.keys())
            raise RuntimeError(f"Split '{split_name}' not in dataset. Available splits: {available}")
        pmqa = ds[split_name]
        N = len(pmqa) if args.limit == 0 else min(len(pmqa), args.limit)

        # Build UnsloTh LLM (may be heavy at instantiation time)
        llm = build_unsloth_or_die(model_name=args.model)
        model = getattr(llm, 'model', None)
        tokenizer = getattr(llm, 'tokenizer', None)
        if model is None or tokenizer is None:
            raise RuntimeError('UnsloTh adapter must expose .model and .tokenizer for evaluation flows')

        import torch
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Helpers for short / label-only evaluation (single-token)
        CLASSES = ['yes','no','maybe']
        CLASS_SET = set(CLASSES)

        def build_label_messages(q, ctx):
            if isinstance(ctx, list): ctx = ' '.join(ctx)
            return [{
                'from': 'human',
                'value': (
                    f"Answer ONLY with one word: yes, no, or maybe.\n\nQuestion: {q}\n\nAbstract:\n{ctx}\n\n"
                    "Answer strictly with one word.\n\nLabel: "
                )
            }]

        def parse_label_token(tok_text: str) -> str:
            lab = tok_text.strip().strip(',.:;!?').lower()
            return lab if lab in CLASS_SET else 'maybe'

        def _first_token_ids(strings):
            out = []
            for s in strings:
                ids = tokenizer(s, add_special_tokens=False).input_ids
                if ids:
                    out.append(ids[0])
            return out

        CAND_IDS = {
            'yes': _first_token_ids([' yes','Yes','yes']),
            'no': _first_token_ids([' no','No','no']),
            'maybe': _first_token_ids([' maybe','Maybe','maybe']),
        }

        import math

        def probs_from_first_step_logits(out_struct):
            # out_struct is the return from model.generate with scores
            logits = out_struct.scores[0][0]
            pv = torch.softmax(logits, dim=-1)
            mass = {
                lab: float(pv[torch.tensor(ids, device=pv.device)].sum().item()) if ids else 0.0
                for lab, ids in CAND_IDS.items()
            }
            Z = sum(mass.values()) + 1e-12
            return {k: v / Z for k, v in mass.items()}

        def brier_multiclass_sum(prob_dict, gold_label, classes=CLASSES):
            return sum((prob_dict[c] - (1.0 if c == gold_label else 0.0))**2 for c in classes)

        # Helpers for long / rationale evaluation
        ASSIST_RE = None
        import re
        ASSIST_RE = re.compile(r'^(?:<\\|assistant\\|>|<\\|start_header_id\\|>\\s*assistant\\s*<\\|end_header_id\\|>|assistant:?)', re.IGNORECASE)

        def strip_assistant_header(text: str) -> str:
            t = text.lstrip()
            t = ASSIST_RE.sub('', t)
            lines = [ln.strip() for ln in t.splitlines()]
            while lines and re.fullmatch(r'(?:assistant:?|<\\|assistant\\|>)', lines[0], re.IGNORECASE):
                lines.pop(0)
            return '\\n'.join(lines).strip()

        def parse_reason(out_text: str) -> str:
            t = strip_assistant_header(out_text)
            m = re.search(r'(?mi)^Reason:\\s*(.*)$', t, flags=re.DOTALL)
            if m:
                return m.group(1).strip()
            cut = t.lower().rfind('reason:')
            return t[cut+len('reason:'):].strip() if cut != -1 else t

        # Metrics accumulators
        preds, golds = [], []
        brier_probs, brier_vals = [], []
        refs_long, hyps_long = [], []

        # Prebuild label prompts via tokenizer.apply_chat_template when available
        def label_prompt_text(row):
            msgs = build_label_messages(row['question'], row['context'])
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                # fallback to plain concatenated text
                return msgs[0]['value']

        def long_prompt_text(row):
            INSTR = (
                "You are answering PubMedQA. Write a concise explanation based only on the abstract."
                "\\n\\nReturn answers in this EXACT format:\\nReason:\\n<your explanation>"
            )
            ctx = ''.join(row['context']) if isinstance(row['context'], list) else (row['context'] or '')
            msgs = [{ 'from': 'human', 'value': f"{INSTR}\\n\\nQuestion: {row['question']}\\n\\nAbstract:\\n{ctx}\\n\\nReason:\\n" }]
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                return msgs[0]['value']

        # iterate
        for i in range(N):
            ex = pmqa[i]
            gold = (ex.get('final_decision') or ex.get('answer') or '').strip().lower()
            if args.mode in ('short','both'):
                prompt = label_prompt_text(ex)
                enc = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True, max_length=768).to(DEVICE)
                with torch.inference_mode():
                    out = model.generate(
                        **enc,
                        max_new_tokens=1,
                        do_sample=False, temperature=0.0,
                        use_cache=False,
                        pad_token_id=tokenizer.eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                new_tok = out.sequences[0, enc.input_ids.shape[1]:]
                label_text = tokenizer.decode(new_tok, skip_special_tokens=True)
                pred = parse_label_token(label_text)
                preds.append(pred)
                golds.append(gold)
                probs = probs_from_first_step_logits(out)
                brier_probs.append(probs)
                brier_vals.append(brier_multiclass_sum(probs, gold))

            if args.mode in ('long','both'):
                refs_long.append((ex.get('long_answer') or '').strip())
                p = long_prompt_text(ex)
                enc = tokenizer(p, return_tensors='pt', padding=False, truncation=True, max_length=1024).to(DEVICE)
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=160, do_sample=False, temperature=0.0, use_cache=True, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True)
                new_tokens = out.sequences[0, enc.input_ids.shape[1]:]
                gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                rationale = parse_reason(gen_text)
                hyps_long.append(rationale)

        # Compute and print metrics
        if args.mode in ('short','both'):
            try:
                from sklearn.metrics import accuracy_score, f1_score, classification_report
                print('\nShort-label metrics:')
                print(f"Accuracy:  {accuracy_score(golds, preds):.4f}")
                print(f"Macro-F1:  {f1_score(golds, preds, average='macro'):.4f}\\n")
                print(classification_report(golds, preds, digits=4))
            except Exception:
                print('sklearn not installed; skipping classification metrics')
            if brier_vals:
                print(f"\\nMean Brier (sum, 0–2): {sum(brier_vals)/len(brier_vals):.6f}")

        if args.mode in ('long','both'):
            try:
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
                r1s = [scorer.score(ref, hyp)['rouge1'].fmeasure for ref, hyp in zip(refs_long, hyps_long)]
                rLs = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(refs_long, hyps_long)]
                print(f"\\nAvg ROUGE-1 F1 (rationales vs gold long_answer): {sum(r1s)/len(r1s):.6f}")
                print(f"Avg ROUGE-L F1 (rationales vs gold long_answer): {sum(rLs)/len(rLs):.6f}")
            except Exception:
                print('rouge_score not available; skipping ROUGE metrics')
            try:
                import textstat
                fre = [textstat.flesch_reading_ease(h) for h in hyps_long]
                print(f"Avg Flesch Reading Ease (rationales): {sum(fre)/len(fre):.2f}")
            except Exception:
                print('textstat not available; skipping readability metrics')

        # save a small CSV of results
        import csv
        out_csv = os.path.join(out_root, f'pubmedqa_lite_results_{args.mode}_{args.split}.csv')
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            fields = ['index','question','context','true_answer','predicted_answer','rationale']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i in range(N):
                q = pmqa[i]['question']
                ctx = pmqa[i]['context']
                true = (pmqa[i].get('final_decision') or pmqa[i].get('answer') or '')
                pred = preds[i] if i < len(preds) else ''
                rat = hyps_long[i] if i < len(hyps_long) else ''
                writer.writerow({'index': i, 'question': q, 'context': str(ctx)[:1000], 'true_answer': true, 'predicted_answer': pred, 'rationale': rat})

        print('Saved results to', out_csv)
        return


if __name__ == '__main__':
    main()
