# PubMedQA — single-token label + Brier from logits (Unsloth one-by-one)

import re, gc, torch
from datasets import load_dataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INP = 768      # lower -> faster prefill
SYSTEM  = "Answer ONLY with one word: yes, no, or maybe."
CLASSES = ["yes","no","maybe"]
CLASS_SET = set(CLASSES)

# ---------- prompt: make the first generated token be the label ----------
def build_messages(q, ctx):
    if isinstance(ctx, list): ctx = " ".join(ctx)
    return [{
        "from": "human",
        "value": (
            f"{SYSTEM}\n\nQuestion: {q}\n\nAbstract:\n{ctx}\n\n"
            "Answer strictly with one word.\n\n"
            "Label: "   # <- next token will be yes|no|maybe
        )
    }]

def prompt_text(row):
    return tokenizer.apply_chat_template(
        build_messages(row["question"], row["context"]),
        tokenize=False, add_generation_prompt=True
    )

# ---------- tiny helpers ----------
def parse_label_token(tok_text: str) -> str:
    lab = tok_text.strip().strip(",.?:;!").lower()
    return lab if lab in CLASS_SET else "maybe"

def _first_token_ids(strings):
    out = []
    for s in strings:
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if ids: out.append(ids[0])
    return out

# include space/no-space + case variants (Llama tokenizers often use leading-space tokens)
CAND_IDS = {
    "yes":   _first_token_ids([" yes","Yes","yes"]),
    "no":    _first_token_ids([" no","No","no"]),
    "maybe": _first_token_ids([" maybe","Maybe","maybe"]),
}

def probs_from_first_step_logits(out_struct):
    logits = out_struct.scores[0][0]    # (vocab,)
    pv = torch.softmax(logits, dim=-1)
    mass = {
        lab: float(pv[torch.tensor(ids, device=pv.device)].sum().item()) if ids else 0.0
        for lab, ids in CAND_IDS.items()
    }
    Z = sum(mass.values()) + 1e-12
    return {k: v/Z for k, v in mass.items()}

def brier_multiclass_sum(prob_dict, gold_label, classes=CLASSES):
    # Sum version ranges [0, 2] for 3 classes (0 is perfect)
    return sum((prob_dict[c] - (1.0 if c == gold_label else 0.0))**2 for c in classes)

# ---------- data ----------
ds   = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
pmqa = ds["train"]       # use "test" for reporting; change to "train" if you want
N    = len(pmqa)        # set smaller for a smoke test

preds, golds = [], []
brier_probs, brier_vals = [], []

for i in tqdm(range(N), desc="Label-only + Brier (one-by-one)", ncols=100):
    row  = pmqa[i]
    gold = row["final_decision"].lower()
    golds.append(gold)

    prompt = prompt_text(row)
    enc = tokenizer(prompt, return_tensors="pt",
                    padding=False, truncation=True, max_length=MAX_INP).to(DEVICE)

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=1,            # exactly the label token
            do_sample=False, temperature=0.0,
            use_cache=False,             # lower KV mem
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,          # <- logits for first step
            return_dict_in_generate=True
        )

    # decode ONLY the new token
    new_tok = out.sequences[0, enc.input_ids.shape[1]:]
    label_text = tokenizer.decode(new_tok, skip_special_tokens=True)
    pred = parse_label_token(label_text)
    preds.append(pred)

    # probs -> Brier
    probs = probs_from_first_step_logits(out)
    brier_probs.append(probs)
    brier_vals.append(brier_multiclass_sum(probs, gold))

    del enc, out, new_tok
    if (i+1) % 100 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect()

# ---------- metrics ----------
print(f"\nAccuracy:  {accuracy_score(golds, preds):.4f}")
print(f"Macro-F1:  {f1_score(golds, preds, average='macro'):.4f}\n")
print(classification_report(golds, preds, digits=4))

print(f"\nMean Brier (sum, 0–2): {sum(brier_vals)/len(brier_vals):.6f}")

# peek a few
for j in range(min(5, N)):
    print(f"[{j:03d}] gold={golds[j]:<6} pred={preds[j]:<6}  probs={brier_probs[j]}")
