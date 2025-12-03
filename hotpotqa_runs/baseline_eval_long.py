# ==== PubMedQA rationale-only eval (ROUGE-1 + readability) ====
# Assumes:
#   - model, tokenizer already loaded (Unsloth)
#   - tokenizer has chat template set (e.g., get_chat_template(..., "llama-3.1"))
#   - FastLanguageModel.for_inference(model) already called



# ---- knobs (no abstract shrinking) ----
MAX_INP = 1024    # cap for (prompt + abstract) tokens
MAX_NEW = 160     # generation budget for rationale

# ---- Single-pass prompt: ONLY rationale required ----
INSTR = (
    "You are answering PubMedQA. "
    "Write a concise explanation in plain language based only on the abstract. "
    "End with: 'This is not medical advice.'\n\n"
    "Return answers in this EXACT format:\n"
    "Reason:\n"
    "<your explanation>"
)

def build_messages(q, ctx):
    if isinstance(ctx, list): ctx = " ".join(ctx)
    return [{
        "from": "human",
        "value": f"{INSTR}\n\nQuestion: {q}\n\nAbstract:\n{ctx}\n\nReason:\n"
    }]

def apply_tpl(msgs):
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# ---- helpers to clean and extract the rationale ----
ASSIST_RE = re.compile(
    r'^(?:<\|assistant\|>|<\|start_header_id\|>\s*assistant\s*<\|end_header_id\|>|assistant:?)[\s\r\n]*',
    re.IGNORECASE
)
def strip_assistant_header(text: str) -> str:
    text = text.lstrip()
    text = ASSIST_RE.sub("", text)
    lines = [ln.strip() for ln in text.splitlines()]
    while lines and re.fullmatch(r'(?:assistant:?|<\|assistant\|>)', lines[0], re.IGNORECASE):
        lines.pop(0)
    return "\n".join(lines).strip()

def parse_reason(out_text: str) -> str:
    t = strip_assistant_header(out_text)
    m = re.search(r'(?mi)^Reason:\s*(.*)$', t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: everything after the last "Reason:"
    cut = t.lower().rfind("reason:")
    return t[cut+len("reason:"):].strip() if cut != -1 else t

# ---- load data ----
ds   = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
pmqa = ds["train"]      # change to "test" if you want test-set numbers
N    = len(pmqa)        # set smaller for a smoke test, e.g., N = 100

# ---- prebuild prompts (saves a little time) ----
prompts = [apply_tpl(build_messages(ex["question"], ex["context"])) for ex in pmqa]

# ---- generation loop ----
refs_long, hyps_long = [], []
for i in tqdm(range(N), desc="Generating rationales", ncols=100):
    ex = pmqa[i]
    refs_long.append((ex.get("long_answer") or "").strip())

    p = prompts[i]
    enc = tokenizer(
        p, return_tensors="pt",
        padding=False, truncation=True, max_length=MAX_INP
    ).to(DEVICE)

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=False, temperature=0.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    # decode only new tokens
    new_tokens = out.sequences[0, enc.input_ids.shape[1]:]
    gen_text   = tokenizer.decode(new_tokens, skip_special_tokens=True)
    rationale  = parse_reason(gen_text)
    hyps_long.append(rationale)

    # free per-iteration
    del enc, out, new_tokens
    if (i+1) % 100 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache(); gc.collect()

# ---- evaluation: ROUGE-1 + readability ----
rouge = load_metric("rouge")
r = rouge.compute(predictions=hyps_long, references=refs_long, use_stemmer=True)
print(f"\nROUGE-1 (rationales vs gold long_answer), n={len(hyps_long)}: {float(r['rouge1']):.6f}")

fre  = [textstat.flesch_reading_ease(h) for h in hyps_long]
fk   = [textstat.flesch_kincaid_grade(h) for h in hyps_long]
smog = [textstat.smog_index(h) for h in hyps_long]
print("\nReadability of generated rationales (mean):")
print(f"  Flesch Reading Ease:   {mean(fre):.2f}")
print(f"  Flesch-Kincaid Grade:  {mean(fk):.2f}")
print(f"  SMOG Index:            {mean(smog):.2f}")

# ---- show first 3 model answers for sanity ----
for j in range(min(3, N)):
    ex  = pmqa[j]
    q   = ex["question"]
    ref = refs_long[j]
    hyp = hyps_long[j]
    print(f"\n[{j:03d}]")
    print("Q:", q)
    print("Gold (first 220ch):", (ref[:220] + "…") if len(ref) > 220 else ref)
    print("Hyp  (first 220ch):", (hyp[:220] + "…") if len(hyp) > 220 else hyp)
