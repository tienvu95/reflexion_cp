"""Interactive REPL that loads the UnsloTh model once and accepts prompts.

Usage (notebook or terminal):
  python hotpotqa_runs/unsloth_repl.py --model unsloth/Meta-Llama-3.1-8B-bnb-4bit --hf-token hf_... --max-seq-length 8192

Or from a notebook cell:
  %run -i hotpotqa_runs/unsloth_repl.py --model unsloth/Meta-Llama-3.1-8B-bnb-4bit

This avoids reloading the model every time you want to try a prompt.
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--hf-token', default=None)
parser.add_argument('--max-seq-length', type=int, default=8192)
parser.add_argument('--load-in-4bit', action='store_true')
args = parser.parse_args()

# Defer heavy imports until after arg parsing so errors are clearer
from hotpotqa_runs.unsloth_llm import UnslothLLM

if args.hf_token:
    os.environ['HUGGINGFACE_HUB_TOKEN'] = args.hf_token

print(f"Loading model {args.model} ...")
llm = UnslothLLM(model_name=args.model, token=args.hf_token, load_in_4bit=args.load_in_4bit, max_seq_length=args.max_seq_length)
print("Model ready. Enter prompts (Ctrl-D to exit)")

try:
    while True:
        prompt = input('PROMPT> ')
        if not prompt.strip():
            continue
        try:
            out = llm(prompt)
            print('\n--- OUTPUT ---')
            print(out)
            print('--- END ---\n')
        except Exception as e:
            print('Generation error:', e)
except EOFError:
    print('\nExiting.')
