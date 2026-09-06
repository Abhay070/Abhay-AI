"""
Generate text from your trained model.

    python scratch/sample.py
    python scratch/sample.py --prompt "To be, or" --tokens 500
    python scratch/sample.py --temperature 0.5     # safer, more repetitive
    python scratch/sample.py --temperature 1.3     # wilder, less coherent
    python scratch/sample.py --stream              # watch it write, token by token

Temperature is the single most useful dial. At 0.1 the model picks its top guess
almost every time and loops. At 1.5 it takes wild swings and stops making sense.
Somewhere around 0.7-0.9 is where most production systems sit -- including the
ones you have been talking to.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

OUT_DIR = Path(__file__).parent / "out"


def load_model(out_dir: Path, device: str):
    ckpt_path = out_dir / "model.pt"
    tok_path = out_dir / "tokenizer.json"
    if not ckpt_path.exists():
        raise SystemExit(f"No trained model at {ckpt_path}. Run: python scratch/train.py")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    tokenizer = CharTokenizer.load(tok_path)
    return model, tokenizer, ckpt


def stream_generate(model, tokenizer, idx, max_new_tokens, temperature, top_k, device):
    """Same sampling loop as GPT.generate, but prints each character as it arrives.

    This is exactly the mechanism behind the typing effect in every chat UI you
    have used. There is no buffering trick -- the model genuinely produces one
    token at a time, and each one depends on all the ones before it."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        sys.stdout.write(tokenizer.decode([next_token.item()]))
        sys.stdout.flush()
    print()
    return idx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", type=str, default="\n", help="text to continue from")
    p.add_argument("--tokens", type=int, default=400, help="how many characters to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k filtering")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, ckpt = load_model(Path(args.out_dir), device)
    print(
        f"[model: {model.num_params():,} params, "
        f"trained {ckpt['iter']} iters, val loss {ckpt['val_loss']:.4f}]\n",
        file=sys.stderr,
    )

    ids = tokenizer.encode(args.prompt) or tokenizer.encode("\n")
    context = torch.tensor([ids], dtype=torch.long, device=device)
    top_k = args.top_k if args.top_k > 0 else None

    for i in range(args.samples):
        if args.samples > 1:
            print(f"\n----- sample {i + 1} -----")
        if args.stream:
            sys.stdout.write(args.prompt)
            stream_generate(model, tokenizer, context, args.tokens, args.temperature, top_k, device)
        else:
            out = model.generate(context, args.tokens, args.temperature, top_k)
            print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
