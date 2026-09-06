"""
Train your GPT.

    python scratch/train.py                    # sensible CPU defaults, ~10-20 min
    python scratch/train.py --preset tiny      # ~2 min, just to prove it runs
    python scratch/train.py --preset big       # use this if you have a GPU

What actually happens here is the whole of machine learning in six steps,
repeated a few thousand times:

    1. grab a random batch of text
    2. ask the model to predict the next character at every position
    3. measure how wrong it was          (the loss)
    4. compute which direction each weight should move  (backward pass)
    5. take a small step in that direction              (the optimizer)
    6. repeat

Watch the loss. It starts near ln(vocab_size) -- about 4.17 for 65 characters,
which is the loss of a model guessing uniformly at random. Anything below that
means it has learned something real.
"""

import argparse
import math
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

HERE = Path(__file__).parent
DATA = HERE / "data" / "input.txt"
OUT_DIR = HERE / "out"

# Three sizes, so you can start small and scale up without editing code.
PRESETS = {
    # Proves the pipeline works in a couple of minutes on any laptop.
    "tiny": dict(n_layer=2, n_head=2, n_embd=64, block_size=64,
                 batch_size=16, max_iters=500, lr=3e-3),
    # The default. Genuinely readable Shakespeare-ish output on CPU.
    "small": dict(n_layer=4, n_head=4, n_embd=128, block_size=128,
                  batch_size=32, max_iters=3000, lr=1e-3),
    # For a GPU (free Colab T4 handles this comfortably in ~15 minutes).
    "big": dict(n_layer=6, n_head=6, n_embd=384, block_size=256,
                batch_size=64, max_iters=5000, lr=3e-4),
}


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """
    Sample a random batch of (context, target) pairs.

    targets is just inputs shifted one position left: at every position the model
    predicts the character that actually came next. One sequence of length T gives
    us T training signals, not one -- that efficiency is why transformers scale.
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits: dict, cfg, batch_size: int, device: str, eval_iters: int = 50):
    """
    Average loss over several batches, for train and validation separately.

    Validation loss is the number that matters. If train keeps falling while val
    turns upward, the model has started memorizing the corpus instead of learning
    its patterns -- that is overfitting, and it means stop, or get more data.
    """
    model.eval()
    out = {}
    for split, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, cfg.block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(it: int, max_iters: int, base_lr: float, warmup: int = 100) -> float:
    """
    Learning-rate schedule: warm up briefly, then decay along a cosine to 10%.

    The warmup stops huge early gradients from wrecking freshly-initialized
    weights. The decay lets the model take fine steps once it is close. Every
    serious training run uses some version of this.
    """
    if it < warmup:
        return base_lr * (it + 1) / warmup
    progress = (it - warmup) / max(1, max_iters - warmup)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * min(progress, 1.0))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=list(PRESETS), default="small")
    p.add_argument("--data", type=str, default=str(DATA))
    p.add_argument("--max-iters", type=int, default=None, help="override the preset")
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true", help="torch.compile (slow start, faster steps)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    preset = dict(PRESETS[args.preset])
    if args.max_iters is not None:
        preset["max_iters"] = args.max_iters

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"No corpus at {data_path}. Run: python scratch/prepare_data.py")
    text = data_path.read_text(encoding="utf-8", errors="ignore")

    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Hold out the last 10% and never train on it. This is the only honest way to
    # know whether the model learned anything or just memorized.
    split_at = int(0.9 * len(ids))
    splits = {"train": ids[:split_at], "val": ids[split_at:]}

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=preset["block_size"],
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        n_embd=preset["n_embd"],
    )
    model = GPT(cfg).to(device)

    batch_size, max_iters, base_lr = preset["batch_size"], preset["max_iters"], preset["lr"]

    print(f"device          {device}")
    print(f"preset          {args.preset}")
    print(f"parameters      {model.num_params():,}")
    print(f"vocab size      {cfg.vocab_size}")
    print(f"training tokens {len(splits['train']):,}")
    print(f"random-guess loss ~{math.log(cfg.vocab_size):.2f}  (beat this)")
    print()

    # AdamW with decay on matrices but not on biases/LayerNorms -- standard practice,
    # and it measurably helps.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=base_lr, betas=(0.9, 0.99),
    )

    if args.compile and hasattr(torch, "compile"):
        print("compiling model (first step will be slow) ...")
        model = torch.compile(model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(OUT_DIR / "tokenizer.json")

    best_val = float("inf")
    t0 = time.time()

    for it in range(max_iters + 1):
        lr = lr_at(it, max_iters, base_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == max_iters:
            losses = estimate_loss(model, splits, cfg, batch_size, device)
            elapsed = time.time() - t0
            flag = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": getattr(model, "_orig_mod", model).state_dict(),
                     "config": cfg.__dict__,
                     "iter": it,
                     "val_loss": best_val},
                    OUT_DIR / "model.pt",
                )
                flag = "  <- saved"
            print(
                f"iter {it:>5}/{max_iters}  train {losses['train']:.4f}  "
                f"val {losses['val']:.4f}  lr {lr:.2e}  {elapsed:.0f}s{flag}"
            )

        if it == max_iters:
            break

        xb, yb = get_batch(splits["train"], cfg.block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip gradients: one freak batch should not be able to blow up the weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print()
    print(f"Done in {time.time() - t0:.0f}s. Best validation loss: {best_val:.4f}")
    print(f"Weights: {OUT_DIR / 'model.pt'}")
    print()
    print("Next:  python scratch/sample.py --prompt \"To be, or\"")


if __name__ == "__main__":
    main()
