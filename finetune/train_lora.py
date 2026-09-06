"""
LoRA fine-tuning: take an open model and teach it your voice.

This is the step where the weights become genuinely yours. You are not training
from zero -- you start from a model that already understands language, and nudge
it toward your data. That is why it costs hours instead of months.

Why LoRA makes this free:
    Full fine-tuning of a 7B model updates 7 billion numbers and needs ~60 GB of
    GPU memory for the optimizer state alone. LoRA freezes the original weights
    and injects small trainable matrices beside them -- typically <1% of the
    parameters. Load the base model in 4-bit and the whole thing fits in the
    16 GB a free Colab T4 gives you.

    You end up with an adapter of a few dozen MB, not a new multi-GB model.

Run it (on a GPU -- free Colab works):

    pip install -r requirements-finetune.txt
    python finetune/train_lora.py --data finetune/data/example.jsonl

Then talk to it:

    python finetune/chat.py
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "adapters" / "my-model"

# Small enough to fine-tune on a free T4, good enough to be useful.
# Bigger alternatives, if you have the VRAM:
#   Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct (gated, needs HF login)
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_jsonl(path: Path) -> list[dict]:
    """Each line is {"messages": [{"role": ..., "content": ...}, ...]}."""
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{i} is not valid JSON: {e}")
        if "messages" not in row:
            raise SystemExit(f"{path}:{i} has no 'messages' key")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="base model from HuggingFace")
    p.add_argument("--data", default=str(HERE / "data" / "example.jsonl"))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8, help="simulates a larger batch")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--rank", type=int, default=16, help="LoRA rank: higher = more capacity")
    p.add_argument("--no-4bit", action="store_true", help="skip quantization (needs more VRAM)")
    args = p.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        print("WARNING: no GPU detected. This will be impractically slow on CPU.")
        print("Use a free Colab GPU: colab.research.google.com -> Runtime -> T4 GPU\n")

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"No dataset at {data_path}. See finetune/README.md for the format.")
    rows = load_jsonl(data_path)
    print(f"Loaded {len(rows)} examples from {data_path}")
    if len(rows) < 50:
        print("Note: under ~50 examples the model tends to memorize rather than")
        print("generalize. A few hundred good examples beats a few thousand sloppy ones.\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if not args.no_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        # 4-bit weights, 16-bit math. This is what makes a 7B model fit on a T4.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,   # the usual rule of thumb
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention and MLP projections. Targeting both is meaningfully better
        # than attention alone, for very little extra cost.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # look at this number -- it is the whole point

    def to_text(row: dict) -> dict:
        """Apply the base model's own chat template. Using the wrong template is
        the single most common reason a fine-tune comes out broken."""
        return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}

    dataset = Dataset.from_list(rows).map(to_text, remove_columns=["messages"])

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,   # trades compute for memory
            learning_rate=args.lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_strategy="epoch",
            bf16=torch.cuda.is_available(),
            optim="paged_adamw_8bit" if quant_config else "adamw_torch",
            report_to="none",
        ),
    )

    print("\nTraining. Watch the loss -- if it does not fall, your data or your")
    print("chat template is wrong, not your hyperparameters.\n")
    trainer.train()

    out = Path(args.out)
    trainer.model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"\nAdapter saved to {out} ({size_mb:.1f} MB)")
    print(f"Base model stays untouched: {args.model}")
    print(f"\nTry it:  python finetune/chat.py --adapter {out}")


if __name__ == "__main__":
    main()
