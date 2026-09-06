"""
Talk to your fine-tuned model.

    python finetune/chat.py                              # uses the default adapter
    python finetune/chat.py --adapter finetune/adapters/my-model
    python finetune/chat.py --base-only                  # compare against the untuned model

That last flag matters. The only way to know your fine-tune did anything is to
ask the base model the same questions and see the difference.
"""

import argparse
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_ADAPTER = HERE / "adapters" / "my-model"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--base-only", action="store_true", help="skip the adapter, for comparison")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    adapter = Path(args.adapter)
    use_adapter = not args.base_only and adapter.exists()
    if not args.base_only and not adapter.exists():
        print(f"No adapter at {adapter} -- falling back to the base model.")
        print("Train one first: python finetune/train_lora.py\n")

    source = str(adapter) if use_adapter else args.base
    tokenizer = AutoTokenizer.from_pretrained(source)

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if use_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"Loaded adapter: {adapter}")
    else:
        print(f"Base model only: {args.base}")
    model.eval()

    print("Type your message. Ctrl-C or 'exit' to quit.\n")
    history: list[dict] = []

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        history.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("ai> ", end="", flush=True)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                top_p=0.9,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        history.append({"role": "assistant", "content": reply.strip()})
        print()


if __name__ == "__main__":
    main()
