"""
Turning text into numbers, and back.

A model never sees text. It sees integers. The tokenizer is the contract between
the two, and it is the most under-appreciated part of a language model -- it
decides what the model is even capable of representing.

We use a character-level tokenizer here: one integer per character. It is tiny,
has no dependencies, is trivially reversible, and is the right choice for a
small model trained on a small corpus.

Real models (GPT-4, Claude, Gemini) use BPE or SentencePiece with ~50k-200k
tokens, where a token is usually a word fragment. The tradeoff is exactly what
you would expect: a bigger vocabulary means fewer tokens per sentence, so the
model sees more text within the same context window -- at the cost of a much
larger embedding table and a training step of its own. See the note at the
bottom of this file when you want to make that jump.
"""

import json
from pathlib import Path


class CharTokenizer:
    """Maps each distinct character in the training corpus to an integer."""

    def __init__(self, chars: list[str]):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a vocabulary from a corpus -- this is the entire 'training'."""
        return cls(list(set(text)))

    def encode(self, text: str) -> list[int]:
        """Text -> token ids. Unknown characters are skipped rather than crashing,
        so a stray emoji at inference time cannot take down your app."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text."""
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])


# ---------------------------------------------------------------------------
# When you outgrow character-level
#
# Install one line of dependency and you get the same tokenizer GPT-2 used:
#
#     pip install tiktoken
#
#     import tiktoken
#     enc = tiktoken.get_encoding("gpt2")   # 50257 tokens
#     ids = enc.encode("hello world")
#     text = enc.decode(ids)
#
# Then set GPTConfig.vocab_size = 50257. Everything else in the codebase works
# unchanged -- which is the point of keeping the tokenizer behind an interface.
#
# Be aware of the cost: the embedding table grows from (vocab_size x n_embd) with
# vocab_size=65 to the same shape with vocab_size=50257. On a small model that
# single table can outweigh the entire rest of the network, and on a small corpus
# most of those tokens will never be seen enough to learn. Character-level is not
# a toy compromise at this scale -- it is the correct engineering choice.
# ---------------------------------------------------------------------------
