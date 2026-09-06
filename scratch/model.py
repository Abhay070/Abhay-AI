"""
A GPT, from scratch, in one readable file.

This is the same architecture family as GPT-2, Llama, Claude, and Gemini.
The difference between this file and a frontier model is scale -- more layers,
more parameters, more data, more compute -- not a different idea. Everything
structural is here.

Read it top to bottom. It is about 200 lines.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Every knob that defines the model's size and shape."""

    vocab_size: int = 256      # how many distinct tokens the model knows
    block_size: int = 128      # context window: how many tokens it can look back over
    n_layer: int = 4           # how many transformer blocks stacked on top of each other
    n_head: int = 4            # attention heads per block
    n_embd: int = 128          # width of the model's internal representation
    dropout: float = 0.1       # regularization, only active during training
    bias: bool = False         # biases in linear layers; modern models mostly drop them

    @property
    def head_size(self) -> int:
        assert self.n_embd % self.n_head == 0, "n_embd must divide evenly into n_head"
        return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
    """
    The one idea that makes transformers work.

    Every token produces three vectors: a query ("what am I looking for?"), a key
    ("what do I offer?"), and a value ("what do I pass along if chosen?"). Each
    token scores every earlier token by query-dot-key, softmaxes those scores into
    weights, and takes a weighted sum of their values. That is how information
    moves between positions.

    "Causal" means a token may only attend to itself and tokens before it. Without
    that mask the model could peek at the answer while predicting it, and would
    learn nothing useful about generation.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # One fused matrix produces query, key and value together -- cheaper than
        # three separate matmuls, and it is what the reference implementations do.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # PyTorch 2.x ships a fused attention kernel. Use it when present, but keep
        # the explicit version below so you can read what it actually computes.
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time (tokens), channels (n_embd)

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Split the channel dimension across heads so each head attends independently.
        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # scores[i][j] = how much token i cares about token j
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            # Mask the future: -inf becomes 0 after softmax.
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Recombine the heads back into one vector per token.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """
    Per-token feed-forward network.

    Attention moves information between tokens; this moves it between features
    within a token. Expand 4x, apply a nonlinearity, project back down. Roughly
    two-thirds of the model's parameters live here.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """
    One transformer layer: communicate, then think.

    Note the `x + ...` pattern. These are residual connections: each sublayer
    proposes an *edit* to the running representation rather than replacing it.
    That is what makes deep stacks trainable -- gradients flow straight through
    the additions. Normalization goes before each sublayer (pre-norm), which is
    what every modern model does because it trains far more stably.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The whole model: embed, stack blocks, predict the next token."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # What does this token mean?
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # Where in the sequence is it? Attention is order-blind without this.
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        # Maps the final representation to one score per vocabulary token.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding and output projection share one matrix.
        # Saves parameters and reliably improves quality.
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scale down the residual projections so deep stacks start out stable.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     (B, T) token ids
        targets: (B, T) the same sequence shifted left by one -- the next token

        Returns (logits, loss). Loss is None at inference time.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence of length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # Predict every position at once -- that is why training is efficient.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        else:
            # At generation time only the last position matters.
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Sample tokens one at a time, feeding each one back in.

        temperature < 1 sharpens the distribution (safer, more repetitive);
        > 1 flattens it (more surprising, more incoherent). top_k discards
        everything outside the k most likely tokens before sampling.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Never feed in more context than the model was built for.
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
