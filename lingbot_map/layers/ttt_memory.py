"""
TTTMemory - a fixed-size, gradient-updated "fast weight" memory module.

Replaces the growing per-layer "special token" trajectory-memory stream
(see attention.py's _apply_kv_cache_eviction / _apply_kv_cache_eviction_causal)
with a small SwiGLU MLP whose weights get updated by one gradient step every
time a frame is evicted from the sliding window, instead of appending that
frame's tokens to an ever-growing list (ZipMap-style test-time training).

Lifecycle (mirrors the existing KV-cache dict lifecycle):
    reset_state()   - call once per new video sequence (from clean_kv_cache())
    update(k, v)    - call once per evicted frame, folds its tokens into memory
    query(x)        - call once per forward step, reads the current memory

Assumes streaming batch size 1, matching the rest of this codebase's
streaming path (see attention.py's FlashInfer comment: "B=1 in streaming mode").
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TTTMemory(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, inner_lr: float = 1.0):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.inner_lr = inner_lr

        # W0: the learned *prior* for the fast weights. Trained by ordinary
        # backprop like any other nn.Parameter (this is what a fine-tuning
        # run actually updates -- see training notes).
        self.W1_0 = nn.Parameter(torch.randn(hidden_dim, dim) * (dim ** -0.5))
        self.W2_0 = nn.Parameter(torch.randn(dim, hidden_dim) * (hidden_dim ** -0.5))
        self.W3_0 = nn.Parameter(torch.randn(hidden_dim, dim) * (dim ** -0.5))

        # Gate combining the memory readout with the normal attention output.
        self.gate_proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -4.0)  # start near-closed (sigmoid(-4) ~ 0.018)

        # Per-sequence working state -- NOT a Parameter, reset every new video.
        self.state = None

    def reset_state(self):
        """Call once per new video sequence, before any frames are processed.

        In training mode the working state is a *plain* clone (no .detach()),
        so it stays connected to W1_0/W2_0/W3_0 in the autograd graph -- this
        is what lets an outer loss.backward() at the end of a training clip
        teach the prior itself "how to be updated" (the actual point of
        training this module -- see training notes). At inference there is no
        outer backward, so we detach to avoid retaining graph history forever;
        the detached clone is a fresh leaf, and .requires_grad_(True) is what
        makes it require grad so the inner update() step below still works
        (that inner gradient is the core TTT mechanism, needed in both modes).
        """
        if self.training:
            self.state = {
                "W1": self.W1_0.clone(),
                "W2": self.W2_0.clone(),
                "W3": self.W3_0.clone(),
            }
        else:
            self.state = {
                "W1": self.W1_0.detach().clone().requires_grad_(True),
                "W2": self.W2_0.detach().clone().requires_grad_(True),
                "W3": self.W3_0.detach().clone().requires_grad_(True),
            }

    @staticmethod
    def _f(x: Tensor, W1: Tensor, W2: Tensor, W3: Tensor) -> Tensor:
        # SwiGLU: W2( SiLU(W1 x) * (W3 x) )
        h = F.silu(x @ W1.T) * (x @ W3.T)
        return h @ W2.T

    def update(self, k: Tensor, v: Tensor):
        """
        One TTT step: fold evicted tokens (k, v) into the fast weights via a
        gradient step on the reconstruction loss L = -f_W(k)^T v.

        Args:
            k, v: [num_evicted_tokens, dim]  (already merged across heads)
        """
        if self.state is None:
            self.reset_state()

        with torch.enable_grad():
            W1, W2, W3 = self.state["W1"], self.state["W2"], self.state["W3"]

            pred = self._f(k, W1, W2, W3)                  # [n, dim]
            loss = -(pred * v).sum(dim=-1).mean()           # scalar

            # create_graph=True keeps this differentiable w.r.t. W1_0/W2_0/W3_0
            # so an outer training loop can backprop "how to update" through
            # the inner step. At pure inference (self.training == False) this
            # is unnecessary overhead -- see training notes for how to disable it.
            grads = torch.autograd.grad(loss, [W1, W2, W3], create_graph=self.training)

            self.state["W1"] = W1 - self.inner_lr * grads[0]
            self.state["W2"] = W2 - self.inner_lr * grads[1]
            self.state["W3"] = W3 - self.inner_lr * grads[2]

    def query(self, x: Tensor) -> Tensor:
        """
        Read the current fast-weight memory for query tokens x, gated.

        Args:
            x: [..., dim]
        Returns:
            gated memory readout, same shape as x
        """
        if self.state is None:
            self.reset_state()

        out = self._f(x, self.state["W1"], self.state["W2"], self.state["W3"])
        gate = torch.sigmoid(self.gate_proj(x))
        return out * gate
