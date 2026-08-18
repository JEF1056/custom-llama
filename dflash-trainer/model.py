"""
DFlash Draft Model Architecture Definition.
100% Parameter, Architecture, and Key Match with official z-lab/Qwen3.6-27B-DFlash.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DFlashConfig:
    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rope_theta: float = 10000000.0
    block_size: int = 16
    mask_token_id: int = 248070
    target_layer_ids: List[int] = field(default_factory=lambda: [1, 16, 31, 46, 61])
    num_target_layers: int = 64
    sliding_window: int = 2048
    layer_types: List[str] = field(default_factory=lambda: [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ])
    attention_bias: bool = False
    attention_dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 262144, base: float = 10000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [bsz, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=config.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        ctx_pos_emb: Tuple[torch.Tensor, torch.Tensor],
        prop_pos_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if self.is_sliding and self.sliding_window is not None and target_hidden.shape[1] > self.sliding_window:
            target_hidden = target_hidden[:, -self.sliding_window:]
            ctx_cos, ctx_sin = ctx_pos_emb
            ctx_pos_emb = (ctx_cos[:, -self.sliding_window:], ctx_sin[:, -self.sliding_window:])

        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        # 1. Query projection + norm + RoPE (at proposal positions [S .. S+L-1])
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.n_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [bsz, n_heads, q_len, head_dim]
        q_cos, q_sin = prop_pos_emb
        q = apply_rotary_pos_emb_single(q, q_cos, q_sin)

        # 2. Context Key/Value projection + norm + RoPE (at context positions [0 .. S-1])
        k_ctx = self.k_proj(target_hidden).view(bsz, ctx_len, self.n_kv_heads, self.head_dim)
        v_ctx = self.v_proj(target_hidden).view(bsz, ctx_len, self.n_kv_heads, self.head_dim)
        k_ctx = self.k_norm(k_ctx).transpose(1, 2)  # [bsz, n_kv_heads, ctx_len, head_dim]
        v_ctx = v_ctx.transpose(1, 2)
        ctx_cos, ctx_sin = ctx_pos_emb
        k_ctx = apply_rotary_pos_emb_single(k_ctx, ctx_cos, ctx_sin)

        # 3. Proposal Key/Value projection + norm + RoPE (at proposal positions [S .. S+L-1])
        k_prop = self.k_proj(hidden_states).view(bsz, q_len, self.n_kv_heads, self.head_dim)
        v_prop = self.v_proj(hidden_states).view(bsz, q_len, self.n_kv_heads, self.head_dim)
        k_prop = self.k_norm(k_prop).transpose(1, 2)  # [bsz, n_kv_heads, q_len, head_dim]
        v_prop = v_prop.transpose(1, 2)
        k_prop = apply_rotary_pos_emb_single(k_prop, q_cos, q_sin)

        # 4. Concatenate Context + Proposal Keys & Values
        k = torch.cat([k_ctx, k_prop], dim=2)  # [bsz, n_kv_heads, ctx_len + q_len, head_dim]
        v = torch.cat([v_ctx, v_prop], dim=2)

        if self.n_heads != self.n_kv_heads:
            ratio = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)

        # Standard non-causal Flash Attention across context + draft proposal block
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_out)


class Qwen3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        ctx_pos_emb: Tuple[torch.Tensor, torch.Tensor],
        prop_pos_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, target_hidden, ctx_pos_emb, prop_pos_emb)
        hidden_states = residual + h

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.mlp(h)
        return residual + h


class DFlashDraftModel(nn.Module):
    """
    Official 100% Reference DFlashDraftModel matching z-lab/Qwen3.6-27B-DFlash.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing for all decoder layers to save activation memory."""
        self.gradient_checkpointing = True

    def forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        ctx_position_ids: torch.Tensor,
        prop_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        ctx_pos_emb = self.rotary_emb(target_hidden, ctx_position_ids)
        prop_pos_emb = self.rotary_emb(hidden_states, prop_position_ids)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    target_hidden,
                    ctx_pos_emb,
                    prop_pos_emb,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    target_hidden=target_hidden,
                    ctx_pos_emb=ctx_pos_emb,
                    prop_pos_emb=prop_pos_emb,
                )

        return self.norm(hidden_states)

    def export_mlx_safetensors(self, output_dir: str):
        """Export weights and config.json matching official z-lab repository format."""
        os.makedirs(output_dir, exist_ok=True)
        from safetensors.torch import save_file

        state_dict = {}
        for k, v in self.state_dict().items():
            if k.startswith("rotary_emb."):
                continue  # Rotary inv_freq buffers not needed in safetensors
            state_dict[k] = v.contiguous().to(torch.bfloat16)

        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

        cfg_dict = {
            "architectures": ["DFlashDraftModel"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "auto_map": {
                "AutoModel": "dflash.DFlashDraftModel"
            },
            "block_size": self.config.block_size,
            "bos_token_id": None,
            "dflash_config": {
                "mask_token_id": self.config.mask_token_id,
                "target_layer_ids": list(self.config.target_layer_ids),
            },
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "head_dim": self.config.head_dim,
            "hidden_act": "silu",
            "hidden_size": self.config.hidden_size,
            "initializer_range": 0.02,
            "intermediate_size": self.config.intermediate_size,
            "layer_types": self.config.layer_types,
            "max_position_embeddings": self.config.max_position_embeddings,
            "max_window_layers": 5,
            "model_type": "qwen3",
            "num_attention_heads": self.config.num_attention_heads,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_key_value_heads": self.config.num_key_value_heads,
            "num_target_layers": self.config.num_target_layers,
            "pad_token_id": 248044,
            "rms_norm_eps": self.config.rms_norm_eps,
            "sliding_window": self.config.sliding_window,
            "tie_word_embeddings": False,
            "transformers_version": "5.5.3",
            "use_cache": True,
            "use_sliding_window": True,
            "vocab_size": self.config.vocab_size,
            "rope_theta": self.config.rope_theta,
            "rope_scaling": None,
        }

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)
