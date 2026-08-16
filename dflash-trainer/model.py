"""
DFlash Draft Model Architecture Definition.
100% Parameter and Key Match with official z-lab/Qwen3.6-27B-DFlash.
"""

from dataclasses import dataclass, field
from typing import List, Optional
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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        S = x_ctx.shape[1]

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        ctx_k = self.k_proj(x_ctx).view(B, S, self.n_kv_heads, self.head_dim)
        ctx_v = self.v_proj(x_ctx).view(B, S, self.n_kv_heads, self.head_dim)

        prop_k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        prop_v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        ctx_k = self.k_norm(ctx_k).transpose(1, 2)
        prop_k = self.k_norm(prop_k).transpose(1, 2)

        ctx_v = ctx_v.transpose(1, 2)
        prop_v = prop_v.transpose(1, 2)

        k = torch.cat([ctx_k, prop_k], dim=2)  # [B, n_kv_heads, S + L, head_dim]
        v = torch.cat([ctx_v, prop_v], dim=2)

        if self.n_heads != self.n_kv_heads:
            ratio = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), x_ctx)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    """
    Official DFlashDraftModel matching z-lab/Qwen3.6-27B-DFlash parameter naming.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_embeds: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        h = input_embeds
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer in self.layers:
            h = layer(h, h_ctx)
        return self.norm(h)

    def export_mlx_safetensors(self, output_dir: str):
        """Export weights and config.json formatted for native mlx-vlm / vLLM / SGLang loading."""
        os.makedirs(output_dir, exist_ok=True)
        from safetensors.torch import save_file

        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state, os.path.join(output_dir, "model.safetensors"))

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
                "target_layer_ids": self.config.target_layer_ids,
            },
            "dtype": "bfloat16",
            "eos_token_id": 248044,
            "head_dim": self.config.head_dim,
            "hidden_act": "silu",
            "hidden_size": self.config.hidden_size,
            "initializer_range": 0.02,
            "intermediate_size": self.config.intermediate_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "model_type": "qwen3",
            "num_attention_heads": self.config.num_attention_heads,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_key_value_heads": self.config.num_key_value_heads,
            "num_target_layers": self.config.num_target_layers,
            "pad_token_id": 248044,
            "rms_norm_eps": self.config.rms_norm_eps,
            "vocab_size": self.config.vocab_size,
            "rope_theta": self.config.rope_theta,
        }

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)
