import math
import torch
from torch import nn
from typing import *
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # \theta = 10000 ^ {-2 i / d}, (head_dim, )
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):

        # m \theta, (sequence_length, head_dim)
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # m \theta_0, m \theta_1, \cdots, m \theta_{d/2-1} | m \theta_0, m \theta_1, \cdots, m \theta_{d/2-1}
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # num_heads * head_dim == hidden_size
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # (batch_size, sequence_length, hidden_size)
        bsz, q_len, _ = hidden_states.size()

        # (batch_size, sequence_length, num_heads * head_dim)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # (batch_size, num_heads, sequence_length, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            # - x_{d/2}, \cdots, - x_{d-1} | x_0, \cdots x_{d/2-1}
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
            # (sequence_length, head_dim) -> (batch_size, 1, sequence_length, head_dim)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
            
            # x_i 与 x_{i + d/2} 作为一对进行旋转
            # (batch_size, num_heads, sequence_length, head_dim)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            
            return q_embed, k_embed

        # (kv_sequence_length, head_dim)
        kv_seq_len = key_states.shape[-2]
        """
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        """
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        """
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None
        """

        # (batch_size, num_heads, sequence_length, hidden_size)
        # (batch_size, num_heads, hidden_size, kv_sequence_length)
        # -> (batch_size, num_heads, sequence_length, kv_sequence_length)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # upcast attention to fp32
        
        # (batch_size, num_heads, sequence_length, kv_sequence_length)
        # (batch_size, num_heads, kv_sequence_length, head_dim)
        # -> (batch_size, num_heads, sequence_length, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        # (batch_size, sequence_length, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # (batch_size, sequence_length, hidden_size)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # (batch_size, sequence_length, hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value
    