import os
import torch
from torch import nn

class LlamaRotaryEmbedding(torch.nn.Module):
   
   def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, type="standard", scaling_factor=1.0):
      super().__init__()
      
      if type == "ntk-scaling":
          base = base * scaling_factor ** (dim / (dim - 2))     # ntk

      # shape(hidden_size // 2, ), θ_i, i = 0, \cdots, d_k / 2 - 1
      #      θ_0, θ_1, ..., θ_{d_k / 2 - 1}
      inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
      self.register_buffer("inv_freq", inv_freq)

      # Build here to make `torch.jit.trace` work.
      self.max_seq_len_cached = max_position_embeddings
      # shape(max_position_embeddings, ), positions
      t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

      if type == "linear-interpolation":
          t = t / scaling_factor   # linear interpolation
      
      # shape(max_position_embeddings, hidden_size // 2)
      #     0 * θ_0 * θ_1, ..., 0 * θ_{d_k / 2 - 1}
      #     1 * θ_0, 1 * θ_1, ..., 1 * θ_{d_k / 2 - 1}
      #     ...
      #     t * θ_0, t * θ_1, ..., t * θ_{d_k / 2 - 1}
      freqs = torch.einsum("i,j->ij", t, self.inv_freq)
      # Different from paper, but it uses a different permutation in order to obtain the same calculation
      # shape(max_position_embeddings, hidden_size)
      #     0 * θ_0 * θ_1, ..., 0 * θ_{d_k / 2 - 1} | 0 * θ_0 * θ_1, ..., 0 * θ_{d_k / 2 - 1}
      #     1 * θ_0, 1 * θ_1, ..., 1 * θ_{d_k / 2 - 1} | 1 * θ_0, 1 * θ_1, ..., 1 * θ_{d_k / 2 - 1}
      #     ...                                                     | ...
      #     t * θ_0, t * θ_1, ..., t * θ_{d_k / 2 - 1} | t * θ_0, t * θ_1, ..., t * θ_{d_k / 2 - 1}
      emb = torch.cat((freqs, freqs), dim=-1)
      # shape(1, 1, max_position_embeddings, hidden_size)
      self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
      # shape(1, 1, max_position_embeddings, hidden_size)
      self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

   def forward(self, x, seq_len=None):
      # x: [bs, num_attention_heads, seq_len, head_size]
      # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
      if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
      # shape(1, 1, sequence_length, hidden_size)
      return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
      )

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    scaling_factor = 4.0
    types = ["standard", "linear-interpolation", "ntk-scaling"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    
    for i in range(len(types)):
        type = types[i]

        dim = 768 * 2   # y
        max_position_embeddings = 512 * int(scaling_factor) # x

        if type == "standard":
            rope = LlamaRotaryEmbedding(
                dim=dim, max_position_embeddings=max_position_embeddings, type=type, scaling_factor=1.0,
            )

        elif type == "linear-interpolation":
            rope = LlamaRotaryEmbedding(
                dim=dim, max_position_embeddings=max_position_embeddings, type=type, scaling_factor=scaling_factor,
            )

        elif type == "ntk-scaling":
            rope = LlamaRotaryEmbedding(
                dim=dim, max_position_embeddings=max_position_embeddings, type=type, scaling_factor=scaling_factor,
            )
    
        xticks = [i for i in range(0, max_position_embeddings + 1, 256)]
        yticks = [i for i in range(0, dim // 2 + 1, 128)]

        # twin_axs = axs[i].twinx()

        axs[i].set_title([
            "Sinusoidal Position Embedding (Standard)",
            "Sinusoidal Position Embedding (Linear Interpolation)",
            "Sinusoidal Position Embedding (NTK-Scaling)",
        ][i])

        cos_im = torch.flip(rope.cos_cached[0][0][..., :dim // 2].T, dims=[0])   # 上下翻转
        if i == 0:
            cos_im[:, int(max_position_embeddings/scaling_factor):] = 0.0
        axs[i].imshow(cos_im, cmap="coolwarm")

        axs[i].set_xticks(ticks=xticks, labels=xticks)
        axs[i].set_yticks(ticks=yticks, labels=yticks[::-1])
        # twin_axs.set_yticks(ticks=yticks, labels=yticks[::-1])
        
        if i == 2:
            axs[i].set_xlabel("position(x)")
        
        axs[i].set_ylabel("dimension(i)")
        # twin_axs.set_ylabel("frequency(i)")
        
        if i > 0:
            axs[i].axvline(x=int(max_position_embeddings/scaling_factor), color='r', linestyle='--')

    plt.axis('on')  # 可以选择是否显示坐标轴
    plt.show()
