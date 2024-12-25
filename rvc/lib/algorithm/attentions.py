from typing import Optional
import torch
from rvc.lib.algorithm.commons import convert_pad_shape
import torch.nn.functional as F


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention module with optional relative positional encoding and proximal bias.

    Args:
        channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_heads (int): Number of attention heads.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.
        window_size (int, optional): Window size for relative positional encoding. Defaults to None.
        heads_share (bool, optional): Whether to share relative positional embeddings across heads. Defaults to True.
        block_length (int, optional): Block length for local attention. Defaults to None.
        proximal_bias (bool, optional): Whether to use proximal bias in self-attention. Defaults to False.
        proximal_init (bool, optional): Whether to initialize the key projection weights the same as query projection weights. Defaults to False.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int = None,
        heads_share: bool = True,
        block_length: int = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert (
            channels % n_heads == 0
        ), "Channels must be divisible by the number of heads."

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size
        self.block_length = block_length
        self.proximal_bias = proximal_bias

        # Define projections
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)

        self.p_dropout = p_dropout

        # Relative positional encodings
        if window_size:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(
                torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = torch.nn.Parameter(
                torch.randn(n_heads_rel, 2 * window_size + 1, self.k_channels)
                * rel_stddev
            )

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)
        torch.nn.init.xavier_uniform_(self.conv_o.weight)

        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        q = self.conv_q(x)  # (B, channels, T_x)
        k = self.conv_k(c)  # (B, channels, T_c)
        v = self.conv_v(c)  # (B, channels, T_c)

        b, d, t_q = q.shape
        # t_k = k.shape[-1]

        # Project to Q, K, V
        query = q.view(b, self.n_heads, self.k_channels, t_q).transpose(2, 3)
        key = k.view(b, self.n_heads, self.k_channels, t_q).transpose(2, 3)
        value = v.view(b, self.n_heads, self.k_channels, t_q).transpose(2, 3)

        # Scaled Dot-Product Attention
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=self.p_dropout)

        # Final output projection
        output = output.transpose(2, 3).contiguous().view(b, d, t_q)
        return self.conv_o(output)


class FFN(torch.nn.Module):
    """
    Feed-forward network module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        filter_channels (int): Number of filter channels in the convolution layers.
        kernel_size (int): Kernel size of the convolution layers.
        p_dropout (float, optional): Dropout probability. Defaults to 0.0.
        activation (str, optional): Activation function to use. Defaults to None.
        causal (bool, optional): Whether to use causal padding in the convolution layers. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str = None,
        causal: bool = False,
    ):
        super().__init__()
        self.padding_fn = self._causal_padding if causal else self._same_padding

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = torch.nn.Dropout(p_dropout)

        self.activation = activation

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding_fn(x * x_mask))
        x = self._apply_activation(x)
        x = self.drop(x)
        x = self.conv_2(self.padding_fn(x * x_mask))
        return x * x_mask

    def _apply_activation(self, x: torch.Tensor):
        if self.activation == "gelu":
            return x.mul_(torch.sigmoid_(1.702 * x))
        return torch.relu_(x)

    def _causal_padding(self, x):
        pad_l, pad_r = self.conv_1.kernel_size[0] - 1, 0
        return torch.nn.functional.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [pad_l, pad_r]])
        )

    def _same_padding(self, x):
        pad = (self.conv_1.kernel_size[0] - 1) // 2
        return torch.nn.functional.pad(
            x, convert_pad_shape([[0, 0], [0, 0], [pad, pad]])
        )
