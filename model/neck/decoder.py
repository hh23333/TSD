import torch
import torch.nn as nn

from .transformer_layer import FFN, TransformerLayerSequence, MaskMultiheadAttention1

class PartDecoder1(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        feedforward_dim: int = 2048,
        ffn_dropout: float = 0.1,
        # num_layers: int = 1,
        post_norm: bool = True,
        num_parts: int = 14,
        return_intermediate: bool = False,  # TODO
        # operation_order = ("norm", "cross_attn", "norm", "self_attn", "norm", "ffn"),
        # batch_first: bool = False,
    ):
        super(PartDecoder1, self).__init__()
        # self.norm0 = nn.LayerNorm(normalized_shape=embed_dim)
        # global part self attention
        self.gp_self_attention = MaskMultiheadAttention1(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        # cross attention with part feature
        self.cross_attention = MaskMultiheadAttention1(embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_dropout, add_identity=False)
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.ffn = FFN(embed_dim=embed_dim, feedforward_dim=feedforward_dim, ffn_drop=ffn_dropout)
        if post_norm:
            self.post_norm = nn.LayerNorm(normalized_shape=embed_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        attn_masks: torch.Tensor = None,
        **kwargs,
    ):
        '''
        query: (1+num_part, c)
        '''
        self_attn_out = self.gp_self_attention(query, query, query)
        # identity = self_attn_out['query']
        query = self.norm1(self_attn_out['query'])[:, 1:]  #
        cross_attn_out = self.cross_attention(query, key, value, attn_mask=attn_masks)
        identity = cross_attn_out['query']
        query = self.norm2(identity)
        query = self.ffn(query, identity)
        out = {'query': query,
               'cross_attn': cross_attn_out['attn'],
            #    'self_attn': self_attn_out['attn'],
               }
        return out