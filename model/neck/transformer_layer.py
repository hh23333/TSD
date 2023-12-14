import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import copy
import warnings


class BaseTransformerLayer(nn.Module):
    # TODO: add more tutorials about BaseTransformerLayer
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.

    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn (list[nn.Module] | nn.Module): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """

    def __init__(
        self,
        attn: List[nn.Module],
        ffn: nn.Module,
        norm: nn.Module,
        operation_order: tuple = None,
    ):
        super(BaseTransformerLayer, self).__init__()
        assert set(operation_order).issubset({"self_attn", "norm", "cross_attn", "ffn"})

        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")

        if isinstance(attn, nn.Module):
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:
            assert len(attn) == num_attn, (
                f"The length of attn (nn.Module or List[nn.Module]) {num_attn}"
                f"is not consistent with the number of attention in "
                f"operation_order {operation_order}"
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1

        self.embed_dim = self.attentions[0].embed_dim

        # count ffn nums
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))

        # count norm nums
        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        query_pos: torch.Tensor = None,
        key_pos: torch.Tensor = None,
        attn_masks: List[torch.Tensor] = None,
        query_key_padding_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        return_attn=False,
        attn_type='cross',
        **kwargs,
    ):
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains the specific arguments of attentions.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                if return_attn and (attn_type=='self'):
                    query, attn = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        return_attn=return_attn,
                        **kwargs,
                    )
                else:
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        # return_attn=return_attn,
                        **kwargs,
                    )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                if return_attn and (attn_type=='cross'):
                    query, attn = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        return_attn=return_attn,
                        **kwargs,
                    )
                else:
                    query = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        # return_attn=return_attn,
                        **kwargs,
                    )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        if return_attn:
            return query, attn
        else:
            return query

class PartDecodeLayer(nn.Module):
    # TODO: add more tutorials about BaseTransformerLayer
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.

    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn (list[nn.Module] | nn.Module): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """

    def __init__(
        self,
        attn: List[nn.Module],
        ffn: nn.Module,
        norm: nn.Module,
        operation_order: tuple = None,
    ):
        super(PartDecodeLayer, self).__init__()
        assert set(operation_order).issubset({"self_attn", "norm", "cross_attn", "ffn"})

        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")

        if isinstance(attn, nn.Module):
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:
            assert len(attn) == num_attn, (
                f"The length of attn (nn.Module or List[nn.Module]) {num_attn}"
                f"is not consistent with the number of attention in "
                f"operation_order {operation_order}"
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1

        self.embed_dim = self.attentions[0].embed_dim

        # count ffn nums
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))

        # count norm nums
        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

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
        """Forward function for `BaseTransformerLayer`.

        **kwargs contains the specific arguments of attentions.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        identity = query
        cross_attn_out = self.attentions[0](query, key, value, 
                                         identity if self.pre_norm else None,
                                         query_pos=query_pos,
                                         key_pos=key_pos,
                                         attn_mask=attn_masks,
                                         )
        identity = cross_attn_out['query']
        query = self.norms[1](cross_attn_out['query'])
        query = self.attentions[1](query, query, query,
                                           identity if self.pre_norm else None, 
                                           )
        identity = query
        query = self.norms[2](query)
        query = self.ffns[0](query, identity if self.pre_norm else None)

        out = {'query': query,
               'cross_attn': cross_attn_out['attn'],
            #    'self_attn': self_attn_out['attn'],
               }
        return out

class FFN(nn.Module):
    """The implementation of feed-forward networks (FFNs)
    with identity connection.

    Args:
        embed_dim (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_dim (int): The hidden dimension of FFNs.
            Defaults: 1024.
        output_dim (int): The output feature dimension of FFNs.
            Default: None. If None, the `embed_dim` will be used.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        activation (nn.Module): The activation layer used in FFNs.
            Default: nn.ReLU(inplace=True).
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
    """

    def __init__(
        self,
        embed_dim=256,
        feedforward_dim=1024,
        output_dim=None,
        num_fcs=2,
        activation=nn.ReLU(inplace=True),
        ffn_drop=0.0,
        fc_bias=True,
        add_identity=True,
    ):
        super(FFN, self).__init__()
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.num_fcs = num_fcs
        self.activation = activation

        output_dim = embed_dim if output_dim is None else output_dim

        layers = []
        in_channels = embed_dim
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_dim, bias=fc_bias),
                    self.activation,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_dim
        layers.append(nn.Linear(feedforward_dim, output_dim, bias=fc_bias))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None) -> torch.Tensor:
        """Forward function of `FFN`.

        Args:
            x (torch.Tensor): the input tensor used in `FFN` layers.
            identity (torch.Tensor): the tensor with the same shape as `x`,
                which will be used for identity addition. Default: None.
                if None, `x` will be used.

        Returns:
            torch.Tensor: the forward results of `FFN` layer
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``

    Implemente MultiheadAttention with identity connection,
    and position embedding is also passed as input.

    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        # proj_drop: float = 0.0,
        bias: bool = True,
        qk_scale = None,
        add_identity = True,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.add_identity = add_identity
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn = False,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function for `MultiheadAttention`

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_query, embed_dim)`
            key (torch.Tensor): Key embeddings with shape
                `(num_key, bs, embed_dim)` if self.batch_first is False,
                else `(bs, num_key, embed_dim)`
            value (torch.Tensor): Value embeddings with the same shape as `key`.
                Same in `torch.nn.MultiheadAttention.forward`. Default: None.
                If None, the `key` will be used.
            identity (torch.Tensor): The tensor, with the same shape as x, will
                be used for identity addition. Default: None.
                If None, `query` will be used.
            query_pos (torch.Tensor): The position embedding for query, with the
                same shape as `query`. Default: None.
            key_pos (torch.Tensor): The position embedding for key. Default: None.
                If None, and `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`.
            attn_mask (torch.Tensor): ByteTensor mask with shape `(num_query, num_key)`.
                Same as `torch.nn.MultiheadAttention.forward`. Default: None.
            key_padding_mask (torch.Tensor): ByteTensor with shape `(bs, num_key)` which
                indicates which elements within `key` to be ignored in attention.
                Default: None.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        B, N_q, C = query.shape
        N_k = key.size(1)
        N_v = value.size(1)
        # (B, num_heads, N, C)
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_v, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.out_proj(x)
        
        if self.add_identity:
            if return_attn:
                return identity + x, attn
            else:
                return identity + x
        else:
            if return_attn:
                return x, attn
            else:
                return x
            
class MaskMultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``

    Implemente MultiheadAttention with identity connection,
    and position embedding is also passed as input.

    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        # proj_drop: float = 0.0,
        bias: bool = True,
        qk_scale = None,
        add_identity = True,
    ):
        super(MaskMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.add_identity = add_identity
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn = False,
        **kwargs,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        B, N_q, C = query.shape
        N_k = key.size(1)
        N_v = value.size(1)
        # (B, num_heads, N, C)
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_v, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_vis = attn.clone()
        #
        if attn_mask is not None:
            attn_mask = attn_mask / attn_mask.max(dim=1, keepdim=True)[0]
            attn_mask = attn_mask.unsqueeze(1).transpose(-2, -1)
            attn = attn * attn_mask
        #
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.out_proj(x)
        
        if self.add_identity:
            out = {'query': identity + x,
                   'attn': attn_vis}
        else:
            out = {'query': x,
                   'attn': attn_vis}
        return out
    
class MaskMultiheadAttention1(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``

    Implemente MultiheadAttention with identity connection,
    and position embedding is also passed as input.

    Args:
        embed_dim (int): The embedding dimension for attention.
        num_heads (int): The number of attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `MultiheadAttention`.
            Default: 0.0.
        batch_first (bool): if `True`, then the input and output tensor will be
            provided as `(bs, n, embed_dim)`. Default: False. `(n, bs, embed_dim)`
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        # proj_drop: float = 0.0,
        bias: bool = True,
        qk_scale = None,
        add_identity = True,
    ):
        super(MaskMultiheadAttention1, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.add_identity = add_identity
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn = False,
        **kwargs,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(
                        f"position encoding of key is" f"missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        B, N_q, C = query.shape
        N_k = key.size(1)
        N_v = value.size(1)
        # (B, num_heads, N, C)
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_v, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #
        attn = (q @ k.transpose(-2, -1)) * self.scale
        #
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1).repeat(1, attn.size(1), 1, 1)
            attn = attn.masked_fill(attn_mask==0, -1e4)
        attn = attn.softmax(dim=-1)
        attn_vis = attn.clone()
        #
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.out_proj(x)
        
        if self.add_identity:
            out = {'query': identity + x,
                   'attn': attn_vis}
        else:
            out = {'query': x,
                   'attn': attn_vis}
        return out

class TransformerLayerSequence(nn.Module):
    """Base class for TransformerEncoder and TransformerDecoder, which will copy
    the passed `transformer_layers` module `num_layers` time or save the passed
    list of `transformer_layers` as parameters named ``self.layers``
    which is the type of ``nn.ModuleList``.
    The users should inherit `TransformerLayerSequence` and implemente their
    own forward function.

    Args:
        transformer_layers (list[BaseTransformerLayer] | BaseTransformerLayer): A list
            of BaseTransformerLayer. If it is obj:`BaseTransformerLayer`, it
            would be repeated `num_layers` times to a list[BaseTransformerLayer]
        num_layers (int): The number of `TransformerLayer`. Default: None.
    """

    def __init__(
        self,
        transformer_layers=None,
        num_layers=None,
        first_layer=None,
        last_layer=None,
    ):
        super(TransformerLayerSequence, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if first_layer is not None:
            self.layers.append(first_layer)
            num_layers -= 1
        if last_layer is not None:
            num_layers -= 1
        if isinstance(transformer_layers, nn.Module):
            for _ in range(num_layers):
                self.layers.append(copy.deepcopy(transformer_layers))
        elif transformer_layers is None:
            pass
        else:
            assert isinstance(transformer_layers, list) and len(transformer_layers) == num_layers
        if last_layer is not None:
            self.layers.append(last_layer)

    def forward(self):
        """Forward function of `TransformerLayerSequence`. The users should inherit
        `TransformerLayerSequence` and implemente their own forward function.
        """
        raise NotImplementedError()
