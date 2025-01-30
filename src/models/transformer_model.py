import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from src.data import utils
from src.frameworks import diffusion_utils
from src.models.layers import Xtoy, Etoy, masked_softmax


class XEyTransformerLayer(nn.Module):
    """Transformer that updates node, edge and global features
    d_x: node features
    d_e: edge features
    dz : global features
    n_head: the number of heads in the multi_head_attention
    dim_feedforward: the dimension of the feedforward network model after self-attention
    dropout: dropout probablility. 0 to disable
    layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        ds: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """Pass the input through the encoder layer.
        X: (bs, n, d)
        E: (bs, n, n, d)
        y: (bs, dy)
        node_mask: (bs, n) Mask for the src keys per batch (optional)
        Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """Self attention layer that also updates the representations on the edges."""

    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask  # (bs, n, dx)
        K = self.k(X) * x_mask  # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)  # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)  # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)  # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2  # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)  # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask  # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)  # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)  # bs, dy

        return newX, newE, new_y


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Projection matrices for queries, keys, and values
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        self.drop_p = dropout

    def forward(self, x, ctx, mask=None):
        batch_size, query_len, _ = x.size()
        _, key_len, _ = ctx.size()
        # Linear projections
        queries = self.query_proj(x)
        keys = self.key_proj(ctx)
        values = self.value_proj(ctx)

        # context = torch.nn.functional.scaled_dot_product_attention(
        #     queries,
        #     keys,
        #     values,
        #     dropout_p=self.drop_p if self.training else 0,
        #     is_causal=False,
        # )
        attn_weights = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training:
            attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, values)
        
        context = context.transpose(1, 2).reshape(batch_size, query_len, self.embed_dim)
        output = self.out_proj(context)
        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, embed_dim, dropout)

    def forward(self, x, ctx, mask=None):
        newx, attention_score = self.cross_attention(x, ctx, mask)
        x = self.norm1(x + newx)
        x = self.norm2(x + self.ffn(x))
        return x, attention_score


class MLP(nn.Module):
    def __init__(self, out_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, out_size)  # Expand feature from 2 to 64

    def forward(self, x):
        return F.relu(self.fc1(x))


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias=False, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

        self.q_layernorm = LayerNorm(n_embd // self.n_head)
        self.k_layernorm = LayerNorm(n_embd // self.n_head)

    def forward(self, x, attn_mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        # Use the newer scaled_dot_product_attention function
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        # Check for NaNs in the final output
        assert not torch.isnan(y).any(), "NaN detected in the final output"

        return y


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, embed_dim, dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attention(x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    def __init__(
        self,
        n_layers: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        act_fn_in: nn.ReLU(),
        act_fn_out: nn.ReLU(),
        addition=True,
        condition = True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]
        self.addition = addition
        self.condition = condition
        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )

        self.mlp_in_s = nn.Sequential(
            nn.Linear(input_dims["s"], hidden_mlp_dims["s"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["s"], hidden_dims["ds"]),
            act_fn_in,
        )

        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    ds=hidden_dims["ds"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                )
                for i in range(n_layers)
            ]
        )

        self.ctx_proj_node_layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dims["dx"], 4) for i in range(n_layers)]
        )
        self.ctx_proj_edge_layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dims["de"], 4) for i in range(n_layers)]
        )
        self.spectrum_embedding = nn.Linear(2, hidden_dims["ds"])
        self.spectrum_encoder = nn.ModuleList(
            [SelfAttentionBlock(hidden_dims["ds"], 4) for _ in range(4)]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
        )

        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )

        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
        )
        self.pad_embedding = nn.Parameter(torch.zeros(hidden_dims["ds"]))

    def forward(self, X, E, y, node_mask, s, s_mask):


        bs, n = X.shape[0], X.shape[1]
        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = utils.PlaceHolder(
            X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)
        ).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y
        
        
        ms = s.reshape(bs, -1, 2)
        
        ms = self.spectrum_embedding(ms)  # [b, m, 64]
        orig_s_mask = s_mask[..., 0].cuda()
        s_mask_int = orig_s_mask.int()

        s_mask = (
            ~s_mask_int.unsqueeze(2).logical_xor(s_mask_int.unsqueeze(1)).unsqueeze(1)
        )
        # Self Attention
        for layer in self.spectrum_encoder:
            ms = layer(ms, s_mask)  
                     
        orig_s_mask = orig_s_mask.unsqueeze(-1)
        
        # Learnable padding
        ms = ms * orig_s_mask + ~orig_s_mask * self.pad_embedding


        for node_layer, edge_layer, layer in zip(
            self.ctx_proj_node_layers, self.ctx_proj_edge_layers, self.tf_layers
        ):
            X, E, y = layer(X, E, y, node_mask)
            if self.condition:
                newX, attention = node_layer(X, ms) 
                X = X + newX # X:[bs, n, 64], s:[bs, m, 64]
                
                newE, attentionE = edge_layer(E.view(E.shape[0], -1, E.shape[-1]), ms)
                E = E + newE.view(
                    bs, n, n, -1
                )  # E: [bs,b, b, F]      
            
            
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        if self.addition:
            X = X + X_to_out
            E = E + E_to_out
            y = y + y_to_out
        
        
        E = E * diag_mask
        E = 1 / 2 * (E + torch.transpose(E, 1, 2))

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
