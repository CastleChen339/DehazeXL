import torch
import torch.hub
from einops import rearrange
from torch import nn
from ..context_encoders import ContextEncoderConfig
from .utils import get_2d_sincos_pos_embed, LlamaRMSNorm
from ..backbones import *


# default_decoder_filters = [48, 96, 176, 256]
# default_last = 48


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.pretraining_tp = 1
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LLMLayer(nn.Module):
    def __init__(
            self, dim, inner_dim, num_heads, causal=False, attention_method="hyper"
    ):
        super().__init__()
        # num_heads = dim // 128
        if attention_method == "hyper":
            from ..context_encoders.attention import LLMAttention
            self.attn = LLMAttention(dim, dim, num_heads, causal=causal)
            # from ..context_encoders.attentionmla import LLM_mlh_Attention
            # self.attn = LLM_mlh_Attention(dim, dim, num_heads, causal=causal)
        else:
            from ..context_encoders.attention import ViTAttention
            self.attn = ViTAttention(dim, dim, num_heads, causal=causal)
        self.input_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.post_attention_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.mlp = LlamaMLP(dim, inner_dim)
        self.causal = causal

    # def forward(self, hidden_states, residual_in=-1):
    def forward(self, hidden_states, residual_in=-1, kv_cache=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.attn(hidden_states)
        # hidden_states, kv_cache = self.attn(hidden_states, kv_cache)
        
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if residual_in != -1:
            return hidden_states, 0.0, kv_cache
        return hidden_states


class LLMBottleneck(nn.Module):
    def __init__(
            self,
            in_dim,
            mlp_ratio=4,
            hidden_size=768,
            num_heads=8,
            n_layers=2,
            attention_method=None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_size)
        assert attention_method in ["hyper", "naive"]
        self.layers = nn.Sequential(
            *[
                LLMLayer(
                    hidden_size,
                    hidden_size * mlp_ratio,
                    num_heads,
                    causal=False,
                    attention_method=attention_method,
                )
                for _ in range(n_layers)
            ]
        )

        self.hidden_size = hidden_size

    def forward(self, x):
        x = x[-1]
        n, _, h, w = x.shape
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, h, cls_token=False)
        x = rearrange(x, "n c h w -> n (h w) c")

        x = self.input_proj(x)
        x = x + torch.tensor(pos_embed).to(x)
        residual = None
        kv_cache = None  ###

        for i, blk in enumerate(self.layers):
            # x, residual = blk(x, residual)
            x, residual, kv_cache = blk(x, residual, kv_cache)
            if i == len(self.layers) - 1:
                x = (x + residual) if residual is not None else x
        x = rearrange(x, "n (h w) c -> n c h w", h=h, w=w)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        if isinstance(input_resolution, int):
            self.input_resolution = (input_resolution, input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution
        else:
            raise TypeError("input_resolution should be tuple or int.")
        self.dim = dim
        self.expand = nn.Linear(dim, int(dim_scale * dim // 4), bias=False)
        self.norm = norm_layer(int(dim // dim_scale))

    def forward(self, x):
        """
        x: B, H*W, C
        """
        h, w = self.input_resolution
        x = self.expand(x)
        b, l, c = x.shape
        assert l == h * w, "input feature has wrong size"

        x = x.view(b, h, w, c)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=c // 4)
        x = x.view(b, -1, c // 4)
        x = self.norm(x)

        return x


class PatchExpandConv(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4):
        super().__init__()
        if isinstance(input_resolution, int):
            self.input_resolution = (input_resolution, input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution
        else:
            raise TypeError("input_resolution should be tuple or int.")
        self.dim = dim
        self.expand = nn.ConvTranspose2d(dim, int(dim // dim_scale), kernel_size=2, stride=2)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        h, w = self.input_resolution
        x = rearrange(x, "n (h w) c -> n c h w", h=h, w=w)
        x = self.expand(x)
        x = rearrange(x, "n c h w -> n (h w) c")
        return x


class LayerUp(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, dim_scale=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if upsample is not None:
            # self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=dim_scale, norm_layer=norm_layer)
            self.upsample = PatchExpandConv(input_resolution, dim=dim, dim_scale=dim_scale)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class SwinDecoder(nn.Module):
    def __init__(self, in_resolution, in_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]):
        # 4 8 16 32 // 3 6 12 24
        super().__init__()
        self.num_layers = len(depths)
        self.crop_size = 256
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            up = LayerUp(dim=int(in_dim * 2 ** (self.num_layers - i_layer)), dim_scale=4,
                         input_resolution=(in_resolution // (2 ** (self.num_layers - 1 - i_layer)),
                                           in_resolution // (2 ** (self.num_layers - 1 - i_layer))),
                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                         depth=depths[(self.num_layers - 1 - i_layer)],
                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                         window_size=8,
                         upsample=True)
            self.layers_up.append(up)
        self.batch_size = 16
        self.up_x2 = PatchExpandConv(2 * in_resolution, in_dim // 2, dim_scale=4)
        self.output = nn.Conv2d(in_dim // 8, 3, 1, bias=False)

    def forward(self, x, fm_list, n_regions):
        x = rearrange(x, "N C (HP HC) (WP WC)-> (N HP WP) (HC WC) C", HP=n_regions, WP=n_regions)
        fm_list = [rearrange(i, "N C (HP HC) (WP WC)-> (N HP WP) (HC WC) C",
                             HP=n_regions, WP=n_regions) for i in fm_list]

        # for i in range(self.num_layers):
        #     x = self.layers_up[i](torch.cat([x, fm_list[self.num_layers - 1 - i]], dim=-1))

        n = x.shape[0]
        outputs = []
        for i in range(0, n, self.batch_size):
            end = min(i + self.batch_size, n)
            x_batch = x[i:end]
            fm_batch = [fm[i:end] for fm in fm_list]
            for j in range(self.num_layers):
                x_batch = self.layers_up[j](torch.cat([x_batch, fm_batch[self.num_layers - 1 - j]], dim=-1))
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        x = self.up_x2(x)
        x = rearrange(x, "n (HC WC) C -> n C HC WC", HC=self.crop_size, WC=self.crop_size)
        x = self.output(x)
        x = rearrange(x, "(N HP WP) C HC WC -> N C (HP HC) (WP WC)", HP=n_regions, WP=n_regions)
        return x


class DehazeXL(AbstractModel):
    def __init__(
            self,
            backbone: nn.Module = swinv2_tiny_window16_256_timm(input_size=256),
            xl_config: ContextEncoderConfig = ContextEncoderConfig,
            channels_last: bool = True,
            crop_size: int = 256,
            mlp_ratio: int = 4,
    ):
        self.channels_last = channels_last
        self.crop_size = crop_size
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.mlp_ratio = mlp_ratio
        self.xl_config = xl_config

        super().__init__()

        self.batch_size = 16
        self._initialize_weights()

        self.encoder = backbone
        self.bottleneck = LLMBottleneck(
            in_dim=self.filters[-1],
            mlp_ratio=self.mlp_ratio,
            hidden_size=self.xl_config.hidden_size,
            n_layers=self.xl_config.n_layer,
            attention_method=self.xl_config.attention_method,
        )
        self.decoder = SwinDecoder(in_resolution=crop_size // 4)

    def forward(self, x):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x_skip = x
        n_regions = x.shape[2] // self.crop_size

        if n_regions > 1:
            x = self.nested_tokenization(x)

            n = x.shape[0]
            outputs = []
            for i in range(0, n, self.batch_size):
                batch = x[i:min(i + self.batch_size, n)]
                output = self.encoder(batch)
                outputs.append(output)
            enc_results = [torch.cat([outputs[j][i] for j in range(len(outputs))], dim=0) for i in range(4)]

            # enc_results = self.encoder(x)

            enc_results = list(
                [
                    rearrange(
                        i,
                        "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
                        HP=n_regions,
                        WP=n_regions,
                    )
                    for i in enc_results
                ]
            )
        else:
            enc_results = self.encoder(x)

        output = self.bottleneck(enc_results)
        output = self.decoder(output, enc_results, n_regions)
        output += x_skip
        return output

    def nested_tokenization(self, x):
        n_regions = x.shape[2] // self.crop_size
        x = rearrange(
            x,
            "N C (HP HC) (WP WC)-> (N HP WP) C HC WC ",
            HP=n_regions,
            WP=n_regions,
            HC=self.crop_size,
            WC=self.crop_size,
        )
        return x
