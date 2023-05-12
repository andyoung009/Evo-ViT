""" Evo-ViT in PyTorch

A PyTorch implement of Evo-ViT as described in
'Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer'

The code is modified from DeiT as described in
'DeiT: Data-efficient Image Transformers' - https://arxiv.org/abs/2012.12877
The official code of DeiT is released and available at https://github.com/facebookresearch/deit
"""
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

# 这段代码是用于创建一个名为__name__的logger对象，用于记录日志信息。__name__是一个特殊的Python变量，表示当前模块的名称。logging.getLogger()函数会返回一个logger对象，
# 可以用于记录日志信息。在这里，使用__name__作为logger的名称，可以方便地跟踪日志信息的来源。例如，如果在其他模块中调用了这个logger对象，就可以通过logger的名称来确定日志信息的来源。
_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


def easy_gather(x, indices):
    # x: B,N,C; indices: B,N
    # 这里获取输入张`x`的形状，其中`B`表示batch size，`N`表示数据点数，`C`表示特征数。
    B, N, C = x.shape
    # 这里获取索引张量`indices`的形状，其中`N_new`表示新的数据点数。
    N_new = indices.shape[1]
    # 这里创建一个偏移量张量`offset`，用于将每个batch的数据点索映射到整个数据集中的唯一索引。
    # 具体来说，偏移量张量的形状为`(B, 1)`，其中每个元素为`i*N`，其中`i`表示batch的索引。
    # 这样，每个batch的数据点索引就可以通过加上对应的偏移量来映射到整个数据集中的唯一索引。
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    # 这里将偏移量张量加到索引张量上，得到映射到整个数据集中的唯一索引。
    indices = indices + offset
    # 这里将输入张量`x`展平成一个二维张量，然后根据索引张量`indices`获取对应的数据点，最后将结果重新变形成三维张量。
    # 具体来说，首先将输入张量`x`变形成一个二维张量，形状为`(B*N, C)`，然后使用`indices.view(-1)`将索引张量展平成一维张量
    # 然后使用`[]`运算符获取对应的数据点，最后使用`reshape`函数将结果变形成三维张量，形为`(B, N_new, C)`。
    out = x.reshape(B * N, C)[indices.view(-1)].reshape(B, N_new, C)
    return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class QKAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., input_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        if input_dim is not None:
            self.qk = nn.Linear(input_dim, dim * 2, bias=qkv_bias)
        else:
            self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn


class VAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.v = nn.Linear(dim, dim * 1, bias=qkv_bias)

    def forward(self, x):
        B, N, C = x.shape
        v = self.v(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  #v.shape = (1,B,H,N,C//H)
        v = v[0]  # make torchscript happy (cannot use tensor as tuple) v[0].shape = (B,H,N,C//H)

        return v


# original mlp block
class MlpBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp=None, return_raw=False):
        super().__init__()
        self.return_raw = return_raw
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mlp is not None:
            self.mlp = mlp
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if not self.return_raw:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        else:
            raw = self.drop_path(self.mlp(self.norm2(x)))
            x = x + raw
            return x, raw


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                # 这段代码是在获取骨干网络的最后一个特征图的通道数。`feature_info`是一个属性，它返回一个`FeatureInfo`对象，
                # 该对象包含有关骨干网络输出特征图的信息，例如通道数、步幅等。`channels()`方法返回一个列表，其中包含每个特征图的通道数。
                # `[-1]`表示获取列表中的最后一个元素，即最后一个特征图的通道数。
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2) #(B,N,C)
        return x


class EvoVisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, momentum=0.,
                 tradeoff=0.5, prune_ratio=None, prunecls=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.vis_dict = {}
        if prune_ratio is None:
            self.prune_ratio = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        else:
            self.prune_ratio = prune_ratio
        self.stage_prune_ratio = [self.prune_ratio[4], self.prune_ratio[8]]
        self.momentum = momentum
        if not isinstance(tradeoff, list):
            self.tradeoff = [tradeoff for i in range(depth)]
        else:
            self.tradeoff = tradeoff

        self.stage_wise_prune = False
        self.prunecls = prunecls

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            MlpBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(12)])

        self.qks = nn.ModuleList([QKAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop_rate, proj_drop=drop_rate) for i in range(12)])

        self.vs = nn.ModuleList([
            VAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       attn_drop=attn_drop_rate, proj_drop=drop_rate) for i in range(12)])

        self.norms = nn.ModuleList([norm_layer(embed_dim) for i in range(12)])
        self.projs = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for i in range(12)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # not used

        self.head_cls = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.norm_cls = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    # 定义了一个名为_init_weights的函数，用于初始化模型的权重,接受一个模块m作为参数。
    def _init_weights(self, m):
        # 判断m是否是nn.Linear类型的模块，如果是，则使用截断正态分布初始化权重，标准差为0.02。
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # 如果模块包含偏置项，则将偏置项初始化为0。
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 判断m是否是nn.LayerNorm类型的模块，如果是，则将偏置项初始化为0，将权重初始化为1。这是LayerNorm的常用初始化方法。
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # @torch.jit.ignore是一个装饰器，用于指示Torch JIT编译器忽略这个函数，因为它不需要被编译。
    @torch.jit.ignore
    # 这里返回一个集合，其中包含不应该被权重衰减的参数名称。
    # 在Vision Transformer模型中，位置嵌入（pos_embed）和分类令牌（cls_token）通常不会被权重衰减，因为它们对模型的性能影响较小。
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def get_vis_dict(self):
        return self.vis_dict

    def forward_features(self, x):
        # print(self.stage_wise_prune)

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        cls_attn = 0

        prune_ratio = self.prune_ratio

        # ori_index = torch.arange(196, device=x.device).unsqueeze(0) # 196 = (224/16) ** 2,应该改成下面所示代码
        ori_index = torch.arange(196, device=x.device).unsqueeze(0).repeat(B, 1)
        if self.stage_wise_prune:
            stage_index = -1
            for index, blk in enumerate(self.blocks):
                # token selection,当index == 4 or index == 8时进行了tokens的稀疏化采样,选取attn前N_个对应的token保存到了self.vis_dict
                if index == 4 or index == 8:
                    stage_index += 1
                    # 因为index=4是第一剪枝,不用补全X,index=8是则在前一个阶段补全了x,所以此处的剪枝仍然是基于总的tokens做剪枝的
                    x_patch = x[:, 1:, :]
                    B, N, C = x_patch.shape
                    N_ = int(N * self.stage_prune_ratio[stage_index])
                    cls_attn, indices = torch.sort(cls_attn, dim=1, descending=True)    #cls_attn, indices维度信息均为(B,N)
                    x_patch = easy_gather(x_patch, indices)

                    # 这2行代码将`ori_index`按照`indices`的顺序进行重新排序，然后将排序后的结果存储在`ori_index`中。
                    # 接着，代码将排序后的`ori_index`的前`N_`个元素存储在`self.vis_dict`字典中，键为`'block{}'.format(index)`，其中`index`是当前块的索引。
                    # 最后，代码将存储在`self.vis_dict`字典中的元素转换为列表并返回。这段代码的作用是将输入序列中的标记按照其在注意力分数中的重要性进行排序，
                    # 并将排序后的结果存储在`self.vis_dict`字典中以便于可视化和分析。
                    ori_index = easy_gather(ori_index.unsqueeze(-1), indices).squeeze(-1)
                    self.vis_dict['block{}'.format(index)] = ori_index.squeeze(0)[:N_].tolist()

                    if self.training:
                        # x_为根据cls_attn有序排列的
                        x_ = torch.cat((x[:, :1, :], x_patch), dim=1)
                    else:
                        x[:, 1:, :] = x_patch
                        x_ = x
                    x = x_[:, :N_ + 1]

                # normal updating
                if index < 4:
                    tmp_x = x
                    B, N, C = x.shape
                    x = self.norms[index](x)
                    v = self.vs[index](x)
                    attn = self.qks[index](x)
                    # with torch.no_grad():
                    if index == 0:
                        cls_attn = torch.sum(attn[:, :, 0, 1:], dim=1)
                    else:
                        # 对应文章中公式4,其中系数α对应的是(1 - self.tradeoff[index]),cls_attn为全局分类注意力,
                        # torch.sum(attn[:, :, 0, 1:], dim=1)为局部(第index+1层)分类注意力
                        cls_attn = (1 - self.tradeoff[index]) * cls_attn + self.tradeoff[index] * torch.sum(
                            attn[:, :, 0, 1:], dim=1)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.projs[index](x)
                    x = blk.drop_path(x)
                    x = x + tmp_x

                    x = blk(x)
                else:
                    # slow updating
                    tmp_x = x
                    B, N, C = x.shape
                    x = self.norms[index](x)
                    v = self.vs[index](x)
                    attn = self.qks[index](x)
                    # with torch.no_grad():
                    if self.training:
                        # 对应文章中公式4,其中系数α对应的是(1 - self.tradeoff[index]),cls_attn为全局分类注意力,
                        # torch.sum(attn[:, :, 0, 1:], dim=1)为局部(第index+1层)分类注意力
                        temp_cls_attn = (1 - self.tradeoff[index]) * cls_attn[:, :N_] + self.tradeoff[
                            index] * torch.sum(
                            attn[:, :, 0, 1:],
                            dim=1)
                        cls_attn = torch.cat((temp_cls_attn, cls_attn[:, N_:]), dim=1)

                    else:
                        cls_attn[:, :N_] = (1 - self.tradeoff[index]) * cls_attn[:, :N_] + self.tradeoff[
                            index] * torch.sum(
                            attn[:, :, 0, 1:],
                            dim=1)

                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.projs[index](x)
                    x = blk.drop_path(x)
                    x = x + tmp_x

                    x = blk(x)

                # fast updating, only preserving the placeholder tokens presents enough good results on DeiT
                # without representative tokens
                if self.prunecls:
                    if index == 7:
                        x_[:, :N_ + 1] = x
                        x = x_
                else:
                    if index == 7 or index == 11:
                        # 当训练时,在第8层和输出前(第11层)要把cls_attn比较低的token也补上,因为x_在推理阶段是全token
                        if self.training:
                            x = torch.cat((x, x_[:, N_ + 1:]), dim=1)
                        # 如果是推理,则把x_的前N_ + 1个token更新,其余不变,x_仍然是全token
                        else:
                            x_[:, :N_ + 1] = x
                            x = x_
        else:
            for index, blk in enumerate(self.blocks):
                if prune_ratio[index] != 1:
                    # token selection
                    x_patch = x[:, 1:, :]

                    B, N, C = x_patch.shape
                    N_ = int(N * prune_ratio[index])
                    indices = torch.argsort(cls_attn, dim=1, descending=True)
                    # 这行代码将`cls_attn`张量按照最后一个维度进行扩展，并将其与`x_patch`张量按照后一个维度进行连接。
                    # 具体来说，`cls_attn.unsqueeze(-1)`将`cls_attn`张量的维度从`(B, N)`扩展为`(B, N, 1)`，
                    # 然后将其与`x_patch`张量按照最后一个维度进行连接，即将`cls_attn`张量的最后一个维度与`x_patch`张量的最后维度进行连接，
                    # 得到一个新的张量`x_patch`。新的张量`x_patch`的形状为`(B, N, C+1)`，其中`C`是`x_patch`张量的最后一个维度的大小，
                    # `1`是`cls_attn`张量的最后一个维度的大小。这行代码的作用是将`cls_attn`张量添加到`x_patch`张量的最后一个维度中，以便在后续的计算中使用。
                    x_patch = torch.cat((x_patch, cls_attn.unsqueeze(-1)), dim=-1)
                    x_sorted = easy_gather(x_patch, indices)
                    # 从最后一个维度将2者拆分
                    x_patch, cls_attn = x_sorted[:, :, :-1], x_sorted[:, :, -1]

                    # 如果ori_index张量的形状为(1, 196)，则在进行easy_gather操作之前，需要将其转换为(B, N)的形状?见L360修改
                    ori_index = easy_gather(ori_index.unsqueeze(-1), indices).squeeze(-1)
                    # 因为将ori_index重复了B次,导致第一个维度上为B,此时squeeze(0)实际上已经不起删除维度1的作用了
                    self.vis_dict['block{}'.format(index)] = ori_index.squeeze(0)[:N_].tolist()

                    if self.training:
                        x_ = torch.cat((x[:, :1, :], x_patch), dim=1)
                    else:
                        x[:, 1:, :] = x_patch
                        x_ = x
                    # 截取了前topN_个cls_attn最大的tokens及cls-token,共N_ + 1个,这么做其实导致上面的条件分支失去了意义,不管推理还是训练,处理方式都是一样的
                    x = x_[:, :N_ + 1]

                    # slow updating
                    # 计算挑选出来的topN_个tokens的自注意力机制
                    tmp_x = x
                    B, N, C = x.shape
                    x = self.norms[index](x)
                    v = self.vs[index](x)
                    attn = self.qks[index](x)

                    # with torch.no_grad():
                    if self.training:
                        # 计算cls_attn时没有cls-token,只计算了patches-token的attn
                        temp_cls_attn = (1 - self.tradeoff[index]) * cls_attn[:, :N_] + self.tradeoff[
                            index] * torch.sum(
                            attn[:, :, 0, 1:],
                            dim=1)
                        # 训练过程中cls_attn的维度实际上没有降低,未更新的cls_token项直接照搬了
                        cls_attn = torch.cat((temp_cls_attn, cls_attn[:, N_:]), dim=1)

                    else:
                        # 推理过程只更新cls_attn的topN_个,利用文章中的公式(4)做更新
                        cls_attn[:, :N_] = (1 - self.tradeoff[index]) * cls_attn[:, :N_] + self.tradeoff[
                            index] * torch.sum(
                            attn[:, :, 0, 1:],
                            dim=1)

                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.projs[index](x)
                    x = blk.drop_path(x)
                    x = x + tmp_x

                    x = blk(x)

                    # fast updating, only preserving the placeholder tokens presents enough good results on DeiT
                    # 此处代码实现和论文中其实是有区别的,可以参考代码仓库的issue3,见这里https://github.com/YifanXu74/Evo-ViT/issues/3
                    if self.prunecls and index == 11:
                        pass
                    else:
                        # 训练时直接把所有token都拼接,包括cls_attn比较低的
                        if self.training:
                            x = torch.cat((x, x_[:, N_ + 1:]), dim=1)
                        # 推理时直接把其余的token扔掉了,确实论文中方法和代码不符合
                        else:
                            x_[:, :N_ + 1] = x
                            x = x_

                # normal updating in the beginning four layers,prune-ratio为1,不做剪枝
                else:
                    tmp_x = x
                    B, N, C = x.shape
                    x = self.norms[index](x)
                    v = self.vs[index](x)
                    attn = self.qks[index](x)

                    if index == 0:
                        cls_attn = torch.sum(attn[:, :, 0, 1:], dim=1)
                    else:
                        cls_attn = (1 - self.tradeoff[index]) * cls_attn + self.tradeoff[index] * torch.sum(
                            attn[:, :, 0, 1:], dim=1)
                    # 自注意力层计算
                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.projs[index](x)
                    x = blk.drop_path(x)
                    x = x + tmp_x

                    # 自注意力层和线性层分开了,前面因为快慢更新,自注意力层没有办法模块化,要利用中间的计算结果attn
                    # blk模块为mlp线性层,Feed Forward模块
                    x = blk(x)

        cls = x[:, 0]
        x = x[:, 1:]
        x = x.mean(1)
        x = self.norm(x)
        if self.training:
            cls = self.norm_cls(cls)
        return x, cls

    def forward(self, x):
        x, cls = self.forward_features(x)
        if not self.training:
            x = self.head(x)
        else:
            x = self.head(x), self.head_cls(cls)
        return x


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


@register_model
def evo_deit_tiny_vis_patch16_224(pretrained=False, **kwargs):
    # drop_path = 0
    tradeoff = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    prune_ratio = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    model = EvoVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), tradeoff=tradeoff, prune_ratio=prune_ratio, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def evo_deit_small_vis_patch16_224(pretrained=False, **kwargs):
    tradeoff = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    prune_ratio = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    model = EvoVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), tradeoff=tradeoff, prune_ratio=prune_ratio, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def evo_deit_base_vis_patch16_224(pretrained=False, **kwargs):
    tradeoff = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    prune_ratio = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    model = EvoVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), tradeoff=tradeoff, prune_ratio=prune_ratio, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def evo_deit_small_vis_patch16_384(pretrained=False, **kwargs):
    tradeoff = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    prune_ratio = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    model = EvoVisionTransformer(
        img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), tradeoff=tradeoff, prune_ratio=prune_ratio, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def evo_deit_base_vis_patch16_384(pretrained=False, **kwargs):
    tradeoff = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    prune_ratio = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    model = EvoVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), tradeoff=tradeoff, prune_ratio=prune_ratio, **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    model = evo_deit_tiny_vis_patch16_224()
    print('Parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    # print(y[1].shape)
    print(y)
