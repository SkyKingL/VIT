"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training: 
        return x
    keep_prob = 1 - drop_prob # 保留的概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适用于不同维度张量，而不仅仅是2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 向下取整
    output = x.div(keep_prob) * random_tensor # 除以保留概率，然后乘以随机张量
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """
    2D图像转换为Patch Embedding
    参数：
        img_size (int, tuple): 输入图像的大小
        patch_size (int, tuple): patch的大小
        in_c (int): 输入图像的通道数，RGB图像为3
        embed_dim (int): 输出的embedding维度，在ViT-B/16中为768
        norm_layer: (nn.Module): normalization layer
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1] # num_patches = 14 * 14 = 196

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size) # kernel_size就是patch_size，stride也是patch_size
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() # Identity()是一个空层,即不做任何操作

    def forward(self, x):
        B, C, H, W = x.shape
        # 如果输入图像的大小不是patch_size的整数倍，那么就会报错。在vit中，输入大小必须是固定的
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2) # 展平处理从第2维开始，再将第1维和第2维交换位置
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """
    参数：
        dim (int): 输入token的dim
        num_heads (int): 多头注意力的头数
        qkv_bias (bool): enable bias for qkv if True，即是否对qkv进行偏置
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set，即缩放因子
        attn_drop_ratio (float): attention dropout rate，即attention层的dropout
        proj_drop_ratio (float): projection dropout rate，即全连接层的dropout
    """
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads # 每个head的维度，即embed_dim / num_heads
        self.scale = qk_scale or head_dim ** -0.5 # 如果qk_scale 为 None，则使用 head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # qkv层，Liner全连接层
        self.attn_drop = nn.Dropout(attn_drop_ratio) # attention层的dropout
        self.proj = nn.Linear(dim, dim) # 全连接层Wo, 用于将多头注意力的输出映射到原始维度
        self.proj_drop = nn.Dropout(proj_drop_ratio) # 全连接层的dropout

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # num_patches + 1: num_patches是输入的patch数，+1是因为cls_token。196+1=197
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]，即[batch_size, 197, 3 * 768]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]，即[batch_size, 197, 3, 12, 64]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]，即[3, batch_size, 12, 197, 64]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)。通过切片获取qkv

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale # q和k的转置相乘，得到注意力矩阵。transpose(-2, -1)表示交换倒数第2维和倒数第1维。@表示矩阵乘法，针对最后两维
        attn = attn.softmax(dim=-1) # softmax归一化。对dim的理解:https://blog.csdn.net/qq_51206550/article/details/132278743
        attn = self.attn_drop(attn) # dropout

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # attn和v相乘，得到注意力矩阵。transpose(1, 2)表示交换第1维和第2维。reshape(B, N, C)表示将最后两维合并
        x = self.proj(x) # 全连接层Wo
        x = self.proj_drop(x) # dropout
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    参数：
        in_features (int): 输入特征的维度
        hidden_features (int): 隐藏层特征的维度
        out_features (int): 输出特征的维度
        act_layer: (nn.Module): activation layer
        drop (float): dropout rate
    """
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


class Block(nn.Module):
    """
    Transformer block
    参数：
        dim (int): 输入token的dim
        num_heads (int): 多头注意力的头数
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True，即是否对qkv进行偏置
        qk_scale (float): override default qk scale of head_dim ** -0.5 if set，即缩放因子
        drop_ratio (float): dropout rate
        attn_drop_ratio (float): attention dropout rate，即attention层的dropout
        drop_path_ratio (float): stochastic depth rate，即drop path的dropout
        act_layer: (nn.Module): activation layer
        norm_layer: (nn.Module): normalization layer
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim) # 第一个LayerNorm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio) # Multi-head Attention
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity() # 如果drop_path_ratio > 0.，则使用DropPath(drop_path_ratio)，否则使用nn.Identity()
        self.norm2 = norm_layer(dim) # 第二个LayerNorm
        mlp_hidden_dim = int(dim * mlp_ratio) # mlp的隐藏层维度
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) # 先norm1，再attention，再drop_path
        x = x + self.drop_path(self.mlp(self.norm2(x))) # 先norm2，再mlp，再drop_path
        return x

class VisionTransformer(nn.Module):
    
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size 输入图像的大小
            patch_size (int, tuple): patch size 输入patch的大小
            in_c (int): number of input channels in_c为输入图像的通道数，RGB图像为3
            num_classes (int): number of classes for classification head 分类头的类别数,数据集目标分类的总数
            embed_dim (int): embedding dimension 嵌入维度
            depth (int): depth of transformer blocks ——***在transformer encoder中重复encoder block 的次数***
            num_heads (int): number of attention heads 注意力头的数量
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim mlp隐藏维度与嵌入维度的比率
            qkv_bias (bool): enable bias for qkv if True 如果为True，则为qkv启用偏置
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set 如果设置，则覆盖head_dim ** -0.5的默认qk比例
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set 指的是在Pre-logits中全连接层的节点个数，如果为none，则不使用Pre-logit
            distilled (bool): model includes a distillation token and head as in DeiT models 如果为True，则模型包含DeiT模型中的蒸馏令牌和头（针对vit的话不用管）
            drop_ratio (float): dropout rate 丢失率
            attn_drop_ratio (float): attention dropout rate 注意力丢失率
            drop_path_ratio (float): stochastic depth rate 随机深度率
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) # 默认归一化层为LayerNorm
        act_layer = act_layer or nn.GELU # 默认激活函数为GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # cls_token是一个可学习的参数，维度为[1, 1, 768]，第一个1表示batch_size为了方便concat拼接，第二个1表示num_tokens，第三个1表示embed_dim
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None # vit里不管这行
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # 位置编码，维度为[1, 197, 768]
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule 随机深度衰减规则
        # 12个Block，每个Block的输入维度为768，输出维度为768，注意力头数为12，mlp隐藏维度为3072，激活函数为GELU，LayerNorm为归一化层，丢失率为0.1，注意力丢失率为0.0，随机深度衰减率为dpr[i]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim) # 最后一个LayerNorm

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh()) 
            ])) # Linear全连接层，输入维度为768，输出维度为768，激活函数为Tanh
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s) 分类头，即全连接层
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init 初始化权重，这里使用了trunc_normal_，即截断正态分布
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768] 
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # 对cls_token进行expand扩展，使其与x的第1维相同，方便concat拼接
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768], concat拼接
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed) # 与位置编码相加，再dropout
        x = self.blocks(x) # 12个Block
        x = self.norm(x) # 最后一个LayerNorm
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]) # x[:, 0]表示取所有batch数据的第0个token，即cls_token
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None: # vit里不管这行
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x) # 分类头，即全连接层得到输出
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

# vit只有在非常大的数据集上预训练之后才有比较好的效果，所以不建议直接训练自己的模型，而是使用下面的权重做迁移学习

def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16, # patch_size越大，patch embedding后的维度越小，计算量越小
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
