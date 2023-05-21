# ------------------------------------------------------------------------
# Mostly a modified copy from timm
# (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
# ------------------------------------------------------------------------
import time

import numpy as np
import torch
from timm.models.layers import DropPath
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as F
# from .cuda import gather_tokens, scatter_tokens


class AfterReconstruction(nn.Identity):
    def __init__(self, in_planes):
        super().__init__()
        self.in_planes = in_planes

# 这段代码定义了一个继承自`nn.Linear`的新类`ALinear_CUDA`，用于在PyTorch中实现自定义的线性层。
class ALinear_CUDA(nn.Linear):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 该类的构造函数初始化了模型权重的形状，并定义了一个`out_zero_mask`参数，用于将线性层的输出值与输入值的形状匹配。
        out_channels = self.weight.shape[0]
        self.out_zero_mask = nn.Parameter(
            torch.zeros(1, out_channels, 1), requires_grad=False
        )

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        """

        :param input:
        :param mask:
        :return:
        """

        if mask is None:
            # 该类的`forward`方法实现了自定义的线性转换操作。在没有提供掩码（`mask`）的情况下，该方法将输入值通过标准的线性层转换，然后返回转换后的结果。
            return F.linear(input, self.weight, self.bias)

        # 如果提供了掩码，则将输入值转置并重新整形，以便在掩码指定的位置上进行线性转换。
        # 然后，通过将线性层的输出值与`out_zero_mask`合并，将转换后的值放回输入值的形状中，最终返回转换后的结果。
        input = input.transpose(-2, -1).contiguous()
        B, D, N = input.shape

        # For Debug Only
        # mask[:, 10:110] = 0.0

        # 一些用于计时和调试的代码，其中包括将时间数据打印到控制台上以进行调试和优化。
        start = []

        start.append(time.time())
        active_position_list = torch.nonzero(mask.flatten()).squeeze(1).int()

        start.append(time.time())
        sampled_tokens = gather_tokens(input, active_position_list)

        start.append(time.time())
        sampled_tokens = sampled_tokens.transpose(-2, -1).squeeze().contiguous()
        # For Debug Only
        # print("{} / {}".format(sampled_tokens.shape[0], B*N))
        start.append(time.time())
        sampled_tokens = F.linear(sampled_tokens, self.weight, self.bias)

        start.append(time.time())
        # out_zero_mask = torch.zeros(flatten_input.shape[0], D_out).to(out.device)
        out_zero_mask = self.out_zero_mask.expand(B, -1, N).contiguous()

        start.append(time.time())
        # 这行代码中调用了一个名为`scatter_tokens`的函数，它的作用是将一个张量`sampled_tokens`中的值按照给定的索引（`active_position_list`）分散到另一个张量`out_zero_mask`中。
        # 具体来说，`active_position_list`是一个一维整数张量，它包含了`sampled_tokens`中需要分散的位置的索引。`out_zero_mask`是一个与`sampled_tokens`形状相同的张量，初始值为全零。
        # `scatter_tokens`函数会将`sampled_tokens`中指定位置的值按照索引分散到`out_zero_mask`中对应位置上，并返回分散后的张量。
        # 这种分散操作可以用于在一个张量中填充指定位置的值，从而实现对稀疏张量的操作。在这个代码片段中，`scatter_tokens`函数的作用是将线性层的输出值按照掩码指定的位置分散到输入值的形状中，并返回转换后的结果。
        out = scatter_tokens(sampled_tokens, out_zero_mask, active_position_list)

        start.append(time.time())
        out = out.transpose(-2, -1)

        start.append(time.time())

        start_gt = time.time()
        gt = F.linear(input.transpose(-2, -1).contiguous(), self.weight, self.bias)
        end_gt = time.time()

        timing = [(start[i] - start[i - 1]) for i in range(1, len(start))]

        total_time = sum(timing)

        print(
            "Linear Layer Timing: Adaptive Linear: {}  Linear: {}  Diff: {}".format(
                total_time, end_gt - start_gt, total_time - end_gt + start_gt
            )
        )

        timing_per = [
            str((start[i] - start[i - 1]) / total_time * 100)
            for i in range(1, len(start))
        ]
        str_timing = " | ".join(timing_per)
        print("Timing Details (%): " + str_timing)
        str_timing = " | ".join([str(t) for t in timing])
        print("Timing Details (s): " + str_timing)
        return out


class ALinear_PyTorch(nn.Linear):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        out_channels = self.weight.shape[0]
        self.out_zero_mask = nn.Parameter(
            torch.zeros(1, out_channels), requires_grad=False
        )

    def forward(
        self, input: Tensor, mask: Tensor = None, sampler: Tensor = None
    ) -> Tensor:
        """

        :param input:
        :param mask:
        :param sampler:
        :return:
        """
        if mask is None:
            return F.linear(input, self.weight, self.bias)
        B, N, D = input.shape
        # timing = []
        D_out = self.weight.shape[0]

        # timing.append(time.time())
        # sampler = torch.nonzero(mask)
        out_mask_size = mask.sum(1).max().int()
        # if out_mask_size < 197:
        #    print("")
        #    pass
        # timing.append(time.time())
        sampler_out = sampler[:, 0] * out_mask_size + sampler[:, 1]
        sampler = sampler[:, 0] * N + sampler[:, 1]
        sampler_input = sampler.unsqueeze(-1).expand(-1, D)
        sampler_output = sampler_out.unsqueeze(-1).expand(-1, D_out)
        flatten_input = input.reshape(-1, D)

        # timing.append(time.time())
        sampled_input = torch.gather(flatten_input, 0, sampler_input)

        # timing.append(time.time())
        out = F.linear(sampled_input, self.weight, self.bias)

        # timing.append(time.time())
        out_zero_mask = self.out_zero_mask.expand(B * out_mask_size, -1)

        # timing.append(time.time())
        out = out_zero_mask.scatter(0, sampler_output, out, reduce="add").reshape(
            (B, out_mask_size, D_out)
        )
        policy = (
            out_zero_mask[:, 0]
            .scatter(0, sampler_out, 1, reduce="add")
            .reshape(B, out_mask_size, 1)
        )
        # timing.append(time.time())

        # gt_start = time.time()
        # gt = F.linear(input, self.weight, self.bias)
        # gt_end = time.time()

        # timing = [
        #    (timing[i] - timing[i - 1])
        #    for i in range(1, len(timing))
        # ]

        # total_time = sum(timing)

        # print(
        #    "Linear Layer Timing: Adaptive Linear: {}  Linear: {}  Diff: {}".format(
        #        total_time, gt_end - gt_start, total_time - gt_end + gt_start
        #    )
        # )

        # timing_per = [
        #    str(timing[i] / total_time * 100)
        #    for i in range(0, len(timing))
        # ]
        # str_timing = " | ".join(timing_per)
        # print("Timing Details (%): " + str_timing)
        # str_timing = " | ".join([str(t) for t in timing])
        # print("Timing Details (s): " + str_timing)

        return out, policy


# 这是一个名为`ALinear_Sparse`的类，它是`nn.Linear`的子类，用于实现稀疏矩阵的线性变换。类中重载了`nn.Linear`的`forward`方法，实现了输入稀疏矩阵的线性变换，并返回变换后的稠密张量。
# 具体来说，`forward`方法接收三个参数：输入张量`input`、稀疏掩码张量`mask`和一个占位符`_`。其中，`input`是一个形状为`(B,N,D)`的三维张量，表示输入的批次大小、序列长度和特征维度。`mask`是一个与`input`形状相同的稀疏张量，其中非零元素表示需要进行线性变换的位置。`_`是一个占位符，表示该方法不使用该参数。
# 在函数中，首先获取输入张量的形状，并将其展平为二维张量`finput`。然后，将`finput`转换为稀疏张量`sp`，使用稀疏矩阵的乘法方法`matmul`实现稀疏矩阵与权重矩阵的乘积，并加上偏置向量。最后，将结果重新变形为原始输入张量的形状，并返回结果张量`out`。
# 需要注意的是，这个实现中并没有使用掩码张量`mask`来过滤输入张量中的非零元素，而是直接将输入张量展平为稀疏张量进行计算。因此，在实际使用中，需要确保输入张量中非零元素的位置与掩码张量中的位置相同，以保证正确的线性变换结果。
class ALinear_Sparse(nn.Linear):
    def forward(self, input: Tensor, mask: Tensor, _) -> Tensor:
        B, N, D = input.shape
        # finput = (input * mask).flatten(0, 1)
        finput = input.flatten(0, 1)
        sp = finput.to_sparse_csr()
        out = sp.matmul(self.weight.transpose(-2, -1)) + self.bias.unsqueeze(0)
        out = out.reshape(B, N, -1)
        return out


# 这是一个名为`ALinear`的类，它是`nn.Linear`的子类，用于实现普通的线性变换。类中重载了`nn.Linear`的`forward`方法，
# 但是在实现中直接调用了父类的`forward`方法，没有对输入张量进行任何额外的处理。因此，`ALinear`类的实现与`nn.Linear`类的实现完全一致，没有任何区别。
# 在实际应用中，如果需要实现稀疏矩阵的线性变换，应该使用`ALinear_Sparse`类实现，而不是`ALinear`类。如果需要实现普通的线性变换，可以直接使用`nn.Linear`类，
# 或者继承`nn.Linear`类并进行必要的重载实现。因此，`ALinear`类的实现并没有实际的意义，只是作为一个示例代码展示了如何继承和重载PyTorch中的神经网络模块类。
class ALinear(nn.Linear):
    def forward(self, input: Tensor, mask: Tensor, _) -> Tensor:
        return super().forward(input)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ALinear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = ALinear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, policy: Tensor = None, sampler: Tensor = None) -> Tensor:
        x = self.fc1(x, policy, sampler)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, policy, sampler)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.qkv = ALinear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = ALinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.n_segment = 8

    # 这段代码实现了一种带有策略向量的 softmax 注意力机制。具体来说，给定一个注意力矩阵 `attn` 和一个策略向量 `policy`，该函数首先将策略向量广播为与注意力矩阵相同的形状，然后通过一个公式将注意力矩阵与策略向量相结合，从而得到一个加权的注意力矩阵。
    # 具体而言，该函数首先将策略向量 `policy` 从形状为 `(B, N, _)` 调整为形状为 `(B, 1, 1, N)`，其中 `B` 是批量大小，`N` 是注意力矩阵的大小。
    # 然后，该函数将单位矩阵 `eye` 调整为与 `policy` 广播兼容的形状 `(1, 1, N, N)`，并计算 `attn_policy + (1.0 - attn_policy) * eye`.
    # 接下来，该函数通过减去每行的最大值来提高计算的稳定性，然后使用指数函数将注意力矩阵的值转换为非负数，并将其与策略向量相乘。
    # 最后，该函数将注意力矩阵除以每行的总和，从而将其归一化为概率分布。最终输出的是归一化后的加权注意力矩阵，与输入的注意力矩阵 `attn` 具有相同的形状。
    @staticmethod
    def softmax_with_policy(attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(
            1, 1, N, N
        )
        # attn_policy的非主对角线元素不变,主对角线元素仍保持为1,此处的主对角线的元素是不是可以调整呢?比如让它小一点,例如'ViT for small datasets'中对主对角线元素做了掩码处理,同时调整了softmax的温度系数
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy, sampler):

        B, N, C = x.shape

        qkv= self.qkv(x, policy, sampler)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x, policy, sampler)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        insert_control_point=False,
    ):
        super().__init__()
        self.insert_control_point = insert_control_point
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, policy: Tensor = None, sampler: Tensor = None) -> Tensor:
        x = x + self.drop_path(
            self.attn(x=self.norm1(x), policy=policy, sampler=sampler)
        )
        if policy is not None:
            x = x * policy
        out = self.mlp(x=self.norm2(x), policy=policy, sampler=sampler)
        x = x + self.drop_path(out)
        if policy is not None:
            x = x * policy
        return x


def get_sinusoid_encoding(n_position, d_hid):
    """
    Sinusoid position encoding table
    """

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
