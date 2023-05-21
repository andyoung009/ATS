import time

from torch import Tensor

# from libs.models.transformers.transformer_block import Attention, Mlp
from .transformer_block import Attention, Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class AdaptiveTokenSampler(Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        drop_tokens=False,
    ):
        super(AdaptiveTokenSampler, self).__init__(
            dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.out_zero_mask = nn.Parameter(torch.zeros(1, dim), requires_grad=False)
        self.drop_tokens = drop_tokens

    @staticmethod
    def get_unique_indices(indices: Tensor, max_value: int) -> Tensor:
        """
        :param indices: indices of the tokens to be sampled
        :param max_value: maximum number of the tokens to be sampled
        :return: unique indices of the tokens to be sampled
        """
        sorted_indices = torch.sort(indices, dim=1)[0]

        shift_left = F.pad(sorted_indices[:, 1:], (0, 1), value=1.0)
        unique_indices = torch.where(
            (shift_left - sorted_indices) == 0,
            max_value * torch.ones_like(indices),
            sorted_indices,
        )
        unique_indices = torch.sort(unique_indices, dim=1)[0]
        return unique_indices
# To explain the get_unique_indices method, it takes in two parameters: indices and max_value. 
# indices is a tensor containing the indices of the tokens to be sampled, # while max_value is the maximum number of tokens to be sampled.
# The method first sorts the indices in ascending order using torch.sort along the second dimension. 
# It then pads the sorted indices tensor with a 1.0 value at the end using F.pad. 
# The shift_left tensor is created by slicing the sorted indices tensor from the second column to the end and concatenating a tensor of 1.0 values at the beginning.
# The unique_indices tensor is created by using torch.where to compare the difference between shift_left and sorted_indices. 
# If the difference is 0, it means that the current index is a duplicate and should be replaced with max_value. 
# Otherwise, the current index is unique and should be kept as is. 
# The unique_indices tensor is then sorted in ascending order along the second dimension using torch.sort and returned.
# The reason why this method can sample unique values is that it first sorts the input indices tensor, which ensures that duplicate values are adjacent to each other. 
# It then compares each value with the value to its left in the shift_left tensor. If the difference is 0, it means that the current value is a duplicate and should be replaced with max_value. 
# Otherwise, the current value is unique and should be kept as is. This ensures that each duplicate value is replaced with a unique value, resulting in a tensor of unique indices.

# # Example usage
# indices = torch.tensor([[3, 1, 2], [2, 3, 1]])
# max_value = 4
# unique_indices = AdaptiveTokenSampler.get_unique_indices(indices, max_value)

# # output:
# tensor([[1, 2, 3],
#         [1, 2, 3]])

    # from chat-gpt
    # 这段代码是一个静态方法 `create_ys` 的实现，它接受两个参数：一个形状为 `(B, n_tokens)` 的张量 `normalized_cdf` 和一个整数 `n_tokens`。函数的返回值是一个形状为 `(B, n_tokens - 1)` 的张量 `ys`。
    # 具体来说，这个方法的目的是从一个经过归一化的累积分布函数（CDF）中均匀地采样一组 `n_tokens - 1` 个 y 坐标，用于构建一个离散的概率分布。其中，`B` 表示样本数，即 CDF 张量的第一维大小。
    # 函数的实现过程如下：
    # 1. 首先，根据 `normalized_cdf` 的形状，得到样本数 `B`。
    # 2. 然后，使用 PyTorch 的 `torch.linspace` 函数生成一个长度为 `n_tokens - 1` 的均匀网格，表示从 0 到 1 之间的 `n_tokens - 1` 个等距的 y 坐标。这里使用 `normalized_cdf.device` 指定了计算设备。
    # 3. 接下来，将这个网格张量 `ys` 复制 `B` 次，得到一个形状为 `(B, n_tokens - 1)` 的张量。
    # 4. 针对每个样本，找到其对应的归一化 CDF 中最小的非零值，以及它的位置。这里使用了 `torch.min` 函数和 `normalized_cdf == 0` 的逻辑运算符。
    # 5. 然后，使用 `torch.range` 函数生成一个长度为 `n_tokens - 1` 的序列，表示从 0 到 `n_tokens - 2` 的整数值，再将这个序列张量扩展为形状与 `ys_start` 相同的张量 `steps`。
    # 6. 最后，根据公式 `(y - y_start) / (n_tokens - 2) = (i - i_start) / (n_tokens - 2)`，计算每个 y 坐标对应的 CDF 值，其中 `y_start` 和 `i_start` 分别表示样本的最小非零 CDF 值和它的位置，`i` 表示当前 y 坐标的位置。这里使用了 PyTorch 的张量广播（broadcasting）功能，将 `ys_start` 和 `steps` 扩展为与 `ys` 相同的形状。最终得到的 `ys` 张量即为均匀采样的 y 坐标。
    # 总之，这个方法的作用是从一个归一化 CDF 中均匀地采样一组 y 坐标，用于构建离散的概率分布。
    # from cursor
    # To explain the create_ys method, it takes in two parameters: normalized_cdf and n_tokens. normalized_cdf is a tensor containing the normalized cumulative distribution function (CDF) of the tokens, while n_tokens is the number of tokens to be sampled.
    # The method first creates a tensor ys of shape (B, n_tokens - 1) by using torch.linspace to generate a sequence of n_tokens - 1 equally spaced values between 0 and 1, and then repeating it B times.
    # The ys_start tensor is created by finding the minimum non-zero value in each row of the normalized_cdf tensor and expanding it to the same shape as ys. The steps tensor is created by using torch.range to generate a sequence of values from 0 to n_tokens - 2, and then expanding it to the same shape as ys_start.
    # The final line of code calculates the ys tensor by subtracting ys_start from ys, multiplying the result by (n_tokens - 2), subtracting ys_start multiplied by steps, and then dividing the result by (n_tokens - 2).
    # The purpose of this calculation is to sample uniformly from the y-axis of the CDF. The ys_start tensor represents the minimum y-value for each sample in the CDF, while the steps tensor represents the distance between each sample. The final calculation maps the equally spaced values in ys to the y-axis of the CDF by scaling them by the distance between samples and adding the minimum y-value for each sample.

    @staticmethod
    def create_ys(normalized_cdf: Tensor, n_tokens: int) -> Tensor:
        """
        Sample uniformly from y-axis.
        """

        B = normalized_cdf.shape[0]
        # epsilon = (1 / (n_tokens - 1)) / 2
        ys = (
            torch.linspace(
                start=0,
                end=1.0,
                steps=n_tokens - 1,
                device=normalized_cdf.device,
            )
            .unsqueeze(0)
            .repeat(B, 1)
        )
        # 这段代码的作用是找到归一化 CDF 中每个样本的最小非零值，并将其扩展为与 ys 张量相同的形状。
        ys_start = (
            torch.min(normalized_cdf + (normalized_cdf == 0).float() * 1e8, dim=1)[0]
            .unsqueeze(-1)
            .expand_as(ys)
        )
        steps = (
            # torch.range()生成整数张量时,比torch.arange()长度多1
            torch.range(0, n_tokens - 2, device=normalized_cdf.device)
            .unsqueeze(0)
            .expand_as(ys_start)
        )
        # 这行代码相当于对于ys_start,ys,n_tokens-2,steps中的每一个元素都做了一下的操作,其中的*运算实际上时相对于张量的并行操作
        ys = ys_start + (((ys * (n_tokens - 2)) - ys_start * steps) / (n_tokens - 2))

        return ys

# Here's an example usage of the create_ys method:
# normalized_cdf = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.5, 0.8]])
# n_tokens = 4
# ys = AdaptiveTokenSampler.create_ys(normalized_cdf, n_tokens)
# print(ys)
# this will output:
# tensor([[0.0000, 0.3333, 0.6667],
#         [0.0000, 0.5000, 1.0000]])
# if __name__ == '__main__':
#     cdf_exp = torch.randn(3,6)
#     AdaptiveTokenSampler.create_ys(cdf_exp,6)


    # 这是一个静态方法，用于执行基于Transformer的模型中的标记分值分配步骤。该方法接受两个参数：`attn`和`v`。`attn`是Transformer的attention矩阵，形状为`[B x H x T x T]`，其中`B`是批大小，`H`是注意力头的数量，`T`是序列长度。`v`是Transformer多头注意力层的value输出，形状为`[B x H x T x C]`，其中`C`是值向量的维度。
    # 该方法首先沿着头部和通道维度计算值向量的L2范数，得到一个形状为`[B x T]`的张量。然后，它通过对每个头的CLS令牌的注意力权重求和，并将其乘以相应值向量的L2范数，计算每个标记的重要性分数。CLS令牌是序列中的第一个标记，通常携带有关整个序列的信息。重要性分数张量的形状为`[B x T-1]`，因为在后续分析中不考虑CLS标记。
    # 接下来，该方法通过将重要性分数沿着标记维度除以它们的总和来归一化这些分数。这样可以确保每个批次中的分数总和为1。最后，该方法对每个批次中的标准化分数按升序排序，并返回排序后的分数及其相应的索引。
    @staticmethod
    def score_assignment_step(attn: Tensor, v: Tensor) -> (Tensor, Tensor):
        """
        Token Score Assignment Step.
        :param attn: attention matrix
        :param v: values
        :return: sorted significance scores and their corresponding indices
        """

        B, H, _, _ = attn.shape
        C = v.shape[3] * H
        # 计算相应value对应的范数
        v_norm = torch.linalg.norm(
            v.transpose(1, 2).reshape(B, attn.shape[2], C), ord=2, dim=2
        )  # value norm of size [B x T]
        significance_score = attn[:, :, 0].sum(
            dim=1
        )  # attention weights of CLS token of size [B x T]
        significance_score = significance_score * v_norm  # [B x T]
        significance_score = significance_score[:, 1:]  # [B x T-1]

        # 论文第七页公式3
        significance_score = significance_score / significance_score.sum(
            dim=1, keepdim=True
        )  # [B x T-1]
        sorted_scores, sorted_indices = torch.sort(
            significance_score, descending=False, dim=1
        )

        return sorted_scores, sorted_indices

    def inverse_transform_sampling(
        self,
        sorted_scores: Tensor,
        sorted_indices: Tensor,
        attn: Tensor,
        n_tokens: int,
        raw_x: Tensor,
        n_ref_tokens: int,
    ) -> (Tensor, Tensor):
        """
        Sample tokens based on their significance scores.
        """
        B, N, C = raw_x.shape

        # 累计求和sorted_scores矩阵
        cdf = torch.cumsum(sorted_scores, dim=1)  # [B x T-1]

        normalized_cdf = (  # normalized cdf
            cdf - cdf.min(dim=1)[0].unsqueeze(dim=1)
        ) / ((cdf.max(dim=1)[0] - cdf.min(dim=1)[0]) / 1.0).unsqueeze(dim=1)

        ys = self.create_ys(normalized_cdf, n_ref_tokens).unsqueeze(
            dim=2
        )  # sampled values from y-axis of size [B, n-1, 1]

        # modified on 23.04.19
        # ys = self.create_ys(normalized_cdf, n_tokens).unsqueeze(
        #     dim=2
        # ) 
        normalized_cdf = normalized_cdf.unsqueeze(dim=1)  # [B, 1, N - 1]

        # expanded_ys = torch.Tensor.expand(ys, (B, n_tokens - 1, N - 1))
        # 好的，让我来进一步解释一下这段代码。

        # 扩展 `ys` 的维度，以便与 CDF 进行比较。
        # 具体将 `ys` 扩展为一个大小为 `(B, n-1, n-1)` 的张量，其中 `B` 是 batch_size，`n-1` 是令牌序列长度减一。
        # 然后，计算 `ys` 和 CDF 之间的差异，并对结果取绝对值，以便找到最接近 CDF 的令牌索引。
        # 最后，使用 PyTorch 的 `torch.min` 函数，沿着 `dim=2` 的维度找到最小值的索引，以得到采样的令牌索引。
        # 这个过程会生成一个大小为 `(B, n-1)` 的张量，其中每个元素都是一个令牌索引，表示在当前的 CDF 下，应该选择哪个令牌作为下一个输入。
        expanded_ys = torch.Tensor.expand(ys, (B, ys.shape[1], ys.shape[1]))
        diff_tokens = ys.shape[1] - (N - 1)
        tokens_to_pick_ind = torch.min(
            torch.abs(expanded_ys - F.pad(normalized_cdf, (diff_tokens, 0))),
            dim=2,
        )[
            1
        ]  # [B x n-1]

        # Offsetting token indices
        tokens_to_pick_ind = tokens_to_pick_ind - diff_tokens

        # Sort attention matrix and add CLS weights.
        attn_sorted = torch.gather(
            attn[:, :, 1:],
            2,
            sorted_indices.unsqueeze(1)
            .unsqueeze(-1)
            .expand(B, self.num_heads, N - 1, N),
        )  # [B x h x T-1 x T]

        attn_tmp = F.pad(attn_sorted, (0, 0, 0, 1), value=0.0)  # [B x h x T x T]

        # # Sort tokens and add CLS token.
        # 这段代码是在对输入进行预处理，以便将采样的令牌插入到正确的位置。
        # 具体来说，它首先从 `raw_x` 中选择除第一个令牌以外的所有令牌，并按照 `sorted_indices` 中的顺序进行排序.
        # 然后，它将这些令牌插入到 `raw_x_temp` 中，并将结果填充到一个大小为 `[B x n x C]` 的张量中。
        # 接下来，它使用 `get_unique_indices` 函数从 `tokens_to_pick_ind` 中获取唯一的索引，并截取前 `N-1` 个索引。这些索引表示应该在哪个位置插入新的令牌。
        # 具体来说，`torch.gather` 函数用于从 `raw_x` 中选择除第一个令牌以外的所有令牌。它的第一个参数是一个大小为 `[B x n x C]` 的张量，其中 `B` 是 batch_size，`n` 是输入序列的长度，`C` 是每个令牌的特征向量维度。第二个参数是一个大小为 `[B x (n-1)]` 的张量，表示要选择的元素的索引。`sorted_indices` 是一个已经按照采样的令牌顺序进行排序的张量，因此我们可以直接将它传递给 `gather` 函数。最后，我们通过使用 `unsqueeze` 和 `expand` 函数将 `sorted_indices` 扩展为 `[B x (n-1) x C]` 的张量，以便与 `raw_x` 进行匹配。
        # 接下来，我们使用 `F.pad` 函数将选择的令牌插入到 `raw_x_temp` 中。具体来说，我们在第二维的末尾填充一个零值，在序列的结尾插入一个额外的零值。这个过程可以保证在插入新的令牌时不会影响模型的输入形状。
        # 最后使用 `get_unique_indices` 函数从 `tokens_to_pick_ind` 中获取唯一的索引，并截取前 `N-1` 个索引，这些索引表示新令牌应该插入到哪个位置。`get_unique_indices` 函数的作用是获取给定张量中每个元素的唯一索引，并返回一个大小为 `[B x max_value]` 的张量。在这里，我们将 `max_value` 设置为 `N-1`，以便获取前 `N-1` 个索引。
        raw_x_tmp = torch.gather(
            raw_x[:, 1:], 1, sorted_indices.unsqueeze(-1).expand(B, N - 1, C)
        )
        raw_x_tmp = F.pad(raw_x_tmp, (0, 0, 0, 1), value=0.0)  # [B x n x C]

        unique_indices = self.get_unique_indices(
            indices=tokens_to_pick_ind, max_value=N - 1
        )[:, : N - 1]

        # modified on 23.04.19
        # unique_indices = self.get_unique_indices(
        #     indices=tokens_to_pick_ind, max_value=N - 1
        # )[:, : n_tokens - 1]

        # Prune the attention matrix and input tokens.
        attn_tmp = torch.gather(
            attn_tmp,
            2,
            unique_indices.unsqueeze(1)
            .unsqueeze(3)
            .expand(B, self.num_heads, n_tokens - 1, N),
        )
        raw_x_tmp = torch.gather(
            raw_x_tmp, 1, unique_indices.unsqueeze(2).expand(B, n_tokens - 1, C)
        )

        attn_tmp = torch.cat([attn[:, :, 0:1], attn_tmp], dim=2)
        raw_x_tmp = torch.cat([raw_x[:, 0:1], raw_x_tmp], dim=1)

        # 这段代码是在对插入位置进行筛选，并将结果返回给调用函数。具体来说，它使用 `unique_indices` 张量来确定哪些位置是有效的插入位置，并将结果存储在 `policy` 张量中。
        # 然后，它将 `policy` 倒数第二个维度上补充一行值1.0。
        # 接下来，它将 `raw_x_tmp` 和 `attn_tmp` 作为输出，并将 `policy` 和 `sampler` 作为附加信息一起返回。
        # 具体来说，我们首先通过将 `unique_indices` 与 `(N-1)` 进行比较来确定哪些位置是有效的插入位置，将结果存储在 `policy` 张量中。
        # 然后，我们使用 `unsqueeze` 函数将 `policy` 的最后一个维度扩展为大小为 1，并使用 `F.pad` 函数在第一维的开头添加一个大小为 1 的零值，以便在序列的开头添加第一个令牌。
        # 最后，我们将 `raw_x_tmp` 和 `attn_tmp` 作为输出，并使用 `torch.nonzero` 函数获取 `policy` 中非零元素的索引，存储在 `sampler` 变量中，并将 `policy` 作为附加信息返回。
        # 这个过程可以保证在采样时只会在有效的位置插入新的令牌。
        policy = (unique_indices != (N - 1)).unsqueeze(-1).float()
        policy = F.pad(policy, (0, 0, 1, 0), value=1.0)
        selected_x = raw_x_tmp
        attn = attn_tmp
        # 采样每行非零值的索引
        sampler = torch.nonzero(policy) #sampler.shape=(B,n,1)

        return selected_x, attn, policy, sampler

    def forward(
        self,
        x: Tensor,
        policy: Tensor,
        sampler: Tensor,
        n_tokens: float,
        raw_x: Tensor,
        n_ref_tokens: int,
    ):
        B, N, C = x.shape

        # 将一个 `torch.Tensor` 类型的标量 `N` 转换为 Python 中的标量类型，并将其存储在一个变量中。
        # 具体来说，它调用了 `cpu()` 方法将 `N` 张量移动到 CPU 上，并调用了 `item()` 方法将其转换为 Python 的标量类型。
        # 这个过程可以将张量中的值提取出来，并将其存储在一个变量中，以便后续使用。
        if isinstance(N, Tensor):
            N = N.cpu().item()

        if n_tokens > N:  # Number of tokens to be sampled can't be larger than N.
            n_tokens = N
        if n_tokens <= 1.0:  # When n_tokens is a ratio.
            n_tokens = n_tokens * N
        if n_tokens < 8:  # Number of tokens to be sampled can't be less than 8.
            n_tokens = 8

        n_tokens = round(n_tokens)
        if N < n_tokens:
            n_tokens = N

        # 此处的self.qkv()实际上时在AdaptiveTokenSampler的父类attention中已经定义好了,policy.shape = (B,N,1)
        qkv = self.qkv(x, policy, sampler)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        qkv = qkv * policy.unsqueeze(0).unsqueeze(
            2
        )  # Get rid of previously removed tokens.  #(3,B,H,N,C//H)*(1,B,1,N,1),利用policy中为1的元素对token进行筛选
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn_no_softmax = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax_with_policy(attn_no_softmax, policy)  # [B x H x T x T] 此处的policy是不是可以做些文章呢?和ViT for small dataset

        # --------------------------
        # Token Score Assignment
        # --------------------------

        sorted_scores, sorted_indices = self.score_assignment_step(attn, v)

        # ---------------------------------------------------
        # Inverse Transform Sampling based on the token score 
        # ---------------------------------------------------

        selected_x, attn, policy, sampler = self.inverse_transform_sampling(
            sorted_scores, sorted_indices, attn, n_tokens, raw_x, n_ref_tokens
        )

        x = (attn @ v).transpose(1, 2).reshape(B, attn.shape[2], C)

        # 这段代码在执行一些令牌屏蔽或丢弃的操作。
        # 条件语句 `self.drop_tokens` 检查是否启用了令牌丢弃。如果启用了，则代码会创建一个 `policy` 张量中令牌和的最大值所对应的掩码。
        # `sampler` 张量是通过随机选择要丢弃的令牌索引创建的。`sampler_out` 张量是通过将 `sampler` 的第一列乘以 `out_mask_size` 并加上第二列创建的。
        # `sampler_input` 和 `sampler_output` 张量是通过扩展 `sampler` 和 `sampler_out` 来匹配 `x` 和 `selected_x` 张量中的列数。
        # `flatten_x` 和 `flatten_selected_x` 张量是通过沿着第一个维度将 `x` 和 `selected_x` 张量展平来创建的。
        # `x_prunned` 和 `selected_x_prunned` 张量是通过从 `flatten_x` 和 `flatten_selected_x` 中选择 `sampler_input` 中指定的索引来创建的。
        # `out_zero_mask` 张量是通过将零张量扩展到与 `x_prunned` 和 `selected_x_prunned` 的展平形状相匹配来创建的。
        # 然后使用此张量通过将修剪后的 `x_prunned` 和 `selected_x_prunned` 张量散布到 `sampler_output` 中指定的索引位置来更新 `x` 和 `selected_x` 张量。
        # 最后，`policy` 张量通过在 `sampler_out` 指定的位置创建一个值为1的掩码进行更新，并将其重新形状以匹配 `x` 和 `selected_x` 的形状。
        # Pruning
        if self.drop_tokens:
            out_mask_size = policy.sum(1).max().int()   # (B,n,1) -> (B,1) -> 单个batch中沿着token维度累计最大的out_mask_size

            sampler_out = sampler[:, 0] * out_mask_size + sampler[:, 1]     # sampler和policy一样的维度(B,n,1),所以sampler_out.shape=(B,1)
            sampler = sampler[:, 0] * n_tokens + sampler[:, 1]
            # 这里是不是有问题,因为按照之前相关函数输出已经有sampler_out.shape=(B,1)?
            sampler_input = sampler.unsqueeze(-1).expand(-1, C)
            sampler_output = sampler_out.unsqueeze(-1).expand(-1, C)
            flatten_x = x.reshape(-1, C)
            flatten_selected_x = selected_x.reshape(-1, C)

            x_prunned = torch.gather(flatten_x, 0, sampler_input)
            selected_x_prunned = torch.gather(flatten_selected_x, 0, sampler_input)

            out_zero_mask = self.out_zero_mask.expand(B * out_mask_size, -1)    # (B * out_mask_size,dim)

            # gather()与scatter()函数实现了相反的功能,前者从flatten_x中按照sampler_input的索引采样,后者把采样到的数据按照x_prunned指定的位置写入到sampler_out中
            x = out_zero_mask.scatter(
                0, sampler_output, x_prunned, reduce="add"
            ).reshape((B, out_mask_size, C))
            selected_x = out_zero_mask.scatter(
                0, sampler_output, selected_x_prunned, reduce="add"
            ).reshape((B, out_mask_size, C))

            policy = (
                out_zero_mask[:, 0]
                .scatter(0, sampler_out, 1, reduce="add")
                .reshape(B, out_mask_size, 1)
            )

        x = self.proj(x, policy, sampler)
        x = x * policy
        x = self.proj_drop(x)
        return x, selected_x, policy, sampler


class ATSBlock(nn.Module):
    """
    Transformer Block + ATS
    """

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
        drop_tokens=False,
    ):
        super().__init__()
        self.insert_control_point = insert_control_point
        self.norm1 = norm_layer(dim)

        self.attn = AdaptiveTokenSampler(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop_path=drop_path,
            drop_tokens=drop_tokens,
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

    def forward(
        self,
        x,
        n_tokens,
        policy: Tensor = None,
        sampler: Tensor = None,
        n_ref_tokens: int = 197,
    ):
        x_out, selected_x, policy, sampler = self.attn(
            x=self.norm1(x),
            policy=policy,
            sampler=sampler,
            n_tokens=n_tokens,
            raw_x=x,
            n_ref_tokens=n_ref_tokens,
        )
        x = selected_x + self.drop_path(x_out)
        x = x * policy
        out = self.mlp(x=self.norm2(x), policy=policy, sampler=sampler)
        x = x + self.drop_path(out)
        x = x * policy
        return x, policy
        