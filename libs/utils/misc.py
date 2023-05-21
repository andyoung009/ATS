

import json
import logging
import math
import os
from datetime import datetime

import numpy as np
import psutil
from fvcore.nn import flop_count

import libs.utils.logging as logging
import libs.utils.multiprocessing as mpu
import torch
from fvcore.nn.activation_count import activation_count
from iopath.common.file_io import g_pathmgr
from matplotlib import pyplot as plt
from libs.datasets.utils import pack_pathway_output
from libs.models.batchnorm_helper import SubBatchNorm3d
from torch import nn

logger = logging.get_logger(__name__)


def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def params_count(model, ignore_bn=False):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    if not ignore_bn:
        return np.sum([p.numel() for p in model.parameters()]).item()
    else:
        count = 0
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm3d):
                for p in m.parameters(recurse=False):
                    count += p.numel()
    return count


def calculate_flops(model, cfg):
    if cfg.TRAIN.ENABLE:
        S = cfg.DATA.TRAIN_CROP_SIZE
        B = 1
    else:
        S = cfg.DATA.TEST_CROP_SIZE
        B = cfg.TEST.NUM_SPATIAL_CROPS * cfg.TEST.NUM_ENSEMBLE_VIEWS
    C = cfg.DATA.INPUT_CHANNEL_NUM[0]
    T = cfg.DATA.NUM_FRAMES

    inputs = [torch.rand([B, C, T, S, S]).cuda(), torch.rand([B, T]).cuda()]
    flops, _ = flop_count(model, inputs)
    return sum(flops.values())


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3

    return usage, total


def _get_model_analysis_input(cfg, use_train_input):
    """
    Return a dummy input for model analysis with batch size 1. The input is
        used for analyzing the model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            libs/config/defaults.py
        use_train_input (bool): if True, return the input for training. Otherwise,
            return the input for testing.

    Returns:
        inputs: the input for model analysis.
    """
    rgb_dimension = 3
    if use_train_input:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_FRAMES,
            cfg.DATA.TRAIN_CROP_SIZE,
            cfg.DATA.TRAIN_CROP_SIZE,
        )
    else:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_FRAMES,
            cfg.DATA.TEST_CROP_SIZE,
            cfg.DATA.TEST_CROP_SIZE,
        )
    if cfg.XVIT.CONSENSUS_TYPE == "vit":
        frames_index = torch.arange(input_tensors.shape[1])
    else:
        frames_index = None
    model_inputs = pack_pathway_output(cfg, input_tensors, frames_index)
    for i in range(len(model_inputs)):
        model_inputs[i] = model_inputs[i].unsqueeze(0)
        if cfg.NUM_GPUS:
            model_inputs[i] = model_inputs[i].cuda(non_blocking=True)

    inputs = (model_inputs,)
    return inputs


def get_model_stats(model, cfg, mode, use_train_input):
    """
    Compute statistics for the current model given the config.
    Args:
        model (model): model to perform analysis.
        cfg (CfgNode): configs. Details can be found in
            libs/config/defaults.py
        mode (str): activation count (mega).
        use_train_input (bool): if True, compute statistics for training. Otherwise,
            compute statistics for testing.

    Returns:
        float: the total number of count of the given model.
    """
    assert mode in ["activation"], "'{}' not supported for model analysis".format(mode)
    if mode == "activation":
        model_stats_fun = activation_count

    # Set model to evaluation mode for analysis.
    # Evaluation mode can avoid getting stuck with sync batchnorm.
    model_mode = model.training
    model.eval()
    # inputs = _get_model_analysis_input(cfg, use_train_input)
    # count_dict, *_ = model_stats_fun(model, inputs)
    count = 0  # sum(count_dict.values())
    model.train(model_mode)
    return count

# 这是一个记录深度学习模型信息的Python函数。该函数接受三个参数：`model`，`cfg`和`use_train_input`。
# `model`是要记录信息的深度学习模型。
# `cfg`是一个配置节点，包含有关模型架构和超参数的信息。
# `use_train_input`是一个布尔标志，用于确定是记录模型的训练输入（如果设置为`True`）还是测试输入（如果设置为`False`）的信息。
# 该函数记录以下信息：
# - `model`：模型的字符串表示形式。
# - `Params`：模型中的参数数量。
# - `Mem`：模型使用的GPU内存量。
# - `Activations`：模型中的激活数量（即每个层的输出值）。
# - `GFLOPs`：模型执行的GFLOP数量（即浮点运算次数）。
# - `nvidia-smi`：`nvidia-smi`命令的输出，显示GPU使用情况信息。

# 此函数可用于在训练或测试过程中监视深度学习模型的性能和资源使用情况。
def log_model_info(model, cfg, use_train_input=True):
    """
    Log info, includes number of parameters, gpu usage  and activation count.
        The model info is computed when the model is in validation mode.
    Args:
        model (model): model to log the info.
        cfg (CfgNode): configs. Details can be found in
            libs/config/defaults.py
        use_train_input (bool): if True, log info for training. Otherwise,
            log info for testing.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    # 这行代码使用gpu_mem_usage()函数获取当前GPU的内存使用情况，并将其以逗号分隔的千位数格式化为字符串，然后将其作为信息消息记录在日志中。
    # 该函数用于在训练或测试过程中监视模型的GPU内存使用情况。
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    logger.info(
        "Activations: {:,} M".format(
            get_model_stats(model, cfg, "activation", use_train_input)
        )
    )
    logger.info("GFLOPs:  {:,}".format(calculate_flops(model, cfg)))
    logger.info("nvidia-smi")
    os.system("nvidia-smi")


def is_eval_epoch(cfg, cur_epoch):
    """
    Determine if the model should be evaluated at the current epoch.
    Args:
        cfg (CfgNode): configs. Details can be found in
            libs/config/defaults.py
        cur_epoch (int): current epoch.
    """
    if cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH:
        return True

    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0


def plot_input(tensor, bboxes=(), texts=(), path="./tmp_vis.png"):
    """
    Plot the input tensor with the optional bounding box and save it to disk.
    Args:
        tensor (tensor): a tensor with shape of `NxCxHxW`.
        bboxes (tuple): bounding boxes with format of [[x, y, h, w]].
        texts (tuple): a tuple of string to plot.
        path (str): path to the image to save to.
    """
    tensor = tensor.float()
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    f, ax = plt.subplots(nrows=1, ncols=tensor.shape[0], figsize=(50, 20))
    for i in range(tensor.shape[0]):
        ax[i].axis("off")
        ax[i].imshow(tensor[i].permute(1, 2, 0))
        # ax[1][0].axis('off')
        if bboxes is not None and len(bboxes) > i:
            for box in bboxes[i]:
                x1, y1, x2, y2 = box
                ax[i].vlines(x1, y1, y2, colors="g", linestyles="solid")
                ax[i].vlines(x2, y1, y2, colors="g", linestyles="solid")
                ax[i].hlines(y1, x1, x2, colors="g", linestyles="solid")
                ax[i].hlines(y2, x1, x2, colors="g", linestyles="solid")

        if texts is not None and len(texts) > i:
            ax[i].text(0, 0, texts[i])
    f.savefig(path)


def frozen_bn_stats(model):
    """
    Set all the bn layers to eval mode.
    Args:
        model (model): model to set bn layers to eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def aggregate_sub_bn_stats(module):
    """
    Recursively find all SubBN modules and aggregate sub-BN stats.
    Args:
        module (nn.Module)
    Returns:
        count (int): number of SubBN module found.
    """
    count = 0
    for child in module.children():
        if isinstance(child, SubBatchNorm3d):
            child.aggregate_stats()
            count += 1
        else:
            count += aggregate_sub_bn_stats(child)
    return count


def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            libs/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.NUM_GPUS > 1:
        # 多进程训练模型
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)


def get_class_names(path, parent_path=None, subset_path=None):
    """
    Read json file with entries {classname: index} and return
    an array of class names in order.
    If parent_path is provided, load and map all children to their ids.
    Args:
        path (str): path to class ids json file.
            File must be in the format {"class1": id1, "class2": id2, ...}
        parent_path (Optional[str]): path to parent-child json file.
            File must be in the format {"parent1": ["child1", "child2", ...], ...}
        subset_path (Optional[str]): path to text file containing a subset
            of class names, separated by newline characters.
    Returns:
        class_names (list of strs): list of class names.
        class_parents (dict): a dictionary where key is the name of the parent class
            and value is a list of ids of the children classes.
        subset_ids (list of ints): list of ids of the classes provided in the
            subset file.
    """
    try:
        with g_pathmgr.open(path, "r") as f:
            class2idx = json.load(f)
    except Exception as err:
        print("Fail to load file from {} with error {}".format(path, err))
        return

    max_key = max(class2idx.values())
    class_names = [None] * (max_key + 1)

    for k, i in class2idx.items():
        class_names[i] = k

    class_parent = None
    if parent_path is not None and parent_path != "":
        try:
            with g_pathmgr.open(parent_path, "r") as f:
                d_parent = json.load(f)
        except EnvironmentError as err:
            print("Fail to load file from {} with error {}".format(parent_path, err))
            return
        class_parent = {}
        for parent, children in d_parent.items():
            indices = [class2idx[c] for c in children if class2idx.get(c) is not None]
            class_parent[parent] = indices

    subset_ids = None
    if subset_path is not None and subset_path != "":
        try:
            with g_pathmgr.open(subset_path, "r") as f:
                subset = f.read().split("\n")
                subset_ids = [
                    class2idx[name]
                    for name in subset
                    if class2idx.get(name) is not None
                ]
        except EnvironmentError as err:
            print("Fail to load file from {} with error {}".format(subset_path, err))
            return

    return class_names, class_parent, subset_ids
