import warnings
import torch
from torch._six import inf
import json
import numpy as np


def clip_grad_norm_for_autograd(grads_tuple, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of grads_tuple.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        grads_tuple (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the grads_tuple (viewed as a single vector).
    """

    if isinstance(grads_tuple, torch.Tensor):
        grads_tuple = [grads_tuple]
    grads_tuple = list(filter(lambda p: p is not None, grads_tuple))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.abs().max() for p in grads_tuple)
    else:
        total_norm = 0
        for p in grads_tuple:
            # print(p)
            param_norm = p.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        new_grads = []
        for p in grads_tuple:
            new_grads.append(p.mul_(clip_coef))
    else:
        new_grads=grads_tuple

    return new_grads


def dict_norm(grad_dict):
    # cal the norm of a dictionary. (each element of the dictionary is a gradients list of a layer.)
    total_norm = 0
    norm_type = 2
    for key in grad_dict:
        param_norm = grad_dict[key].data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def list_norm(grads_list):
    # cal the norm of a tuple. (each element of the dictionary is a gradients list of a layer.)
    total_norm = 0
    norm_type = 2
    for grads in grads_list:
        param_norm = grads.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_percentile(json_file, percentile_value):
    with open(json_file) as f:
        data_dict = json.load(fp=f)
    clip_s = data_dict['clip_s']
    s_statis = np.percentile(clip_s, percentile_value) 
    print(s_statis)
    return s_statis
