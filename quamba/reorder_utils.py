import logging
import functools
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, KMeans

import torch
import torch.nn as nn
from datasets import load_dataset

from .qActLayer import ActIdentity

@torch.no_grad()
def get_act_channel_stats_mamba2(
    model, tokenizer, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_act_hook(m, inputs, outputs, name):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
            assert isinstance(inputs, torch.Tensor)
        hidden_dim = inputs.shape[-1]
        inputs = inputs.view(-1, hidden_dim)
        coming_scales = torch.mean(inputs.abs(), dim=0).float().cpu()

        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], coming_scales)
        else:
            act_scales[name] = coming_scales

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, ActIdentity)):
            # print(name)
            hooks.append(m.register_forward_hook(
                    functools.partial(stat_act_hook, name=name)
                )
            )
            logging.debug(f"Register forward hook for {name} for getting reorder scales")

    logging.info("Prepare calibration input")
    calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
    calibration_dataset.shuffle(seed=42)
    logging.info("Get channel stats for reordering")
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(calibration_dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids) 

    for h in hooks:
        h.remove()

    return act_scales

def reorder_linear(linear, in_reorder_index=None, out_reorder_index=None):
    device = linear.weight.device
    if in_reorder_index is not None:
        in_reorder_index = in_reorder_index.to(device)
        linear.weight.data = torch.index_select(linear.weight.data, 1, in_reorder_index).contiguous()
    if out_reorder_index is not None:
        out_reorder_index = out_reorder_index.to(device)
        linear.weight.data = torch.index_select(linear.weight.data, 0, out_reorder_index).contiguous()
        if linear.bias is not None:
            linear.bias.data = torch.index_select(linear.bias.data, 0, out_reorder_index).contiguous()

def reorder_norm(norm, reorder_index=None):
    if reorder_index is not None:
        device = norm.weight.device
        reorder_index = reorder_index.to(device)
        norm.weight.data = torch.index_select(norm.weight.data, 0, reorder_index).contiguous()
        if norm.bias is not None:
            norm.bias.data = torch.index_select(norm.bias.data, 0, reorder_index).contiguous()

def reorder_conv(conv, reorder_index=None):
    if reorder_index is not None:
        device = conv.weight.device
        reorder_index = reorder_index.to(device)
        conv.weight.data = torch.index_select(conv.weight.data, 0, reorder_index).contiguous()
        if conv.bias is not None:
            conv.bias.data = torch.index_select(conv.bias.data, 0, reorder_index).contiguous()


@torch.no_grad()
def group_wise_sort_indices(tensor, headdim, ssd_ngroups, nhead_groups=4, ndim_groups=4):
    device = tensor.device
    reshaped_tensor = tensor.view(ssd_ngroups, -1, headdim)     # calibrated channel max: (ssd_ngroups*nheads*headdim), per-channel statistics
    _, nheads, _ = reshaped_tensor.shape                        # Reshape the per-channel statistics to (ssd_ngroups, nheads, headdim)
    dim_indices = torch.argsort(reshaped_tensor, dim=-1)        # channel-wised sort for each head : [sort([0, 1, 2, ..., hdim-1]), sort([0, 1, 2, ..., hdim-1]), ...]
    base_indices = torch.arange(tensor.numel()).view(ssd_ngroups, -1, headdim)   # create base indices for each head: [[0, 1, 2, ..., hdim-1], [hdim, hdim+1, hdim+2, ..., 2*hdim-1], ...]
    sorted_dim_indices = base_indices.gather(-1, dim_indices)   # channel-wised sort for each head
    
    # group heads and dims
    sorted_dim_tensor = reshaped_tensor.gather(-1, dim_indices)  # The scales of each dim in the head are sorted: small->large, reshaped_tensor: [ssd_ngroups, nheads, headdim]
    sorted_dim_tensor_norm = sorted_dim_tensor / sorted_dim_tensor.norm(dim=-1, keepdim=True) # normalize
    sorted_dim_tensor_np = sorted_dim_tensor_norm.clone().cpu().numpy()
    sorted_head_dim_indices = sorted_dim_indices.clone()
    head_indices = torch.arange(ssd_ngroups*nheads).reshape(ssd_ngroups, nheads)
    head_groups_size = []
    dim_groups_size = []
    for g in range(ssd_ngroups):
        # Cluster the sorted heads (small->large) based on euclidean distances
        head_clustering = AgglomerativeClustering(
            n_clusters=nhead_groups, metric='euclidean', linkage='ward').fit(sorted_dim_tensor_np[g])
        head_indices[g, :] = head_indices[g, np.argsort(head_clustering.labels_)]
        sorted_head_dim_indices[g, :] = sorted_head_dim_indices[g, np.argsort(head_clustering.labels_)] # reordering head indices
        # get head group size
        grouped_head_indices = np.sort(head_clustering.labels_)
        _, h_start_indices = np.unique(grouped_head_indices, return_index=True)
        h_end_indices = np.append(h_start_indices[1:], len(grouped_head_indices))
        head_groups_size.append(list(h_end_indices - h_start_indices))

        head_group_ranges = tuple(zip(h_start_indices, h_end_indices))
        dim_groups_size_tmp = list()
        for row_start, row_end in head_group_ranges:
            head_scales = sorted_dim_tensor_np[g][row_start:row_end, :]
            head_scales_t = head_scales.transpose()
            init_center = head_scales_t[0::headdim//ndim_groups, :]
            # initize KMeans with centers to avoid reordering the channels
            dim_clustering = KMeans(n_clusters=ndim_groups, init=init_center).fit(head_scales_t)
            assert np.all(np.diff(dim_clustering.labels_) >= 0), "dim clustering label should be monotonic increasing"
            _, d_start_indices = np.unique(dim_clustering.labels_, return_index=True)
            d_end_indices = np.append(d_start_indices[1:], len(dim_clustering.labels_))
            dim_groups_size_tmp.append(list(d_end_indices - d_start_indices))
        dim_groups_size.append(dim_groups_size_tmp)
    
    # Flatten the sorted indices to get a 1D tensor
    head_groups_size = torch.tensor(head_groups_size, dtype=torch.int32).to(device)
    head_dim_indices_flat = sorted_head_dim_indices.flatten().to(device)
    head_indices_flat = head_indices.flatten().to(device)
    dim_groups_size = torch.tensor(dim_groups_size, dtype=torch.int32).to(device)
    
    return head_dim_indices_flat, head_groups_size, head_indices_flat, dim_groups_size


@torch.no_grad()
def get_reorder_params(model, model_type, tokenizer, num_samples=512, seq_len=512):

    reorder_params = {}
    if model_type == 'mamba2':
        act_scales = get_act_channel_stats_mamba2(model, tokenizer, num_samples=num_samples, seq_len=seq_len)
        
        device = next(model.parameters()).device
        layers = model.backbone.layers
        head_groups_list = []
        head_index_list = []
        channel_group_list = []
        channel_index_list = []
        for i in range(len(layers)):
            x_channel_scales = act_scales[f"backbone.layers.{i}.mixer.x_conv_out"]
            assert x_channel_scales.shape[-1] % layers[i].mixer.headdim == 0, "The hidden dim must be divisible by headdim"
            channel_index, head_groups, head_index, dim_groups = group_wise_sort_indices(
                x_channel_scales, layers[i].mixer.headdim, layers[i].mixer.ngroups)
            # TODO: use dynamic programming to get optimal channel_group, and remove the hardcoded channel groups
            channel_group = torch.tensor([48, 12, 2, 2], dtype=torch.int32, device=device)
            head_groups_list.append(head_groups)
            head_index_list.append(head_index)
            # channel_group_list.append(channel_group)
            channel_group_list.append(dim_groups)
            channel_index_list.append(channel_index)
        reorder_params["head_groups"] = head_groups_list
        reorder_params["head_index"] = head_index_list
        reorder_params["channel_group"] = channel_group_list
        reorder_params["channel_index"] = channel_index_list
    else:
        raise NotImplementedError(f"{model_type} do not support reorder")
    
    return reorder_params


def reorder_mamba(model, reorder_params):
    logging.info("reorder mamba channels and heads")
    # reorder model weights
    layers = model.backbone.layers
    for i in range(len(layers)):
        head_idx = reorder_params["head_index"][i]
        ch_idx = reorder_params["channel_index"][i]
        # NOTE(brian1009): In_proj's output is the combination of z,x,b,c,dt, we reorder z, x, and dt
        in_proj_output_index = torch.arange(layers[i].mixer.in_proj.out_features)
        in_proj_output_index[0:layers[i].mixer.d_ssm] = ch_idx # reorder z
        in_proj_output_index[layers[i].mixer.d_ssm:2*layers[i].mixer.d_ssm] = ch_idx + layers[i].mixer.d_ssm # reorder x
        in_proj_output_index[-layers[i].mixer.nheads:] = head_idx + 2*layers[i].mixer.d_ssm + 2*layers[i].mixer.ngroups*layers[i].mixer.d_state # reorder dt
        reorder_linear(layers[i].mixer.in_proj, out_reorder_index=in_proj_output_index)
        # NOTE(brian1009): We reorder the indices for conv1d
        conv1d_indices = torch.arange(layers[i].mixer.conv1d.in_channels)
        conv1d_indices[0:layers[i].mixer.d_ssm] = ch_idx
        reorder_conv(layers[i].mixer.conv1d, reorder_index=conv1d_indices)
        # NOTE(HY): reorder A, D, and dt_bias to match the reordered heads
        layers[i].mixer.A_log.data = layers[i].mixer.A_log[head_idx].data
        layers[i].mixer.D.data = layers[i].mixer.D[head_idx].data
        layers[i].mixer.dt_bias.data = layers[i].mixer.dt_bias[head_idx].data
        # NOTE(brian1009): reorder the gated_rms_norm, out_proj, to match the reordered channels
        reorder_norm(layers[i].mixer.norm, reorder_index=ch_idx)
        reorder_linear(layers[i].mixer.out_proj, in_reorder_index=ch_idx)