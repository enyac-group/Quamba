import os
import sys
import time
import gzip
import json
import socket
import logging
import argparse
import numpy as np
from datetime import datetime
from functools import partial

import torch
from torch.autograd.profiler import record_function
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from hta.trace_analysis import TraceAnalysis
# mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from utils import set_deterministic
from utils import get_quantize_options
from quamba.megatron_utils import _GPTSentencePieceTokenizer
from quamba.quamba_mixer_seq import QuambaLMHeadModel
from quamba.modelutils_mamba import quantize_model_mamba
from quamba.qMambaLayer import W4A8QMamba, W4A16QMamba, W8A8QMamba
from quamba.qMamba2 import W4A8QMamba2, W4A16QMamba2, W8A8QMamba2

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def trace_handler(prof: torch.profiler.profile, dir_name="torch_profile_output",
                  worker_name = None, use_gzip: bool = False,
                  file_prefix="prefilling", device="cuda:0"):
    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            raise RuntimeError("Can't create directory: " + dir_name) from e
    if not worker_name:
        worker_name = f"{socket.gethostname()}_{os.getpid()}"
    # Use nanosecond here to avoid naming clash when exporting the trace
    timestamp = time.time_ns()
    file_name = f"{file_prefix}.{worker_name}.{timestamp}.pt.trace.json"
    if use_gzip:
        file_name = file_name + ".gz"
    prof.export_chrome_trace(os.path.join(dir_name, file_name))
    # Fix the rank issue for  HolisticTraceAnalysis
    # reference: https://github.com/facebookresearch/HolisticTraceAnalysis/issues/107
    # FIXME: This does not work for json.gz
    # rn_rank = np.random.randint(low=0, high=16, dtype=int) # If there are multiple traces files, then each file should have a unique rank value.
    if use_gzip:
        with gzip.open(os.path.join(dir_name, file_name), mode="rt") as fin:
            data = json.loads(fin.read())
        data["distributedInfo"] = {"rank": 0} # must use 0. I don't know why. If there are multiple traces files, then each file should have a unique rank value.
        with gzip.open(os.path.join(dir_name, file_name), 'w') as fout:
            fout.write(json.dumps(data).encode('utf-8')) 
    else:
        with open(os.path.join(dir_name, file_name), "r") as fin:
            data = json.load(fin)
        data["distributedInfo"] = {"rank": 0} # must use 0. I don't know why. If there are multiple traces files, then each file should have a unique rank value.
        with open(os.path.join(dir_name, file_name), "w") as fout:
            json.dump(data, fout, indent=2)

    analyzer = TraceAnalysis(trace_files={0: file_name}, trace_dir=dir_name)
    kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown(visualize=False, num_kernels=100)
    kernel_type_metrics_df.to_csv(os.path.join(dir_name, f'kernel_type_metrics.{file_prefix}.{timestamp}.csv'), index=False)
    kernel_metrics_df.to_csv(os.path.join(dir_name, f'kernel_metrics.{file_prefix}.{timestamp}.csv'), index=False)
    # this feature is at https://github.com/facebookresearch/HolisticTraceAnalysis/pull/209
    # To get accurate kernel results, checkout this branch https://github.com/hychiang-git/HolisticTraceAnalysis/tree/dev/no_merge_cpu_kernels
    if hasattr(analyzer, "get_gpu_user_annotation_breakdown"):
        try:
            user_annotation_kernel_type_metrics_df, user_annotation_metrics_df = analyzer.get_gpu_user_annotation_breakdown(visualize=False, num_kernels=100)
            user_annotation_kernel_type_metrics_df.to_csv(os.path.join(dir_name, f'user_annotation_kernel_type_metrics.{file_prefix}.{timestamp}.csv'), index=False)
            user_annotation_metrics_df.to_csv(os.path.join(dir_name, f'user_annotation_metrics.{file_prefix}.{timestamp}.csv'), index=False)
        except Exception as e:
            logging.warning(f"Failed to get user annotation breakdown: {e}")
    # Construct the memory timeline file.
    # !!! This does not work for graph cache !!!
    html_name = f"{file_prefix}.{worker_name}.{timestamp}.html"
    prof.export_memory_timeline(os.path.join(dir_name, html_name), device=device)

def get_size(module):
    if module is None:
        return 0
    if isinstance(module, torch.nn.Parameter) or isinstance(module, torch.Tensor):
        return module.nelement() * module.element_size()
    param_size = 0
    buffer_size = 0
    for param in module.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in module.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return param_size + buffer_size


def profile_size(model, batch_size=1, prompt_len=1024):
    logging.info(">>> Profiling model size")
    max_length = prompt_len + 1
    device = next(iter(model.parameters())).device
    inf_cache = model.allocate_inference_cache(batch_size, max_length)
    lengths_per_sample = torch.full((batch_size,), prompt_len, dtype=torch.int32, device=device)
    inference_params = InferenceParams(
        max_seqlen=max_length,
        max_batch_size=batch_size,
        seqlen_offset=prompt_len, # set the model to generation mode
        key_value_memory_dict=inf_cache,
        lengths_per_sample=lengths_per_sample,
    )
    logging.info("Start profiling...")
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    conv_state_size = 0
    ssm_state_size = 0
    for _, (conv_state, ssm_state) in inference_params.key_value_memory_dict.items():
        conv_state_size += conv_state.nelement() * conv_state.element_size()
        ssm_state_size += ssm_state.nelement() * ssm_state.element_size()

    op_types = {
        "linear": 0,
        "conv": 0,
        "norm": get_size(model.backbone.norm_f),
        "sscan": 0,
        "embedding": get_size(model.backbone.embedding),
        "output": 0 if model.lm_head.weight is model.backbone.embedding.weight # tie weights
            else get_size(model.lm_head),
    }
    for layer in model.backbone.layers:
        if isinstance(layer, Block):
            if isinstance(layer.mixer, (Mamba2, W4A16QMamba2)):
                op_types["linear"] += (get_size(layer.mixer.in_proj) + get_size(layer.mixer.out_proj))
                op_types["conv"] += get_size(layer.mixer.conv1d)
                op_types["norm"] += get_size(layer.norm) + get_size(layer.mixer.norm)
                op_types["sscan"] += get_size(layer.mixer.A_log) + get_size(layer.mixer.D) + get_size(layer.mixer.dt_bias)
            elif isinstance(layer.mixer, (W4A8QMamba2, W8A8QMamba2)):
                op_types["linear"] += (get_size(layer.mixer.in_proj) + get_size(layer.mixer.out_proj))
                op_types["conv"] += get_size(layer.mixer.conv1d)
                op_types["norm"] += get_size(layer.norm) + get_size(layer.mixer.norm)
                op_types["sscan"] += get_size(layer.mixer.qchunk_scan)
            elif isinstance(layer.mixer, (Mamba, W4A16QMamba)):
                op_types["linear"] += (
                    get_size(layer.mixer.in_proj) +
                    get_size(layer.mixer.x_proj) +
                    get_size(layer.mixer.dt_proj) +
                    get_size(layer.mixer.out_proj)
                )
                op_types["conv"] += get_size(layer.mixer.conv1d)
                op_types["norm"] += get_size(layer.norm)
                op_types["sscan"] += get_size(layer.mixer.A_log) + get_size(layer.mixer.D)
                if hasattr(layer.mixer, "dt_proj_bias"):
                    op_types["sscan"] += get_size(layer.mixer.dt_proj_bias)
            elif isinstance(layer.mixer, (W4A8QMamba, W8A8QMamba)):
                op_types["linear"] += (
                    get_size(layer.mixer.in_proj) +
                    get_size(layer.mixer.x_proj) +
                    get_size(layer.mixer.dt_proj) +
                    get_size(layer.mixer.out_proj)
                )
                op_types["conv"] += get_size(layer.mixer.conv1d)
                op_types["norm"] += get_size(layer.norm)
                op_types["sscan"] += get_size(layer.mixer.selective_scan)
            else:
                raise ValueError(f"Unsupported mixer type: {layer.mixer.__class__}")

    model_mb = (param_size + buffer_size) / 1024**2
    state_mb = (conv_state_size + ssm_state_size) / 1024**2
    logging.info('state size: {:.3f} MB, detailed breakdown:'.format(state_mb))
    logging.info(f"-- conv state: {conv_state_size / 1024**2:.3f} MB")
    logging.info(f"-- ssm state: {ssm_state_size / 1024**2:.3f} MB")
    logging.info('model size: {:.3f} MB, detailed breakdown:'.format(model_mb))
    op_sum_mb = 0
    for k, v in op_types.items():
        logging.info(f"-- {k}: {v / 1024**2:.3f} MB")
        op_sum_mb += v / 1024**2
    assert op_sum_mb == model_mb, f"Model size breakdown does not match the total model size: {op_sum_mb} != {model_mb}"


def profile_ttft(model, batch_size=1, prompt_len=1024, repeats=100, torch_profile=False, torch_profile_dir=""):
    # no graph cache mode for TTFT (prefilling stage)
    logging.info(">>> Profiling TTFT (prefilling stage)")
    max_length = prompt_len + 1
    prompt = torch.randint(low=0, high=50277, size=(batch_size, prompt_len,)).cuda()
    inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
    logging.info(f"Testing (batch_size, prompt_len): ({batch_size}, {prompt_len})")
    logging.info("Warmup...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(prompt, inference_params=inference_params, num_last_tokens=1)
    torch.cuda.synchronize()

    logging.info("Start profiling...")
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True) 
        end   = torch.cuda.Event(enable_timing=True) 
        start.record()
        for _ in range(repeats):
            _ = model(prompt, inference_params=inference_params, num_last_tokens=1)
        end.record()
        torch.cuda.synchronize()
    dur = start.elapsed_time(end)
    logging.info(f"Finished, latency: {dur/repeats:.2f} milliseconds")
    
    if torch_profile:
        logging.info("Run torch profiler...")
        outfile_prefix = f"ttft_prompt_len_{prompt_len}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, dir_name=torch_profile_dir, use_gzip=True, file_prefix=outfile_prefix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=0, warmup=0, active=5) , repeat=1
                for _ in range(5):
                    with record_function("## forward ##"):
                        out = model(prompt, inference_params=inference_params, num_last_tokens=1)
                    prof.step()

def profile_tpot(model, cache_type=torch.int8, batch_size=1, prompt_len=1024, repeats=100, cache_graph=False, torch_profile=False, torch_profile_dir=""):
    logging.info(">>> Profiling TPOT (generation stage)")
    max_length = prompt_len + 1
    device = next(iter(model.parameters())).device
    inf_cache = model.allocate_inference_cache(batch_size, max_length, cache_type)
    lengths_per_sample = torch.full((batch_size,), prompt_len, dtype=torch.int32, device=device)
    inference_params = InferenceParams(
        max_seqlen=max_length,
        max_batch_size=batch_size,
        seqlen_offset=prompt_len, # set the model to generation mode
        key_value_memory_dict=inf_cache,
        lengths_per_sample=lengths_per_sample,
    )

    input_token = torch.randint(low=0, high=50277, size=(batch_size, 1)).cuda() # only input 1 token at a time

    # warmup
    logging.info("Warmup...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        with torch.cuda.stream(s):
            for _ in range(10):
                _ = model(input_token, inference_params=inference_params).logits
    torch.cuda.current_stream().wait_stream(s)

    if cache_graph:
        with torch.no_grad():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = model(input_token, inference_params=inference_params).logits
            
        def generate(new_input_token, new_inference_params):
            input_token.copy_(new_input_token)
            inference_params.lengths_per_sample[:] = new_inference_params.seqlen_offset
            graph.replay()
            return out
    else:
        def generate(new_input_token, new_inference_params):
            out = model(new_input_token, inference_params=new_inference_params).logits
            return out
        
    logging.info("Start profiling...")
    new_input_token = torch.randint(low=0, high=50277, size=(batch_size, 1)).cuda() # only input 1 token at a time
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True) 
        end   = torch.cuda.Event(enable_timing=True) 
        start.record()
        for _ in range(repeats):
            generate(new_input_token, inference_params)
        end.record()
        torch.cuda.synchronize()
    dur = start.elapsed_time(end)
    logging.info(f"Finished, latency: {dur/repeats:.2f} milliseconds (cache_graph={cache_graph})")

    if torch_profile:
        logging.info("Run torch profiler...")
        outfile_prefix = f"tpot"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, dir_name=torch_profile_dir, use_gzip=False, file_prefix=outfile_prefix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=0, warmup=0, active=5) , repeat=1
                for _ in range(5):
                    generate(new_input_token, inference_params)
                    prof.step()

def profile_ttlt(model, cache_type=torch.int8, batch_size=1, prompt_len=1024, gen_len=128, repeats=100, cache_graph=False, torch_profile=False, torch_profile_dir=""):
    logging.info(">>> Profiling TTLT (prefilling + generation)")
    logging.info(f"batch_size: {batch_size}, prompt_len: {prompt_len}, gen_len:{gen_len}")

    # cache the graph for generation
    if cache_graph:
        device = next(iter(model.parameters())).device
        max_length = prompt_len + gen_len
        inf_cache = model.allocate_inference_cache(batch_size, max_length, cache_type)
        lengths_per_sample = torch.full((batch_size,), prompt_len, dtype=torch.int32, device=device)
        inference_params = InferenceParams(
            max_seqlen=max_length,
            max_batch_size=batch_size,
            seqlen_offset=prompt_len, # set the model to generation mode
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        input_token = torch.randint(low=0, high=50277, size=(batch_size, 1)).cuda()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad():
            with torch.cuda.stream(s):
                for _ in range(10):
                    out = model(input_token, inference_params=inference_params).logits
        torch.cuda.current_stream().wait_stream(s)

        with torch.no_grad():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = model(input_token, inference_params=inference_params).logits
        
        def generate(new_input_token, new_inference_params):
            input_token.copy_(new_input_token)
            inference_params.lengths_per_sample[:] = new_inference_params.seqlen_offset
            graph.replay()
            return out
    else:
        def generate(new_input_token, new_inference_params):
            out = model(new_input_token, inference_params=new_inference_params).logits
            return out

    def run(batch_size, prompt_len, gen_len):
        max_length = prompt_len + gen_len
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
        prompt = torch.randint(low=0, high=50277, size=(batch_size, prompt_len)).cuda()
        sequences = [prompt]
        # prefilling
        out = model(sequences[-1], inference_params, num_last_tokens=1)
        inference_params.seqlen_offset += sequences[-1].shape[1]
        sampled_tokens = out.logits.squeeze(dim=1).argmax(dim=-1) # CausalLMOutput
        sampled_tokens = sampled_tokens.unsqueeze(1) # "b -> b 1"
        sequences.append(sampled_tokens)
        # generate
        while inference_params.seqlen_offset < max_length - 1:
            out = generate(sequences[-1], inference_params)
            inference_params.seqlen_offset += sequences[-1].shape[1]
            sampled_tokens = out.squeeze(dim=1).argmax(dim=-1)
            sampled_tokens = sampled_tokens.unsqueeze(1) # "b -> b 1"
            sequences.append(sampled_tokens)

    logging.info("Warmup...")
    with torch.no_grad():
        for _ in range(5):
            run(batch_size, prompt_len, gen_len)

    logging.info("Start profiling...")
    with torch.no_grad():
        start = torch.cuda.Event(enable_timing=True) 
        end   = torch.cuda.Event(enable_timing=True) 
        start.record()
        for _ in range(repeats):
            run(batch_size, prompt_len, gen_len)
        end.record()
        torch.cuda.synchronize()
    dur = start.elapsed_time(end)
    logging.info(f"Finished, latency: {dur/repeats:.2f} milliseconds (cache_graph={cache_graph})")

    if torch_profile:
        logging.info("Run torch profiler...")
        logging.warning("Profile ttlt with torch profiler is very slow...")
        outfile_prefix = f"ttlt_prompt_len_{prompt_len}_gen_len_{gen_len}_cache_graph_{cache_graph}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, dir_name=torch_profile_dir, use_gzip=False, file_prefix=outfile_prefix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=0, warmup=0, active=5) , repeat=1
                for _ in range(5):
                    run(batch_size, prompt_len, gen_len)
                    prof.step()

def main(args):
    device = "cuda"
    dtype = torch.float16

    logging.info(f"Loading {args.model}")
    model_name = args.model.lower().split('/')[-1]
    model_type = model_name.split('-')[0] # Assume that the models name is like "model_type-<model_size, model version>"
    is_mamba = args.model.split("/")[-1].startswith("mamba") # mamba or mamba2
    is_quamba = args.model.split("/")[-1].startswith("quamba") # quamba or quamba2
    # cg_dtype = torch.float16
    # load model
    start = time.time()
    if is_mamba:
        if "mamba2-8b" not in args.model: # for mamba or mamba2
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", resume_download=None)
        else:
            # NOTE(hychiang): Special handle for mamba2-8b's tokenizer from NVIDIA Megatron
            tokenizer_ckpt = os.path.join(args.model, "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model")
            tokenizer = _GPTSentencePieceTokenizer(tokenizer_ckpt)
        model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
        if args.quantize:
            model = quantize_model_mamba(model, model_type, tokenizer, device, args)
            # if "a8" in args.model:
            #     cg_dtype = torch.int8
    elif is_quamba:
        # ut-enyac/quamba-xb-wxax --pretrained_dir pretrained_models
        # ut-enyac/quamba2-xb-wxax --pretrained_dir pretrained_models
        assert args.pretrained_dir, "Please specify the --pretrained_dir for quamba models"
        quantized_model_path = os.path.join(args.pretrained_dir, args.model)
        assert os.path.exists(quantized_model_path), f"Quantized model {quantized_model_path} not found"
        if "quamba2-8b" not in args.model: # for mamba or mamba2
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", resume_download=None)
        else:
            # NOTE(hychiang): Special handle for mamba2-8b's tokenizer from NVIDIA Megatron
            tokenizer_ckpt = os.path.join(args.pretrained_dir, args.model, "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model")
            tokenizer = _GPTSentencePieceTokenizer(tokenizer_ckpt)
        model = QuambaLMHeadModel.from_pretrained(quantized_model_path, device="cuda")
        # if "a8" in args.model:
        #     cg_dtype = torch.int8
    else:
        # TODO: Not sure if this will run correctly for some transformer models
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map={"": device}, torch_dtype=dtype)
        if args.quantize:
            raise ValueError(f"Unsupport quantizing {args.model}, only supports mamba now")
    elaspe_time = time.time() - start
    model.eval()
    logging.info(f"Loading model takes: {elaspe_time:.2f} s")
    # logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_mb = (param_size + buffer_size) / 1024**2
    logging.info(f"model size: {model_mb:.3f} MB")
    logging.info(f"cache CUDA graph: {args.cache_graph}")

    block_name = model.backbone.layers[0].mixer.__class__.__name__
    if args.ttft:
        if args.cache_graph:
            logging.warning("TTFT does not support cache_graph mode, set to False")
        profile_ttft(model, args.batch_size, args.prompt_len, args.repeats, args.torch_profile, f"torch_profile/{model_name}_{block_name}")
    if args.tpot:
        profile_tpot(model, None, args.batch_size, args.prompt_len, args.repeats, args.cache_graph, args.torch_profile, f"torch_profile/{model_name}_{block_name}")
    if args.ttlt:
        profile_ttlt(model, None, args.batch_size, args.prompt_len, args.gen_len, args.repeats, args.cache_graph, args.torch_profile, f"torch_profile/{model_name}_{block_name}")
    if args.size:
        profile_size(model, args.batch_size, args.prompt_len)
    if not args.ttft and not args.tpot and not args.ttlt and not args.size:
        logging.warning("No profiling task to run with, try `--ttft`, `--tpot`, `--ttlt`, `--size`?")

if __name__ =='__main__':    
    # Fix all possible random seef for reproduce
    set_deterministic(1234)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str,
        help='Mamba to load; pass location of hugginface converted checkpoint. (e.g., state-spaces/mamba-130m)'
    )
    parser.add_argument(
        '--repeats', type=int, default=100,
        help='The number of profiling to repeat (default: 100)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='The input batch size to Mamba. (default: 1)'
    )
    parser.add_argument(
        '--prompt_len', type=int, default=1024,
        help='The number of input tokens to Mamba. (default: 1024)'
    )
    parser.add_argument(
        '--gen_len', type=int, default=128,
        help='The number of generation tokens output from Mamba. Only for TTLT. (default: 128)'
    )
    parser.add_argument(
        '--size', action='store_true',
        help='Profile model total size (i.e. parameters + buffers)'
    )
    parser.add_argument(
        '--ttft', action='store_true',
        help='Profile time to first token (TTFT, i.e. prefilling stage)'
    )
    parser.add_argument(
        '--tpot', action='store_true',
        help='Profile time per output token (TPOT) (TPOT, i.e. generation stage)'
    )
    parser.add_argument(
        '--ttlt', action='store_true',
        help='Profile time to last token (TTLT, i.e. total latency: prefilling + generation)'
    )
    parser.add_argument(
        '--cache_graph', action='store_true', default=False,
        help='To enable CUDA graph cache, this only works for the generation stage (TPOT and TTLT)'
    )
    parser.add_argument(
        '--torch_profile', action='store_true',
        help='Whether to launch the pytorch profiler.'
    )
    get_quantize_options(parser)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)3d] %(message)s",
        datefmt="%d/%b/%Y %H:%M:%S",
        stream=sys.stdout)
    main(args)
