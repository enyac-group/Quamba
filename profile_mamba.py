import sys
import logging
from functools import partial

import torch
from transformers import AutoTokenizer

# mamba
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams

from utils import set_deterministic
from quamba.real_quant.modelutils_mamba import quantize_blocks, run_calibration

import argparse
import os

import socket
from datetime import datetime
from torch.autograd.profiler import record_function

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def trace_handler(prof: torch.profiler.profile, file_postfix="prefilling", device="cuda:0"):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}_{file_postfix}.json.gz")

   # Construct the memory timeline file.
   # !!! This does not work for graph cache !!!
   prof.export_memory_timeline(f"{file_prefix}_{file_postfix}.html", device=device)

def build_mamba(args):
    device = "cuda:0"
    dtype = torch.float16

    model_name = args.model.lower().split('/')[-1]
    assert model_name != None, "Please check the model path."
    logging.info(f"Creating Model:{model_name}, dtype: {dtype}, device: {device}")
    rms_norm = True
    fused_add_norm = True
    if args.disable_triton:
        rms_norm = False
        fused_add_norm = False
    if model_name == "mamba-2.8b":
        """mamba-2.8b config"""
        d_state = 16 # default
        cfg = MambaConfig(
            d_model=2560,
            n_layer=64,
            vocab_size=50277,
            ssm_cfg={},
            rms_norm=rms_norm,
            residual_in_fp32=True,
            fused_add_norm=fused_add_norm, # 72.98 ms
            pad_vocab_size_multiple=8
        )
    elif model_name == "mamba-1.4b":
        """mamba-1.4b config"""
        cfg = MambaConfig(
            d_model=2048,
            n_layer=48,
            vocab_size=50277,
            ssm_cfg={},
            rms_norm=rms_norm,
            residual_in_fp32=True,
            fused_add_norm=fused_add_norm,
            pad_vocab_size_multiple=8
        )
    elif model_name == "mamba-790m":
        cfg = MambaConfig(
            d_model=1536,
            n_layer=48,
            vocab_size=50277,
            ssm_cfg={},
            rms_norm=rms_norm,
            residual_in_fp32=True,
            fused_add_norm=fused_add_norm,
            pad_vocab_size_multiple=8
        )
    elif model_name == "mamba-370m":
        """mamba-370m config"""
        d_state = 16 # default
        cfg = MambaConfig(
            d_model=1024,
            n_layer=48,
            vocab_size=50277,
            ssm_cfg={},
            rms_norm=rms_norm,
            residual_in_fp32=True,
            fused_add_norm=fused_add_norm,
            pad_vocab_size_multiple=8
        )
    elif model_name == "mamba-130m":
        """mamba-130m config"""
        d_state = 16 # default
        cfg = MambaConfig(
            d_model=768,
            n_layer=24,
            vocab_size=50277,
            ssm_cfg={},
            rms_norm=rms_norm,
            residual_in_fp32=True,
            fused_add_norm=fused_add_norm,
            pad_vocab_size_multiple=8
        )
    else:
        raise ValueError(f"Unrecognized model: {model_name}")
    model = MambaLMHeadModel(cfg, device=device, dtype=dtype)
    return model, "mamba"

def profile_size(model, batch_size=1, prompt_len=1024):
    logging.info(">>> Profiling model size")
    max_length = prompt_len + 1
    input_ids = torch.randint(low=0, high=50277, size=(batch_size, prompt_len,)).cuda()
    inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)
    logging.info("Warmup...")
    with torch.no_grad():
        # to initialize conv_state and ssm_state
        for i in range(5):
            out = model(input_ids, inference_params=inference_params)
    torch.cuda.synchronize()

    logging.info("Start profiling...")
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    state_size = 0
    for _, (conv_state, ssm_state) in inference_params.key_value_memory_dict.items():
        state_size += conv_state.nelement() * conv_state.element_size()
        state_size += ssm_state.nelement() * conv_state.element_size()

    model_mb = (param_size + buffer_size) / 1024**2
    state_mb = (state_size) / 1024**2
    logging.info('model size: {:.3f} MB'.format(model_mb))
    logging.info('state size: {:.3f} MB'.format(state_mb))


def profile_ttft(model, batch_size=1, prompt_len=1024, repeats=100, torch_profile=False, outfile=""):
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
        outfile_postfix = f"{outfile}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, file_postfix=outfile_postfix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=1, warmup=5, active=6) , repeat=1
                for _ in range(12):
                    with record_function("## forward ##"):
                        out = model(prompt, inference_params=inference_params, num_last_tokens=1)
                    prof.step()

def profile_tpot(model, cache_type=torch.int8, batch_size=1, prompt_len=1024, repeats=100, cache_graph=False, torch_profile=False, outfile=""):
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
        outfile_postfix = f"{outfile}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, file_postfix=outfile_postfix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=1, warmup=5, active=6) , repeat=1
                for _ in range(12):
                    generate(new_input_token, inference_params)
                    prof.step()

def profile_ttlt(model, cache_type=torch.int8, batch_size=1, prompt_len=1024, gen_len=128, repeats=100, cache_graph=False, torch_profile=False, outfile=""):
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
        logging.warn("Profile ttlt with torch profiler is slow")
        outfile_postfix = f"{outfile}"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(
                trace_handler, file_postfix=outfile_postfix, device="cuda:0"
            )
        ) as prof:

            with torch.no_grad():
                # (wait=1, warmup=5, active=6) , repeat=1
                for _ in range(12):
                    run(batch_size, prompt_len, gen_len)
                    prof.step()


def main(args):    
    model, model_type = build_mamba(args)
    model.eval()
    model.config.use_cache = False
    
    if args.eval_fp16:
        if args.ttft:
            profile_ttft(model, args.batch_size, args.prompt_len, args.repeats, args.torch_profile, "ttft_fp16")
        if args.tpot:
            profile_tpot(model, torch.float16, args.batch_size, args.prompt_len, args.repeats, args.cache_graph, args.torch_profile, "tpot_fp16")
        if args.ttlt:
            profile_ttlt(model, torch.float16, args.batch_size, args.prompt_len, args.gen_len, args.repeats, args.cache_graph, args.torch_profile, "ttlt_fp16")
        if args.size:
            profile_size(model, args.batch_size, args.prompt_len)
    else:
        if not os.path.isfile(args.act_scales_cache):
            logging.warning(f"Not found {args.act_scales_cache}, get scaling factors from the random initialized model")
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", resume_download=None)
            act_scales = run_calibration(model, "mamba", tokenizer)
        else:
            act_scales = torch.load(args.act_scales_cache)
        # Replace module with quantized version
        model = quantize_blocks(model, model_type, act_scales, "cuda")
        model.eval()
            
        if args.ttft:
            profile_ttft(model, args.batch_size, args.prompt_len, args.repeats, args.torch_profile, "ttft_int8")
        if args.tpot:
            profile_tpot(model, torch.int8, args.batch_size, args.prompt_len, args.repeats, args.cache_graph, args.torch_profile, "tpot_int8")
        if args.ttlt:
            profile_ttlt(model, torch.int8, args.batch_size, args.prompt_len, args.gen_len, args.repeats, args.cache_graph, args.torch_profile, "ttlt_int8")
        if args.size:
            profile_size(model, args.batch_size, args.prompt_len)
    
    if not args.ttft and not args.tpot and not args.ttlt and not args.size:
        logging.warn("No profiling task to run with, try `--ttft`, `--tpot`, `--ttlt`, `--size`?")

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
        '--act_scales_cache', type=str, 
        help='The pre-calibrated activaction scaling factors for static quant.'
            'Performing daynamic quant if not provided. (default: None)'
    )
    parser.add_argument(
        '--eval_fp16', action='store_true',
        help='Whether to evaluate the performance of fp16 unquantized model.'
    )
    parser.add_argument(
        '--disable_triton', action='store_true',
        help='Whether to disable Triton. This will set rmsnorm and use_add_norm to False for Mamba fp16.'
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
    
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)3d] %(message)s",
        datefmt="%d/%b/%Y %H:%M:%S",
        stream=sys.stdout)
    main(args)
