# Copyright (c) 2023, Tri Dao, Albert Gu.
import os
import sys
import logging
import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from quamba.real_quant.modelutils_mamba import quantize_blocks, run_calibration


def main(args):

    device = "cuda"
    dtype = torch.float16

    logging.info(f"Loading {args.model}")
    is_mamba = args.model.split("/")[-1].startswith("mamba-")
    if not is_mamba:
        raise ValueError("Not support other models now")
    
    # load model
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
    model.eval()
    elaspe_time = time.time() - start
    logging.info(f"Loading model takes: {elaspe_time:.2f} s")
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.quantize:
        if os.path.isfile(args.act_scales_cache):
            logging.info(f"Found activation scales cache {args.act_scales_cache}")
            act_scales = torch.load(args.act_scales_cache)
        else:
            act_scales = run_calibration(model, "mamba", tokenizer)
            
            if args.act_scales_cache:
                print(f"Store activation scales at {args.act_scales_cache}")
                torch.save(act_scales, args.act_scales_cache)
        
        # quantization
        logging.info("Start quantizing model...")
        model = quantize_blocks(model, "mamba", act_scales, "cuda")
        model.eval()

    torch.random.manual_seed(0)
    if args.prompt is None:
        input_ids = torch.randint(1, 1000, (args.batch_size, args.promptlen), dtype=torch.long, device="cuda")
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
    else:
        tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        attn_mask = tokens.attention_mask.to(device=device)
    max_length = input_ids.shape[1] + args.genlen
 
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=args.cache_graph,
        cg_dtype=torch.int8 if args.quantize else torch.float16,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
    )
    out = fn()
    if args.prompt is not None:
        logging.info(tokenizer.batch_decode(out.sequences.tolist())[0])
 
    if args.benchmark:
        repeats = 100
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        logging.info(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
        logging.info(f"{args.model} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")

if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate from mamba")
    parser.add_argument(
        'model', type=str, default="state-spaces/mamba-130m",
        help='Mamba to load; pass location of hugginface converted checkpoint. (default: state-spaces/mamba-130m)'
    )
    parser.add_argument('--prompt', type=str, default=None,
        help='input prompt'
    )
    parser.add_argument(
        '--promptlen', type=int, default=100,
    )
    parser.add_argument(
        '--genlen', type=int, default=100,
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
    )
    parser.add_argument(
        '--topk', type=int, default=1,
    )
    parser.add_argument(
        '--topp', type=float, default=1.0,
    )
    parser.add_argument(
        '--minp', type=float, default=0.0,
    )
    parser.add_argument(
        '--repetition_penalty', type=float, default=1.0,
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
    )
    parser.add_argument(
        '--cache_graph', action='store_true', default=False,
    )
    parser.add_argument(
        '--benchmark', action='store_true', default=False,
        help='To benchmark the latency'
    )
    # quantization parameters
    parser.add_argument(
        '--quantize', action='store_true', default=False,
    )
    parser.add_argument(
        '--act_scales_cache', type=str, 
        help='The pre-calibrated activaction scaling factors for static quant.'
            'Performing daynamic quant if not provided. (default: None)'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)3d] %(message)s",
        datefmt="%d/%b/%Y %H:%M:%S",
        stream=sys.stdout)
    main(args)
