import os
import sys
import time
import logging
import argparse

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import TextStreamer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from utils import get_quantize_options
from quamba.modelutils_mamba import quantize_model_mamba
from quamba.megatron_utils import _GPTSentencePieceTokenizer
from quamba.quamba_mixer_seq import QuambaLMHeadModel


def main(args):

    device = "cuda"
    dtype = torch.float16

    logging.info(f"Loading {args.model}")
    model_name = args.model.lower().split('/')[-1]
    model_type = model_name.split('-')[0] # Assume that the models name is like "model_type-<model_size, model version>"
    is_mamba = args.model.split("/")[-1].startswith("mamba") # mamba or mamba2
    is_quamba = args.model.split("/")[-1].startswith("quamba") # quamba or quamba2
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
    else:
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
    logging.info('model size: {:.3f} MB'.format(model_mb))


    torch.random.manual_seed(0)
    if args.prompt is None:
        input_ids = torch.randint(1, 1000, (args.batch_size, args.promptlen), dtype=torch.long, device="cuda")
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
    else:
        tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        attn_mask = tokens.attention_mask.to(device=device)
 
    max_length = input_ids.shape[1] + args.genlen
 
    # addtional generate arguments for mamba
    model_kwargs = {}
    if is_mamba:
        model_kwargs = {
            "cg": args.cache_graph,
        }
    
    if args.streaming:
        if args.benchmark:
            logging.warning("Unsupport benchmarking with streaming mode")
        if args.prompt is not None:
            logging.info(f"Input prompt: {tokenizer.batch_decode(input_ids.tolist())[0]}")
            logging.info(f"Input prompt token length: {input_ids.shape[-1]}")
        # init streamer from the tokenizer
        streamer = TextStreamer(tokenizer, skip_prompt=False)
        # generate function
        fn = lambda: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            min_p=args.minp,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            **model_kwargs,
        )
        out = fn()

    else:
        fn = lambda: model.generate(
            input_ids=input_ids,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            min_p=args.minp,
            repetition_penalty=args.repetition_penalty,
            **model_kwargs,
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
    parser.add_argument('--prompt', type=str,
        default="My cat wrote all this CUDA code for a new language model and ",
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
        '--streaming', action='store_true', default=False,
        help='enable streaming mode on stdout'
    )
    parser.add_argument(
        '--benchmark', action='store_true', default=False,
        help='To benchmark the latency'
    )
    get_quantize_options(parser)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)3d] %(message)s",
        datefmt="%d/%b/%Y %H:%M:%S",
        stream=sys.stdout)
    main(args)
