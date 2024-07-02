# the code is adapted from: https://github.com/redotvideo/mamba-chat/blob/main/chat.py
import os
import sys
import time
import logging
from functools import partial

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import TextStreamer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from quamba.real_quant.modelutils_mamba import quantize_blocks, run_calibration

def preprocess(conversation, tokenizer, conversation_template, max_tokens, device):
    """
    Preprocess the data by tokenizing.
    """
    all_input_ids = []
    all_label_ids = []
    tokenizer.use_default_system_prompt = False
    messages = conversation["messages"]
    tokenized_messages = tokenizer.apply_chat_template(messages, chat_template=conversation_template, max_length=max_tokens, truncation=True)
    input_ids = torch.LongTensor([tokenized_messages]).to(device) # expand dim
    return input_ids

def main(args):

    device = "cuda"
    dtype = torch.float16

    logging.info(f"Loading {args.model}")
    is_mamba = args.model.split("/")[-1].startswith("mamba-")
    if not is_mamba:
        raise ValueError("Not support other models now")

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template
    # init streamer from the tokenizer
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    # load model and quantize it
    start = time.time()
    model = MambaLMHeadModel.from_pretrained(args.model, device="cuda", dtype=dtype)
    elaspe_time = time.time() - start
    logging.info(f"Loading model takes: {elaspe_time:.2f} s")
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.quantize:
        if os.path.isfile(args.act_scales_cache):
            logging.info(f"Found activation scales cache {args.act_scales_cache}")
            act_scales = torch.load(args.act_scales_cache)
        else:
            calibration_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")
            calibration_dataset.shuffle(seed=42)
            preprocess_fn = partial(preprocess, tokenizer=tokenizer,
                                conversation_template=tokenizer.chat_template,
                                max_tokens=1024, device=device)
            act_scales = run_calibration(model, "mamba", tokenizer, seq_len=1024,
                            calibration_dataset=calibration_dataset,
                            preprocess_fn=preprocess_fn)
            
            if args.act_scales_cache:
                logging.info(f"Store activation scales at {args.act_scales_cache}")
                torch.save(act_scales, args.act_scales_cache)
        # quantization
        logging.info("Start quantizing model...")
        model = quantize_blocks(model, "mamba", act_scales, device)
        model.eval()
    
    # generate function
    generate_fn = partial(model.generate,
        max_length=256,
        cg=args.cache_graph,
        cg_dtype=torch.int8 if args.quantize else torch.float16,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )



    if args.use_testing_prompts:
        logging.info("Test the model with testing prompts...")
        # test with prompts
        messages = []
        testing_prompt = [
            "Create a YouTube tutorial on how to bake a gluten-free cake.",
            "Can you provide me with some tips to make sure my gluten-free cake turns out perfect?",
            "Can you add some tips on how to make a gluten-free cake without eggs? Or maybe, can you suggest some frosting options that are also gluten-free?",
            "Hey there, thanks for the tips on making a gluten-free cake without eggs and the gluten-free frosting options. Do you have any suggestions on how to make a vegan and gluten-free cake? And what about some garnishing ideas that are both gluten-free and vegan?",
        ]

        for prompt in testing_prompt:
            print("\nYour message: ", prompt)
            messages.append(dict(
                role="user",
                content=prompt
            ))
            print("Model:\n")
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
            generate_fn(input_ids=input_ids)
    else:
        logging.info("Start chatting with the model...")
        # start chatting
        messages = []
        while True:
            user_message = input("\nYour message: ")
            messages.append(dict(
                role="user",
                content=user_message
            ))
            print("Model:\n")
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
            generate_fn(input_ids=input_ids)

if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Chat with mamba')
    parser.add_argument(
        'model', type=str, default='havenhq/mamba-chat', nargs='?',
        help='Mamba to load; pass location of hugginface converted checkpoint. (default: havenhq/mamba-chat)'
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
        '--repetition_penalty', type=float, default=1.2,
    )
    parser.add_argument(
        '--cache_graph', action='store_true', default=False,
    )
    parser.add_argument(
        '--use_testing_prompts', action='store_true', default=False,
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
