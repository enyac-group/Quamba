import argparse
import random
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def build_mamba_and_tokenizer(args, model_type="mamba"):
    device = "cuda"
    dtype = torch.float16 # use half, otherwise real quant won't run
    if model_type == "jamba":
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if args.eval_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model,
                                            trust_remote_code=True,
                                            torch_dtype=torch.bfloat16,
                                            attn_implementation="flash_attention_2", # pip install flash-attn --no-build-isolation
                                            device_map="auto")
        else:
            # BitsAndBytes is slower than fp16
            quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                            llm_int8_skip_modules=args.skip_modules)
            model = AutoModelForCausalLM.from_pretrained(args.model,
                                            trust_remote_code=True,
                                            torch_dtype=torch.bfloat16,
                                            device_map="auto",
                                            # llm_int8_enable_fp32_cpu_offload=True,
                                            quantization_config=quantization_config)
    elif model_type == "mamba":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", resume_download=None)
        model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'jamba'")
    return model, tokenizer



def set_deterministic(seed):
    # Fix all possible random seef for reproduce
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str,
        help='Mamba to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'quant_type', choices=['real', 'fake'],
        help='Real quant for fake quant (Options: real, fake)'
    )
    parser.add_argument(
        '--do_calibrate', action='store_true', default=False,
        help='Whether to calibrate the model'
    )
    parser.add_argument(
        '--calib_data_num', type=int, default=512,
        help='Number of calibration data'
    )
    parser.add_argument(
        '--act_scales_cache', type=str, 
        help='The pre-calibrated activaction scaling factors for static quant.'
            'Performing daynamic quant if not provided. (default: None)'
    )
    parser.add_argument(
        '--do_smoothing', action='store_true', default=False,
        help='Whether to smooth the model (smoothQuant)'
    )
    parser.add_argument(
        '--do_hadamard', action='store_true', default=False,
        help='Whether to apply hadamard transform (hadQuant)'
    )
    parser.add_argument(
        '--do_percentile_u', action='store_true', default=False,
        help='Whether to use percentile_u for calibrating SSMs inputs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--task_list', type=lambda s: [item for item in s.split(',')], default=["lambada_openai"],
        help='Task to be evaled, e.g., --task_list lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande'
    )
    parser.add_argument(
        '--skip_modules', type=lambda s: [item for item in s.split(',')], default=["mamba"],
        help='llm.int8 modules to skip. Make sure to skip self_attn if you are quantizing it with our setup!'
    )
    parser.add_argument(
        '--fewshot', type=int, default=0,
        help='Number of shots for few-shot evaluation (0 for zero-shot)'
    )
    parser.add_argument(
        '--eval_fp16', action='store_true',
        help='Whether to evaluate the performance of fp16 unquantized model.'
    )
    parser.add_argument(
        '--testing', action='store_true',
        help='testing with decreased sample count'
    )
    parser.add_argument(
        '--log_dir', type=str,
        help='path to the json log file storing the result of lm_evan and quantization settingarg'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Whether to print the debug level information'
    )
    parser.add_argument(
        '--eval_ppl', action='store_true', default=False,
        help='Whether to evaluate the wikitext2 ppl'
    )
    parser.add_argument(
        '--ppl_dataset', type=str, default='wikitext2',
        help='Dataset for ppl evaluation'
    )
    parser.add_argument(
        '--eval_zero_shot', action='store_true', default=False,
        help='Whether to evaluate the zero-shot performance'
    )
    args = parser.parse_args()
    return args
