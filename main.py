import json
from utils import (
    build_mamba_and_tokenizer, 
    set_deterministic, 
    parse_options
)
import logging
import sys
import os
from quamba.eval_utils import eval_mamba_zero_shot, evaluate_ppl

def main(args):    
    model_name = args.model.lower().split('/')[-1]
    model_type = model_name.split('-')[0] # Assume that the models name is like "model_type-<model_size, model version>"
    assert model_name != None, "Please check the model path."
    logging.info(f"Creating Model:{model_name}")
    model, tokenizer = build_mamba_and_tokenizer(args, model_type)
    model.config.use_cache = False
    logs = {}
    if args.eval_fp16:
        if args.eval_ppl:
            logging.info(f"Evaluating ppl result (fp16), dataset: {args.ppl_dataset}")
            ppl_results = evaluate_ppl(model, tokenizer, model_name, batch_size=args.batch_size, device="cuda", dataset=args.ppl_dataset)
            logs['ppl'] = ppl_results
        if args.eval_zero_shot:
            logging.info(f"Evaluating Result using lm_eval (fp16), task(s): {args.task_list}")
            lm_eval_results = eval_mamba_zero_shot(
                model, tokenizer, 
                model_type=model_type,
                task_list=args.task_list,
                batch_size=args.batch_size,
                fewshot=args.fewshot,
                limit=100 if args.testing else None
            )
            logs['lm_eval'] = lm_eval_results['results']
        if not args.eval_ppl and not args.eval_zero_shot:
            logging.warn("No task to run with, try `--eval_ppl`, `--eval_zero_shot`?")
        if args.log_dir:
            logs['args'] = vars(args)
            logs['quantization_config'] = 'fp16'
            os.makedirs(args.log_dir, exist_ok=True)
            log_paths = os.path.join(args.log_dir, f"{model_name}_fp16.json")
            with open(log_paths, 'a') as fp:
                logging.info(f"Saving result to {log_paths}")
                json.dump(logs, fp, indent=4)
        return

    """
    Start evaluating Quantized Models from here
    """
    logging.info(f"evaluating quantization type: {args.quant_type}")
    if args.quant_type.lower() == "real":
        from quamba.real_quant.modelutils_mamba import quantize_model_mamba as rlq_mamba
        model = rlq_mamba(model, model_type, tokenizer, "cuda", args)
    elif args.quant_type.lower() == "fake":
        from quamba.fake_quant.modelutils_mamba import quantize_model_mamba as faq_mamba
        model = faq_mamba(model, model_type, tokenizer, "cuda", args)
    else:
        logging.error(f"Unrecognized quantization type: {args.quant_type}")
    model.eval()
    
    logs = {}
    if args.eval_ppl:
        logging.info(f"Evaluating ppl result (quantized), dataset: {args.ppl_dataset}")
        ppl_results = evaluate_ppl(model, tokenizer, model_name, batch_size=args.batch_size, device="cuda", dataset=args.ppl_dataset)
        logs['ppl'] = ppl_results
    if args.eval_zero_shot:
        logging.info(f"Evaluating Result using lm_eval (quantized), task(s): {args.task_list}")
        lm_eval_results = eval_mamba_zero_shot(
            model, tokenizer, 
            model_type=model_type,
            task_list=args.task_list, 
            batch_size=args.batch_size,
            limit=100 if args.testing else None
        )
        logs['lm_eval'] = lm_eval_results['results']
    if not args.eval_ppl and not args.eval_zero_shot:
        logging.warn("No task to run with, try `--eval_ppl`, `--eval_zero_shot`?")
        
    if args.log_dir:
        logs['args'] = vars(args)
        os.makedirs(args.log_dir, exist_ok=True)
        log_paths = os.path.join(args.log_dir, f"{model_name}_int8.json")
        with open(log_paths, 'a') as fp:
            logging.info(f"Saving result to {log_paths}")
            json.dump(logs, fp, indent=4)
    

if __name__ =='__main__':    
    set_deterministic(1234)
    args = parse_options()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    main(args)

