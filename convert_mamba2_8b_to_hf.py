import os
import shutil
import logging
import argparse

import torch
from transformers import T5Tokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.hf import load_config_hf

def main(args):
    # Step 1: Load and convert state_dict
    source_state_dict_path = args.source_state_dict_path
    
    # Load source state_dict
    source_state_dict = torch.load(source_state_dict_path, map_location='cpu', weights_only=False)['model']
    target_state_dict = {}
    
    # Define specific mappings for layers
    specific_mappings = {
        "embedding.word_embeddings.weight": "backbone.embedding.weight",
        "decoder.final_norm.weight": "backbone.norm_f.weight",
        "output_layer.weight": "lm_head.weight"
    }
    
    logging.info("Start converting state_dict...")
    
    # Convert layers dynamically for layers 0-55
    for layer_idx in range(56):
        source_prefix = f'decoder.layers.{layer_idx}'
        target_prefix = f'backbone.layers.{layer_idx}'
        
        for source_key, value in source_state_dict.items():
            if source_key.startswith(source_prefix):
                target_key = source_key.replace(source_prefix, target_prefix)
                target_state_dict[target_key] = value

    # Apply specific mappings
    for source_key, target_key in specific_mappings.items():
        if source_key in source_state_dict:
            target_state_dict[target_key] = source_state_dict[source_key]

    logging.info("State dicionary conversion complete.")
    
    logging.info("Try loading the converted state_dict...")
    # Step 2: Load model config and initialize model with converted state_dict
    config_data = load_config_hf("state-spaces/mamba2-2.7b")
    config_data['ssm_cfg']['ngroups'] = 8
    config_data['ssm_cfg']['chunk_size'] = 128
    config_data['d_model'] = 4096
    config_data['n_layer'] = 56
    config_data['tie_embeddings'] = False
    config_data['vocab_size'] = 256000
    config_data['pad_vocab_size_multiple'] = 128
    config = MambaConfig(**config_data)
    
    # Initialize the model
    model = MambaLMHeadModel(config, device='cuda', dtype=torch.float16)
    incompatible_key = model.load_state_dict(target_state_dict)
    logging.info(incompatible_key)
    logging.info("Model loaded successfully.")
    # Step 3: Load tokenizer and save model and tokenizer
    vocab_file = args.vocab_file_path
    tokenizer = T5Tokenizer(vocab_file=vocab_file)

    # Save the model and tokenizer for future use
    model_save_path = args.model_save_path
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    # Copy the original vocal_file for back up
    vocab_filename = os.path.basename(vocab_file)
    shutil.copyfile(vocab_file, os.path.join(args.model_save_path, vocab_filename))


    logging.info("Conversion, loading, and saving complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert and load state_dict for MambaLMHeadModel")
    parser.add_argument('source_state_dict_path', type=str, help="Path to the source state_dict file")
    parser.add_argument("vocab_file_path", type=str, help="Path to the vocab file")
    parser.add_argument('--model_save_path', type=str, default='./mamba2-8b-3t-4k_converted',
                        help="Path to save the converted model and tokenizer (default: ./mamba2-8b-3t-4k_converted)")    
    args = parser.parse_args()
    main(args)