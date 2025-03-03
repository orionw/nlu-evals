#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import tempfile
import time
import logging
import atexit
import glob
from pathlib import Path
from typing import List, Optional, Set, Dict
import torch
from tqdm import tqdm
import requests
from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_folder, list_repo_files
from composer.models import write_huggingface_pretrained_from_composer_checkpoint

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_temp_dirs():
    """Clean up temporary directories created during script execution."""
    patterns = [
        "temp-*",
        "tokenizer-save-dir-*",
    ]
    
    for pattern in patterns:
        for dir_path in glob.glob(pattern):
            if os.path.isdir(dir_path):
                try:
                    logger.info(f"Cleaning up temporary directory: {dir_path}")
                    shutil.rmtree(dir_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary directory {dir_path}: {str(e)}")

# Register cleanup function to run on script exit
atexit.register(cleanup_temp_dirs)

def get_modernbert_config(size: str) -> dict:
    """Get the ModernBERT config for a given size."""
    if size not in ["base", "large"]:
        raise ValueError(f"Size {size} not supported. Use 'base' or 'large'.")
    
    repo_name = f"answerdotai/ModernBERT-{size}"
    api = HfApi()
    try:
        # Download the config file
        logger.info(f"Downloading config from {repo_name}")
        config_path = hf_hub_download(repo_id=repo_name, filename="config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to get config from {repo_name}: {str(e)}")
        raise

def update_config(config_path: str, size: str):
    """Update the config file with ModernBERT settings while preserving specific fields."""
    try:
        # Load the current config to get values we want to preserve
        logger.info("Loading current checkpoint config...")
        with open(config_path, 'r') as f:
            current_config = json.load(f)
        
        # Get the ModernBERT config as our base
        logger.info("Getting ModernBERT config as base...")
        modernbert_config = get_modernbert_config(size)
        
        # Define field mappings from old to new config names
        field_mappings = {
            'rotary_emb_base': 'global_rope_theta',
            'num_hidden_layers': 'num_hidden_layers',
            'hidden_size': 'hidden_size',
            'intermediate_size': 'intermediate_size',
            'num_attention_heads': 'num_attention_heads',
            'vocab_size': 'vocab_size'
        }
        
        # Update ModernBERT config with our preserved fields
        for old_field, new_field in tqdm(field_mappings.items(), desc="Updating config fields"):
            if old_field in current_config:
                old_value = modernbert_config.get(new_field, None)
                modernbert_config[new_field] = current_config[old_field]
                logger.info(f"Preserved {old_field} -> {new_field}: {old_value} -> {current_config[old_field]}")
        
        # Set fixed values
        modernbert_config['max_position_embeddings'] = 8192
        modernbert_config['bos_token_id'] = 2
        modernbert_config['mask_token_id'] = 4
        modernbert_config['pad_token_id'] = 0
        modernbert_config['eos_token_id'] = 1
        modernbert_config['sep_token_id'] = 1
        modernbert_config["cls_token_id"] = 1       
        modernbert_config["position_embedding_type"] = "sans_pos"
        
        # Save the updated config
        with open(config_path, 'w') as f:
            json.dump(modernbert_config, f, indent=2)
            
        logger.info(f"Config updated successfully at {config_path}")
    except Exception as e:
        logger.error(f"Failed to update config: {str(e)}")
        raise

def modify_state_dict(state_dict_path: str):
    """Modify the state dictionary to match ModernBERT architecture."""
    logger.info("Loading state dict...")
    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))

    logger.info("Modifying state dict keys...")
    new_state_dict = {}
    for old_key, tensor in tqdm(state_dict.items(), desc="Processing state dict"):
        # Replace both 'bert.' and 'bert.encoder.' in one go
        new_key = old_key.replace('bert.encoder.', 'model.').replace('bert.', 'model.')
        new_state_dict[new_key] = tensor
        if old_key != new_key:
            logger.debug(f"Renamed: {old_key} -> {new_key}")
    
    # Save the modified state dict
    logger.info("Saving modified state dict...")
    torch.save(new_state_dict, state_dict_path)
    logger.info("State dict modification complete")
    return new_state_dict

def process_checkpoints(input_dir: str, output_dir: str, size: str, tokenizer_path: str = "google/gemma-2b-9b"):
    """Process all checkpoints in the input directory, skipping already processed ones."""
    # Find all checkpoint files
    logger.info(f"Scanning {input_dir} for checkpoints...")
    checkpoint_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.pt'):
            checkpoint_files.append(os.path.join(input_dir, file))
    
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in {input_dir}")
        return
    
    # sort by name
    checkpoint_files.sort(key=lambda x: os.path.basename(x))
    logger.info(f"Found {len(checkpoint_files)} checkpoints to process")
    
    # Get HF token from environment
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required for downloading gated tokenizer files")
    
    # Download tokenizer files from specified path
    logger.info(f"Downloading tokenizer files from {tokenizer_path}")
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    saved_tokenizer_files = {}
    
    try:
        for file_name in tokenizer_files:
            file_content = hf_hub_download(
                repo_id=tokenizer_path,
                filename=file_name,
                token=hf_token
            )
            with open(file_content, 'rb') as f:
                saved_tokenizer_files[file_name] = f.read()
            logger.info(f"Successfully downloaded {file_name}")
    except Exception as e:
        logger.error(f"Failed to download tokenizer files from {tokenizer_path}: {str(e)}")
        raise

    # Create a single temporary directory for all conversions
    with tempfile.TemporaryDirectory() as shared_tmp_dir:
        logger.info(f"Using shared temporary directory: {shared_tmp_dir}")
        
        # Convert first checkpoint to get initial config
        first_checkpoint = checkpoint_files[0]
        logger.info("Converting first checkpoint to get initial config...")
        write_huggingface_pretrained_from_composer_checkpoint(
            first_checkpoint,
            shared_tmp_dir
        )
        
        # Get the original config that we want to preserve values from
        original_config_path = os.path.join(shared_tmp_dir, "config.json")
        with open(original_config_path, 'r') as f:
            original_config = json.load(f)
        
        # Process each checkpoint
        for checkpoint_file in tqdm(checkpoint_files, desc="Processing checkpoints"):
            output_subdir = os.path.join(output_dir, os.path.basename(checkpoint_file).split(".")[0])
            
            # Check if this checkpoint was already processed
            if os.path.exists(output_subdir) and os.path.isfile(os.path.join(output_subdir, "pytorch_model.bin")):
                logger.info(f"Skipping already processed checkpoint: {checkpoint_file}")
                continue
            
            os.makedirs(output_subdir, exist_ok=True)
            
            try:
                # Convert checkpoint
                logger.info(f"Converting checkpoint: {checkpoint_file}")
                write_huggingface_pretrained_from_composer_checkpoint(
                    checkpoint_file,
                    shared_tmp_dir
                )
                
                # Restore original config values and update with ModernBERT base
                with open(original_config_path, 'w') as f:
                    json.dump(original_config, f, indent=2)
                update_config(original_config_path, size)
                
                # Modify state dict
                state_dict_path = os.path.join(shared_tmp_dir, "pytorch_model.bin")
                modify_state_dict(state_dict_path)
                
                # Copy necessary files to output directory
                shutil.copy2(state_dict_path, os.path.join(output_subdir, "pytorch_model.bin"))
                shutil.copy2(original_config_path, os.path.join(output_subdir, "config.json"))
                
                # Restore saved tokenizer files
                for file_name, content in saved_tokenizer_files.items():
                    dst = os.path.join(output_subdir, file_name)
                    with open(dst, 'wb') as f:
                        f.write(content)
                
                logger.info(f"Successfully processed checkpoint {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to process checkpoint {checkpoint_file}: {str(e)}")
                raise

def infer_model_size(input_dir: str) -> str:
    """Infer model size from directory name."""
    dir_name = input_dir.lower()
    if "large" in dir_name:
        return "large"
    return "base"  # default to base if not explicitly large

def main():
    parser = argparse.ArgumentParser(description="Convert checkpoints to HuggingFace format")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing checkpoint files")
    parser.add_argument("--output_dir", type=str, help="Output directory for converted checkpoints (default: input_dir-converted)")
    parser.add_argument("--size", type=str, choices=["base", "large"], help="Model size (base or large, inferred from directory name if not specified)")
    parser.add_argument("--tokenizer_path", type=str, default="google/gemma-2-9b", help="HuggingFace tokenizer path to use (default: google/gemma-2b-9b)")
    args = parser.parse_args()
    
    # Set output_dir to input_dir-converted
    args.output_dir = args.input_dir.rstrip("/") + "-converted"
    
    # Infer size if not specified
    if args.size is None:
        args.size = infer_model_size(args.input_dir)
        logger.info(f"Inferred model size: {args.size}")
    
    try:
        process_checkpoints(args.input_dir, args.output_dir, args.size, args.tokenizer_path)
        logger.info("Conversion complete!")
    except Exception as e:
        logger.error(f"Failed to complete conversion process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    # Example usage:
    # ~/miniconda3/envs/mosiacbert/bin/python /home/oweller2/my_scratch/retrieval_pretraining/bert24/retrieval_pretraining/scripts/convert_all_to_hf_encoder.py --target_repo "orionweller/enc-150m-hf" --source_repo "orionweller/enc-150m"