#!/usr/bin/env python3
"""
Script to upload a folder to Hugging Face Hub repository, preserving paths.
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_folder


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Upload a folder to Hugging Face Hub repository preserving paths.'
    )
    
    parser.add_argument(
        '-s', '--source',
        required=True,
        type=str,
        help='Source folder path to upload'
    )
    
    parser.add_argument(
        '-r', '--repo',
        required=True,
        type=str,
        help='Repository name on Hugging Face Hub (e.g., username/repo-name)'
    )
    
    parser.add_argument(
        '-d', '--destination',
        type=str,
        default='',
        help='Destination folder within the repository (default: repository root)'
    )
    
    parser.add_argument(
        '-b', '--branch',
        type=str,
        default='main',
        help='Branch to upload to (default: main)'
    )
    
    parser.add_argument(
        '-m', '--message',
        type=str,
        default='Add folder upload',
        help='Commit message'
    )
    
    parser.add_argument(
        '--repo-type',
        type=str,
        default='model',
        choices=['model', 'dataset', 'space'],
        help='Repository type on Hugging Face Hub (default: model)'
    )
    
    parser.add_argument(
        '--token-env',
        type=str,
        default='HF_TOKEN',
        help='Environment variable name storing the Hugging Face token (default: HF_TOKEN)'
    )
    
    parser.add_argument(
        '--create-repo',
        action='store_true',
        help='Create the repository if it does not exist'
    )
    
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create a private repository (only used with --create-repo)'
    )
    
    return parser.parse_args()


def get_hf_token(token_env):
    """Get Hugging Face token from environment variable."""
    token = os.environ.get(token_env)
    if not token:
        print(f"Error: Hugging Face token not found in environment variable '{token_env}'")
        print(f"Please set your token with: export {token_env}=your_token")
        sys.exit(1)
    return token


def upload_to_hub(args, token):
    """Upload folder to Hugging Face Hub."""
    api = HfApi(token=token)
    source_path = Path(args.source)
    
    if not source_path.exists():
        print(f"Error: Source path {source_path} does not exist")
        sys.exit(1)
    
    print(f"Preparing to upload folder: {source_path}")
    print(f"Target repository: {args.repo} ({args.repo_type})")
    print(f"Branch: {args.branch}")
    
    # Check if repo exists and create if needed
    try:
        repo_exists = api.repo_exists(
            repo_id=args.repo,
            repo_type=args.repo_type
        )
        
        if not repo_exists and args.create_repo:
            print(f"Repository {args.repo} does not exist. Creating...")
            api.create_repo(
                repo_id=args.repo,
                repo_type=args.repo_type,
                private=args.private,
                exist_ok=True
            )
        elif not repo_exists:
            print(f"Error: Repository {args.repo} does not exist. Use --create-repo to create it.")
            sys.exit(1)
    except Exception as e:
        print(f"Error checking/creating repository: {e}")
        sys.exit(1)
    
    # Prepare the destination path
    dest_path = args.destination.strip("/")
    
    try:
        # Upload the folder
        uploaded_files = upload_folder(
            folder_path=str(source_path),
            repo_id=args.repo,
            repo_type=args.repo_type,
            path_in_repo=dest_path,
            commit_message=args.message,
            revision=args.branch,
            token=token
        )
        
        print(f"Successfully uploaded {len(uploaded_files)} files to {args.repo}")
        for i, file in enumerate(uploaded_files[:5]):
            print(f"  - {file}")
        
        if len(uploaded_files) > 5:
            print(f"  ... and {len(uploaded_files) - 5} more files")
        
        return True
    
    except Exception as e:
        print(f"Error uploading files: {e}")
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    # Get token from environment variable
    token = get_hf_token(args.token_env)
    
    # Upload to Hugging Face Hub
    if upload_to_hub(args, token):
        print("Upload completed successfully")
    else:
        print("Failed to upload files to Hugging Face Hub")
        sys.exit(1)


if __name__ == "__main__":
    main()