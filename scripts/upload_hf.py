#!/usr/bin/env python3
"""
Upload trained DFlash Speculative Drafter to HuggingFace.
"""

import os
import sys
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("HF_REPO_ID", "jfan/Qwen3.8-27B-heretic-dflash")
MODEL_DIR = os.environ.get("MODEL_DIR", "/output/Qwen3.8-27B-heretic-dflash")

if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable is required.")
    sys.exit(1)

def main():
    print(f"Connecting to Hugging Face with token for repo: {REPO_ID}...")
    api = HfApi(token=HF_TOKEN)
    
    print(f"Ensuring repository {REPO_ID} exists...")
    create_repo(repo_id=REPO_ID, token=HF_TOKEN, repo_type="model", exist_ok=True)
    
    print(f"Uploading files from {MODEL_DIR} to https://huggingface.co/{REPO_ID}...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        token=HF_TOKEN
    )
    print(f"Successfully uploaded DFlash drafter to https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
