from huggingface_hub import HfApi
import time
from loguru import logger
import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def run_job():
    while True:
        # Run the Python script
        api = HfApi(token=HF_TOKEN)
        try:
            api.upload_large_folder(
                folder_path=".",
                repo_id="ShuoChen99/breakoss",
                repo_type="dataset",
                ignore_patterns=[".git/*", ".idea/*", "*.pyc", "__pycache__/*", ".DS_Store", ".venv/*",
                                 "sync.py"],
            )
            api.upload_folder(
                folder_path="logs_bash",
                path_in_repo="logs_bash",
                repo_id="ShuoChen99/breakoss",
                repo_type="dataset",
            )
        except Exception as e:
            logger.error(f"Error during upload: {e}")
            time.sleep(10 * 60)
            continue
        logger.info("Sync completed. Waiting for the next run...")
        time.sleep(10 * 60)

def run_one():
    api = HfApi(token=HF_TOKEN)
    api.upload_large_folder(
        folder_path=".",
        repo_id="ShuoChen99/breakoss",
        repo_type="dataset",
        ignore_patterns=[".git/*", ".idea/*", "*.pyc", "__pycache__/*", ".DS_Store", ".venv/*",
                         "sync.py"],
    )
    api.upload_folder(
        folder_path="logs_bash",
        path_in_repo="logs_bash",
        repo_id="ShuoChen99/breakoss",
        repo_type="dataset",
    )

if __name__ == "__main__":
    run_job()

    # run_one()