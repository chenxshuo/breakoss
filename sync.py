from huggingface_hub import HfApi
import time
from loguru import logger
import os
import fire
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
                                 "sync.py", ".env", "*.swp"],
            )
            api.upload_folder(
                folder_path="logs_bash",
                path_in_repo="logs_bash",
                repo_id="ShuoChen99/breakoss",
                repo_type="dataset",
                ignore_patterns=["*.swp"],
            )
        except Exception as e:
            logger.error(f"Error during upload: {e}")
            if ".swp" in str(e):
                logger.error("Detected .swp file issue, retrying...")
            else:
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

def download():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="ShuoChen99/breakoss", local_dir=".", token=HF_TOKEN, repo_type="dataset")


import os
import shutil


def merge_folders(folder1: str, folder2: str):
    """
    Recursively merge folder1 into folder2:
    - If an item exists in folder1 but not in folder2, copy/move it to folder2.
    - If an item exists in both, recurse into subfolders (if directories).
    """
    if not os.path.exists(folder1):
        raise ValueError(f"{folder1} does not exist")
    if not os.path.exists(folder2):
        os.makedirs(folder2)

    for item in os.listdir(folder1):
        src_path = os.path.join(folder1, item)
        dst_path = os.path.join(folder2, item)

        if os.path.isdir(src_path):
            if not os.path.exists(dst_path):
                # 整个目录直接拷贝
                shutil.copytree(src_path, dst_path)
                print(f"Copied directory {src_path} -> {dst_path}")
            else:
                # 两边都有目录 -> 递归
                merge_folders(src_path, dst_path)
        else:
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied file {src_path} -> {dst_path}")
            else:
                # 两边都有同名文件 -> 什么也不做
                pass


if __name__ == "__main__":
    # fire.Fire()
    run_job()

    # run_one()