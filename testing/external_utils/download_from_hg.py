# SPRINGLab/IndicTTS-Hindi

# download this dataset from huggingface : chunk it and download it 

repo_name = "SPRINGLab/IndicTTS-Hindi"

#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['TORCH_DISABLE_LOAD_GLOBAL_DEPS'] = '1'

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download, login
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub", "hf_transfer", "-q"])
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download, login


def get_local_files(local_dir: Path) -> set:
    if not local_dir.exists():
        return set()
    
    local_files = set()
    for f in local_dir.rglob('*'):
        if f.is_file() and not f.name.startswith('.'):
            rel_path = str(f.relative_to(local_dir))
            local_files.add(rel_path)
    return local_files


def get_remote_files(api: HfApi, repo_id: str, repo_type: str) -> list:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        return [f for f in files if not f.startswith('.')]
    except Exception as e:
        print(f"Error fetching remote files: {e}")
        return []


def download_dataset(
    repo_id: str,
    output_dir: str,
    repo_type: str = "model",
    token: str = None,
    skip_existing: bool = True,
    max_workers: int = 4
):
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Hugging Face Dataset Download")
    print(f"=" * 60)
    print(f"Repository: {repo_id}")
    print(f"Repo type: {repo_type}")
    print(f"Output path: {output_path}")
    print(f"Skip existing: {skip_existing}")
    print(f"Max workers: {max_workers}")
    print(f"=" * 60)
    print()
    
    if token:
        print("Logging in with provided token...")
        login(token=token)
    else:
        print("Using cached credentials or logging in interactively...")
        try:
            login()
        except Exception:
            print("No cached credentials, continuing without login...")
    
    api = HfApi()
    
    print("Fetching remote file list...")
    remote_files = get_remote_files(api, repo_id, repo_type)
    
    if not remote_files:
        print("No files found in remote repository or access denied.")
        sys.exit(1)
    
    print(f"Found {len(remote_files)} files in remote repository")
    
    files_to_download = remote_files
    
    if skip_existing:
        print("\nChecking for existing local files...")
        local_files = get_local_files(output_path)
        print(f"Found {len(local_files)} local files")
        
        files_to_download = [f for f in remote_files if f not in local_files]
        skipped = len(remote_files) - len(files_to_download)
        
        if skipped > 0:
            print(f"✓ Skipping {skipped} files that already exist locally")
        
        if not files_to_download:
            print("\n✓ All files already exist locally. Nothing to download.")
            return
    
    print(f"\nFiles to download: {len(files_to_download)}")
    
    if len(files_to_download) == len(remote_files):
        print("\nDownloading entire repository...")
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=str(output_path),
                local_dir_use_symlinks=False,
                max_workers=max_workers,
            )
            print(f"\n✓ Download completed successfully!")
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            sys.exit(1)
    else:
        print(f"\nDownloading {len(files_to_download)} new files...")
        
        successful = 0
        failed = 0
        
        for idx, filename in enumerate(files_to_download, 1):
            print(f"[{idx}/{len(files_to_download)}] Downloading: {filename}")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False,
                )
                successful += 1
                print(f"  ✓ Done")
            except Exception as e:
                failed += 1
                print(f"  ✗ Failed: {e}")
        
        print(f"\n{'='*60}")
        print(f"Download Summary")
        print(f"{'='*60}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Skipped (existing): {len(remote_files) - len(files_to_download)}")
        print(f"{'='*60}")
    
    if repo_type == "model":
        print(f"\nDataset location: {output_path}")
        print(f"View repo at: https://huggingface.co/{repo_id}")
    else:
        print(f"\nDataset location: {output_path}")
        print(f"View repo at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset from Hugging Face Hub with incremental sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download dataset to default location
  python download_hugging_face_dataset.py --repo-id username/dataset-name
  
  # Download to specific directory
  python download_hugging_face_dataset.py --repo-id username/dataset-name -o ./my_data
  
  # Force re-download all files (don't skip existing)
  python download_hugging_face_dataset.py --repo-id username/dataset-name --no-skip
  
  # Download from a model repo type
  python download_hugging_face_dataset.py --repo-id username/model-name --repo-type model
  
  # Download with more workers for faster download
  python download_hugging_face_dataset.py --repo-id username/dataset-name --workers 8
        """
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: ./data/<repo_name>)"
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["dataset", "model", "space"],
        help="Type of repository (default: model)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing files, re-download everything"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of download workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    if args.output:
        output_dir = args.output
    else:
        repo_name = args.repo_id.replace("/", "_")
        output_dir = f"./data/{repo_name}"
    
    download_dataset(
        repo_id=args.repo_id,
        output_dir=output_dir,
        repo_type=args.repo_type,
        token=args.token,
        skip_existing=not args.no_skip,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()


# python download_from_hg.py --repo-id SPRINGLab/IndicTTS-Hindi -o /Users/mohitdulani/Desktop/personal/audio-models/Qwen3-TTS/data --token  --repo-type dataset
