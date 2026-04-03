#!/usr/bin/env python3
import os
import subprocess
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import HfApi, snapshot_download

REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
CACHE = Path.home() / ".cache/huggingface/hub"
REPO_CACHE = CACHE / f"models--{REPO.replace('/', '--')}"


def du_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    try:
        r = subprocess.run(
            ["du", "-sk", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        return int(r.stdout.split()[0]) * 1024
    except (subprocess.CalledProcessError, ValueError, IndexError, subprocess.TimeoutExpired):
        return 0


def main() -> None:
    api = HfApi()
    total = sum(
        s.size for s in api.model_info(REPO, files_metadata=True).siblings
        if s.size is not None
    )
    if total <= 0:
        total = int(4.52 * 1024**3)
    exp_mib = total / (1024 * 1024)

    err: list[BaseException] = []
    path_holder: dict[str, str | None] = {"p": None}

    def worker() -> None:
        try:
            path_holder["p"] = snapshot_download(REPO)
        except BaseException as e:
            err.append(e)

    print(REPO)
    print(
        f"Expect ~{exp_mib:.0f} MiB. HF tqdm is off (broken in some terminals); "
        f"this uses `du` on the cache folder every 2s.\n"
    )

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        n = du_bytes(REPO_CACHE)
        mib = n / (1024 * 1024)
        pct = 100.0 * n / total if total else 0.0
        print(f"  du cache: {mib:8.1f} MiB  (~{pct:5.1f}% of ~{exp_mib:.0f} MiB)", flush=True)
        time.sleep(2)
    t.join()

    if err:
        raise err[0]
    snap = Path(path_holder["p"] or "")
    ckpt = snap / "model.safetensors"
    incompletes = list((REPO_CACHE / "blobs").glob("*.incomplete")) if (REPO_CACHE / "blobs").is_dir() else []
    print(f"\nDone: {path_holder['p']}")
    if not ckpt.is_file():
        print(
            "\nERROR: snapshot has no model.safetensors — download did not finish.\n"
            f"  incomplete blobs left: {len(incompletes)} (delete them only if you want a clean retry)\n"
            "  Re-run this script and let `du` reach ~100% before stopping the process.\n",
            flush=True,
        )
        raise SystemExit(1)
    if incompletes:
        print(f"warning: {len(incompletes)} .incomplete blob(s) still under blobs/ (should be 0 when fully done)", flush=True)


if __name__ == "__main__":
    main()
