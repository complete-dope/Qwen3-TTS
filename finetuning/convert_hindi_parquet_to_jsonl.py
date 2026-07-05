# coding=utf-8
import argparse
import io
import json
import os
from pathlib import Path

import librosa
import pyarrow.parquet as pq
import soundfile as sf


def iter_parquet_rows(parquet_dir: Path):
    files = sorted(parquet_dir.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train-*.parquet under {parquet_dir}")
    for path in files:
        table = pq.read_table(path, columns=["audio", "text", "gender"])
        d = table.to_pydict()
        n = len(d["text"])
        for i in range(n):
            yield d["audio"][i], d["text"][i], int(d["gender"][i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parquet_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "data",
    )
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument(
        "--gender",
        choices=("female", "male", "all"),
        default="female",
        help="female=0 male=1 per dataset card; 'all' keeps every row (mixed-speaker).",
    )
    p.add_argument("--target_sr", type=int, default=24000)
    p.add_argument("--max_samples", type=int, default=0, help="0 = no limit")
    p.add_argument("--jsonl_name", type=str, default="train_raw.jsonl")
    args = p.parse_args()

    gender_keep = None
    if args.gender == "female":
        gender_keep = 0
    elif args.gender == "male":
        gender_keep = 1

    wav_dir = args.out_dir / "wavs" # store all wavs to this folder
    wav_dir.mkdir(parents=True, exist_ok=True)

    lines = [] # appends audio , text , ref-audio , language here
    ref_path = None
    n_written = 0
    for audio_cell, text, gender in iter_parquet_rows(args.parquet_dir):
        if gender_keep is not None and gender != gender_keep:
            continue
        blob = audio_cell.get("bytes") # fetch audio_cell 
        if not blob:
            continue
        wav, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        wav = wav.astype("float32")
        if int(sr) != args.target_sr:
            wav = librosa.resample(y=wav, orig_sr=int(sr), target_sr=args.target_sr).astype(
                "float32"
            )

        name = f"utt_{n_written:07d}.wav"
        out_wav = wav_dir / name
        sf.write(out_wav, wav, args.target_sr, subtype="FLOAT")

        ap = str(out_wav.resolve())
        if ref_path is None:
            ref_copy = args.out_dir / "reference_speaker.wav"
            ref_copy.write_bytes(out_wav.read_bytes())
            ref_path = str(ref_copy.resolve())

        lines.append(
            {
                "audio": ap,
                "text": text,
                "ref_audio": ref_path,
                "language": "hi",
            }
        )
        n_written += 1
        if args.max_samples and n_written >= args.max_samples:
            break

    out_jsonl = args.out_dir / args.jsonl_name
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "written": n_written,
                "jsonl": str(out_jsonl.resolve()),
                "wav_dir": str(wav_dir.resolve()),
                "reference": ref_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


#python convert_hindi_parquet_to_jsonl.py --parquet_dir

