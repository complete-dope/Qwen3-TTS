import os

WHISPER_MODEL_ID = "openai/whisper-small"

_pipe = None


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return -1


def get_asr_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from transformers import pipeline

    device = _pick_device()
    dtype = torch.float16 if device == 0 else torch.float32
    kwargs = dict(model=WHISPER_MODEL_ID, dtype=dtype)
    if device == -1:
        kwargs["device"] = -1
    else:
        kwargs["device"] = device
    _pipe = pipeline("automatic-speech-recognition", **kwargs)
    return _pipe


def transcribe_wav(wav_path: str) -> str:
    path = os.path.abspath(wav_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    pipe = get_asr_pipeline()
    out = pipe(
        path,
        generate_kwargs={"language": "en", "task": "transcribe"},
    )
    return (out.get("text") or "").strip()


def transcribe_wavs_to_log(out_dir: str, items: list[tuple[str, str]], log_name: str = "asr_transcripts.log") -> str:
    lines = []
    for label, rel in items:
        p = os.path.join(out_dir, rel)
        try:
            t = transcribe_wav(p)
        except Exception as e:
            t = f"<error: {e}>"
        lines.append(f"{label} ({rel}): {t}")
    log_path = os.path.join(out_dir, log_name)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return log_path


def transcribe_outputs_in_dir(out_dir: str, basenames: list[str] | None = None) -> dict[str, str]:
    if basenames is None:
        basenames = [
            "e2e_voice_clone.wav",
            "sanity_gen_codes_only.wav",
            "sanity_ref_codec_roundtrip.wav",
            "ref_source.wav",
        ]
    results = {}
    for name in basenames:
        p = os.path.join(out_dir, name)
        if not os.path.isfile(p):
            results[name] = f"<missing file {name}>"
            continue
        results[name] = transcribe_wav(p)
    return results


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for k, v in transcribe_outputs_in_dir(here).items():
        print(f"{k}: {v!r}")
