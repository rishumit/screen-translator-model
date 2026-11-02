"""CTranslate2 translation server for ScreenTranslator.

Loads a CTranslate2 model once (int8 on CPU, or CUDA when available) and translates batches
received as JSON Lines on stdin:  {"texts": [...], "requestId": "1"}
and answers on stdout:            {"translations": [...], "requestId": "1"}

Works with quickmt models (e.g. quickmt/quickmt-en-he: model.bin + src.spm.model + tgt.spm.model)
and with OPUS-MT / Marian models converted by ct2-transformers-converter (source.spm / target.spm).

Requirements:  python -m pip install ctranslate2 sentencepiece
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def emit(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def find_spm(model_dir: Path, names) -> Path | None:
    for name in names:
        p = model_dir / name
        if p.is_file():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0, help="0 = number of physical cores (max 8)")
    args = parser.parse_args()

    try:
        import ctranslate2
        import sentencepiece as spm
    except ImportError as exc:
        emit({"error": f"Missing Python dependencies: {exc}. Install with: python -m pip install ctranslate2 sentencepiece",
              "requestId": "startup"})
        return 1

    model_dir = args.model_dir
    src_spm = find_spm(model_dir, ("src.spm.model", "source.spm", "sentencepiece.bpe.model", "spm.model"))
    tgt_spm = find_spm(model_dir, ("tgt.spm.model", "target.spm")) or src_spm
    if not (model_dir / "model.bin").is_file() or src_spm is None:
        emit({"error": f"Not a CTranslate2 model folder (need model.bin and a SentencePiece model): {model_dir}",
              "requestId": "startup"})
        return 1

    try:
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        threads = args.threads or max(1, min(8, (os.cpu_count() or 4) // 2))
        translator = ctranslate2.Translator(
            str(model_dir),
            device=device,
            compute_type="int8_float16" if device == "cuda" else "int8",
            inter_threads=1,
            intra_threads=threads,
        )
        sp_src = spm.SentencePieceProcessor(model_file=str(src_spm))
        sp_tgt = spm.SentencePieceProcessor(model_file=str(tgt_spm))
        # Warm-up so the first real request is fast.
        translator.translate_batch([sp_src.encode("Hello", out_type=str)], beam_size=1, max_decoding_length=8)
        emit({"status": "ready", "device": f"{device} ({threads} threads)"})
    except Exception as exc:  # noqa: BLE001 - report any load failure to the app
        emit({"error": f"Failed to load model: {exc}", "requestId": "startup"})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = None
        try:
            request = json.loads(line)
            request_id = request.get("requestId", "unknown")
            texts = request.get("texts", [])

            # Translate non-empty texts only, keep positions.
            idx = [i for i, t in enumerate(texts) if t and t.strip()]
            out = [""] * len(texts)
            if idx:
                tokens = sp_src.encode([texts[i] for i in idx], out_type=str)
                results = translator.translate_batch(
                    tokens,
                    beam_size=args.beam_size,
                    max_batch_size=32,
                    batch_type="examples",
                    max_decoding_length=256,
                    repetition_penalty=1.1,
                )
                for i, r in zip(idx, results):
                    out[i] = sp_tgt.decode(r.hypotheses[0]).strip()

            emit({"translations": out, "requestId": request_id})
        except Exception as exc:  # noqa: BLE001
            req_id = request.get("requestId", "unknown") if isinstance(request, dict) else "unknown"
            emit({"error": f"Translation error: {exc}", "requestId": req_id})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
