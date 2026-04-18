from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
from scipy import signal

MODEL_URL = "https://tfhub.dev/google/yamnet/1"


def ensure_sample_rate(
    original_sample_rate: int,
    waveform: np.ndarray,
    desired_sample_rate: int = 16000,
) -> tuple[int, np.ndarray]:
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform


def to_mono(wav: np.ndarray) -> np.ndarray:
    """Convert (N,) or (N, C) waveform to mono (N,)."""
    if wav.ndim == 1:
        return wav
    return wav.mean(axis=1)


def normalize_to_minus1_1(wav: np.ndarray) -> np.ndarray:
    """Convert waveform to float32 in approximately [-1, 1]."""
    if wav.dtype == np.int16:
        wav_f = wav.astype(np.float32) / np.float32(np.iinfo(np.int16).max)
    elif wav.dtype == np.int32:
        wav_f = wav.astype(np.float32) / np.float32(np.iinfo(np.int32).max)
    elif wav.dtype == np.uint8:
        wav_f = (wav.astype(np.float32) - 128.0) / 128.0
    else:
        wav_f = wav.astype(np.float32)
        m = np.max(np.abs(wav_f)) if wav_f.size else 0.0
        if m > 1.0:
            wav_f = wav_f / m
    return wav_f


def sanitize_dirname(name: str) -> str:
    """Sanitize label names for use as directory names."""
    name = name.strip()
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_\-().,]+", "_", name)
    return name[:120] if len(name) > 120 else name


def load_class_names_from_hub(model) -> list[str]:
    """Load YAMNet class display names from the TF Hub class map."""
    class_map_path = model.class_map_path().numpy()
    class_names: list[str] = []
    with tf.io.gfile.GFile(class_map_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])
    return class_names


def iter_audio_files(input_dir: Path, exts: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for ext in exts:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(set(files))


def format_source_for_report(src: Path, input_dir: Path, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "absolute":
        return str(src.resolve())
    try:
        return str(src.relative_to(input_dir))
    except ValueError:
        return src.name


def copy_or_move_file(src: Path, dst: Path, mode: str, overwrite: bool) -> str:
    if dst.exists():
        if not overwrite:
            return "skipped_exists"
        dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "move":
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

    return "done"


def main():
    ap = argparse.ArgumentParser(
        description="Classify audio files with YAMNet and copy/move them into class-wise folders."
    )
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing audio files.")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to save classified files.")
    ap.add_argument("--mode", choices=["copy", "move"], default="copy", help="Copy or move files.")
    ap.add_argument("--exts", type=str, default=".wav,.WAV", help="Comma-separated extensions.")
    ap.add_argument("--top_k", type=int, default=5, help="Number of top predictions to record.")
    ap.add_argument("--min_score", type=float, default=0.0, help="Put low-confidence files into uncertain_dir.")
    ap.add_argument("--uncertain_dir", type=str, default="__UNCERTAIN__", help="Folder for uncertain files.")
    ap.add_argument("--report", type=str, default="report.csv", help="CSV filename to write under output_dir.")
    ap.add_argument(
        "--report_path_mode",
        choices=["none", "relative", "absolute"],
        default="relative",
        help="How to store source paths in the report.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.strip() for e in args.exts.split(",") if e.strip())
    files = iter_audio_files(input_dir, exts)
    if not files:
        raise SystemExit(f"No audio files found in {input_dir} with extensions: {exts}")

    print(f"Loading YAMNet from TF Hub: {MODEL_URL}")
    model = hub.load(MODEL_URL)
    class_names = load_class_names_from_hub(model)

    report_path = output_dir / args.report
    processed = 0
    skipped = 0
    failed = 0

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "src",
                "dst_subdir",
                "pred_label",
                "pred_score",
                "topk_labels",
                "topk_scores",
                "status",
            ]
        )

        for src in files:
            try:
                sr, wav = wavfile.read(src, "rb")
                wav = to_mono(wav)
                sr, wav = ensure_sample_rate(sr, wav, desired_sample_rate=16000)
                waveform = normalize_to_minus1_1(wav)

                scores, _, _ = model(waveform)
                scores_np = scores.numpy()
                mean_scores = scores_np.mean(axis=0)

                top_indices = np.argsort(mean_scores)[::-1][: max(1, args.top_k)]
                pred_idx = int(top_indices[0])
                pred_label = class_names[pred_idx]
                pred_score = float(mean_scores[pred_idx])

                if pred_score < args.min_score:
                    subdir = args.uncertain_dir
                else:
                    subdir = sanitize_dirname(pred_label)

                dst_dir = output_dir / subdir
                dst = dst_dir / src.name

                status = copy_or_move_file(src, dst, args.mode, args.overwrite)
                if status == "skipped_exists":
                    skipped += 1
                else:
                    processed += 1

                topk_labels = [class_names[int(i)] for i in top_indices]
                topk_scores = [float(mean_scores[int(i)]) for i in top_indices]

                writer.writerow(
                    [
                        format_source_for_report(src, input_dir, args.report_path_mode),
                        subdir,
                        pred_label,
                        pred_score,
                        "|".join(topk_labels),
                        "|".join(map(str, topk_scores)),
                        status,
                    ]
                )

                print(f"{src.name} -> {subdir}/ (top1={pred_label}, score={pred_score:.3f}, status={status})")

            except Exception as e:
                failed += 1
                writer.writerow(
                    [
                        format_source_for_report(src, input_dir, args.report_path_mode),
                        "",
                        "",
                        "",
                        "",
                        "",
                        f"error: {e}",
                    ]
                )
                print(f"[SKIP] failed to process: {src} ({e})")

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()