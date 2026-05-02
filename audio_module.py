import json
import logging
import gc
from pathlib import Path

from config import (
    PATHS,
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    SILENCE_THRESHOLD_MS,
)

logger = logging.getLogger("pipeline.audio")


def extract_audio(video_path: str | Path) -> Path:
    from moviepy.editor import VideoFileClip

    video_path = Path(video_path)
    audio_path = PATHS["temp"] / f"{video_path.stem}.wav"
    logger.info(f"Extracting audio from {video_path} -> {audio_path}")
    clip = VideoFileClip(str(video_path))
    try:
        if clip.audio is None:
            raise ValueError(f"Input video has no audio track: {video_path}")
        clip.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
    finally:
        clip.close()
    return audio_path


def transcribe(audio_path: str | Path) -> dict:
    import subprocess
    try:
        subprocess.run(["ollama", "stop", "gemma4:e4b", "gemma4:26b", "gemma4:31b-cloud"], timeout=15, capture_output=True)
        logger.info("Stopped Ollama models to free VRAM for Whisper")
    except Exception:
        pass

    import whisper

    audio_path = Path(audio_path)
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)

    logger.info(f"Transcribing {audio_path} (language={WHISPER_LANGUAGE})")
    result = model.transcribe(
        str(audio_path),
        language=WHISPER_LANGUAGE,
        word_timestamps=True,
        verbose=False,
    )

    _unload_whisper(model)

    base_name = audio_path.stem
    transcript_path = PATHS["temp"] / f"{base_name}_transcript.txt"
    srt_path = PATHS["temp"] / f"{base_name}.srt"
    words_path = PATHS["temp"] / f"{base_name}_words.json"

    transcript_text = result["text"].strip()
    transcript_path.write_text(transcript_text, encoding="utf-8")
    logger.info(f"Transcript saved: {transcript_path}")

    srt_content = _format_srt(result["segments"])
    srt_path.write_text(srt_content, encoding="utf-8")
    logger.info(f"SRT saved: {srt_path}")

    words_data = _extract_words(result["segments"])
    words_path.write_text(json.dumps(words_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Word timestamps saved: {words_path}")

    return {
        "transcript_path": transcript_path,
        "srt_path": srt_path,
        "words_path": words_path,
        "transcript_text": transcript_text,
        "duration": result["segments"][-1]["end"] if result["segments"] else 0,
    }


def detect_silence(words_path: str | Path, video_duration: float) -> dict:
    words_path = Path(words_path)
    words_data = json.loads(words_path.read_text(encoding="utf-8"))

    silence_cuts = []
    silence_threshold = SILENCE_THRESHOLD_MS / 1000.0

    if not words_data:
        logger.warning("No word data available for silence detection")
        return {"silence_cuts_path": None, "trimmed_srt_path": None}

    prev_end = 0.0
    for word in words_data:
        gap = word["start"] - prev_end
        if gap >= silence_threshold:
            silence_cuts.append({
                "start": _format_timestamp(prev_end),
                "end": _format_timestamp(word["start"]),
                "duration_ms": int(gap * 1000),
                "start_sec": round(prev_end, 3),
                "end_sec": round(word["start"], 3),
            })
        prev_end = max(prev_end, word["end"])

    gap = video_duration - prev_end
    if gap >= silence_threshold:
        silence_cuts.append({
            "start": _format_timestamp(prev_end),
            "end": _format_timestamp(video_duration),
            "duration_ms": int(gap * 1000),
            "start_sec": round(prev_end, 3),
            "end_sec": round(video_duration, 3),
        })

    base_name = words_path.stem.replace("_words", "")
    cuts_path = PATHS["temp"] / f"{base_name}_silence_cuts.json"
    cuts_path.write_text(json.dumps(silence_cuts, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Silence cuts saved: {cuts_path} ({len(silence_cuts)} cuts found)")

    return {"silence_cuts_path": cuts_path, "silence_cuts": silence_cuts}


def adjust_srt_for_silence(srt_path: str | Path, silence_cuts: list) -> Path:
    srt_path = Path(srt_path)
    if not silence_cuts:
        trimmed_path = PATHS["temp"] / f"{srt_path.stem}_trimmed.srt"
        trimmed_path.write_text(srt_path.read_text(encoding="utf-8"), encoding="utf-8")
        return trimmed_path

    total_shift = 0.0
    shift_map = []
    for cut in silence_cuts:
        duration = cut["end_sec"] - cut["start_sec"]
        shift_map.append({"before": cut["start_sec"], "shift": total_shift, "cut_duration": duration})
        total_shift += duration

    srt_content = srt_path.read_text(encoding="utf-8")
    blocks = srt_content.strip().split("\n\n")

    adjusted_blocks = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        time_line = lines[1]
        start_sec, end_sec = _parse_srt_time(time_line)

        in_cut = False
        for cut in silence_cuts:
            if start_sec >= cut["start_sec"] and start_sec < cut["end_sec"]:
                in_cut = True
                break

        if in_cut:
            continue

        new_start = start_sec
        new_end = end_sec
        for shift_entry in shift_map:
            if start_sec >= shift_entry["before"] + shift_entry["cut_duration"]:
                new_start -= shift_entry["cut_duration"]
                new_end -= shift_entry["cut_duration"]

        lines[1] = f"{_sec_to_srt_time(new_start)} --> {_sec_to_srt_time(new_end)}"
        adjusted_blocks.append("\n".join(lines))

    base_name = srt_path.stem
    trimmed_path = PATHS["temp"] / f"{base_name}_trimmed.srt"
    trimmed_path.write_text("\n\n".join(adjusted_blocks), encoding="utf-8")
    logger.info(f"Trimmed SRT saved: {trimmed_path}")
    return trimmed_path


def _unload_whisper(model):
    logger.info("Unloading Whisper model, freeing VRAM")
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    except ImportError:
        pass


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _format_srt(segments: list) -> str:
    srt_blocks = []
    for i, seg in enumerate(segments, 1):
        start = _sec_to_srt_time(seg["start"])
        end = _sec_to_srt_time(seg["end"])
        text = seg["text"].strip()
        srt_blocks.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(srt_blocks)


def _sec_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_srt_time(time_line: str) -> tuple:
    parts = time_line.strip().split(" --> ")
    if len(parts) != 2:
        return 0.0, 0.0
    return _srt_time_to_sec(parts[0]), _srt_time_to_sec(parts[1])


def _srt_time_to_sec(time_str: str) -> float:
    time_str = time_str.strip().replace(",", ".")
    parts = time_str.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def _extract_words(segments: list) -> list:
    words = []
    for seg in segments:
        if "words" in seg:
            for w in seg["words"]:
                words.append({
                    "word": w["word"],
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })
    return words
