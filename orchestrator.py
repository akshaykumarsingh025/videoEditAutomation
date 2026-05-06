import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path

os.environ["IMAGEMAGICK_BINARY"] = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"

from config import (
    PATHS,
    PROGRESS_FILE,
    ensure_dirs,
    setup_logging,
    EXPORT_PROFILES,
    OLLAMA_MODEL,
)
from director_module import (
    format_ollama_models,
    generate_timeline,
    generate_seo,
    list_ollama_models,
    resolve_ollama_model,
    unload_ollama,
)
from preflight_module import format_preflight_report, run_preflight

PHASES = [1, 1.5, 2, 2.5, 3, 4, 5]


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"current_phase": 0, "completed_phases": [], "input_file": None}


def save_progress(progress):
    progress["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def is_completed(progress, phase):
    return phase in progress.get("completed_phases", [])


def mark_completed(progress, phase):
    if phase not in progress["completed_phases"]:
        progress["completed_phases"].append(phase)
    save_progress(progress)


def run_pipeline(
    input_file,
    draft=False,
    profile="landscape",
    resume=False,
    skip_phases=None,
    all_formats=False,
    model_name=None,
    interactive=True,
    skip_check=False,
):
    logger = setup_logging()
    ensure_dirs()

    skip_phases = skip_phases or []
    progress = load_progress()
    fresh_run = False

    if resume and progress.get("input_file"):
        input_file = progress["input_file"]
        logger.info(f"Resuming pipeline for: {input_file}")
    elif resume and not input_file:
        logger.error("No checkpoint input file found. Pass --input to start a fresh run.")
        sys.exit(1)
    else:
        fresh_run = True

    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if fresh_run:
        progress = {
            "current_phase": 0,
            "completed_phases": [],
            "input_file": str(input_file),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_progress(progress)

    needs_director = not is_completed(progress, 2) and 2 not in skip_phases
    selected_model = model_name or OLLAMA_MODEL
    if needs_director:
        selected_model = resolve_ollama_model(selected_model, interactive=interactive)
        logger.info(f"AI director model selected: {selected_model}")

    if not skip_check:
        report = run_preflight(model_name=selected_model, require_model=needs_director, print_report=False)
        for warning in report["warnings"]:
            logger.warning(f"Preflight: {warning}")
        if report["errors"]:
            for error in report["errors"]:
                logger.error(f"Preflight: {error}")
            raise SystemExit("Preflight failed. Fix the errors above or run with --skip-check if you know what you are doing.")

    video_name = input_path.stem
    transcript_text = ""
    srt_path = None
    trimmed_srt_path = None
    silence_cuts = []
    timeline_data = None
    seo_data = None
    duration_sec = 0

    # === Phase 1: Ingestion & Transcription ===
    if not is_completed(progress, 1) and 1 not in skip_phases:
        from audio_module import extract_audio, transcribe

        logger.info("=" * 60)
        logger.info("PHASE 1: Ingestion & Transcription (Whisper)")
        logger.info("=" * 60)
        progress["current_phase"] = 1
        save_progress(progress)

        audio_path = extract_audio(input_path)
        result = transcribe(audio_path)

        transcript_text = result["transcript_text"]
        srt_path = result["srt_path"]
        duration_sec = result["duration"]

        mark_completed(progress, 1)
    else:
        logger.info("Skipping Phase 1 (already completed)")
        transcript_path = PATHS["temp"] / f"{video_name}_transcript.txt"
        srt_path = PATHS["temp"] / f"{video_name}.srt"
        words_path = PATHS["temp"] / f"{video_name}_words.json"
        transcript_text = transcript_path.read_text(encoding="utf-8") if transcript_path.exists() else ""
        if words_path.exists():
            words_data = json.loads(words_path.read_text(encoding="utf-8"))
            if words_data:
                duration_sec = words_data[-1]["end"]

    # === Phase 1.5: Silence Detection ===
    if not is_completed(progress, 1.5) and 1.5 not in skip_phases:
        from audio_module import detect_silence, adjust_srt_for_silence

        logger.info("=" * 60)
        logger.info("PHASE 1.5: Silence Detection & Auto-Trim")
        logger.info("=" * 60)
        progress["current_phase"] = 1.5
        save_progress(progress)

        words_path = PATHS["temp"] / f"{video_name}_words.json"
        if words_path.exists() and duration_sec > 0:
            silence_result = detect_silence(words_path, duration_sec)
            silence_cuts = silence_result.get("silence_cuts", [])

            if srt_path and srt_path.exists() and silence_cuts:
                trimmed_srt_path = adjust_srt_for_silence(srt_path, silence_cuts)
            else:
                trimmed_srt_path = srt_path
        else:
            logger.warning("No word-level data available, skipping silence detection")
            trimmed_srt_path = srt_path

        mark_completed(progress, 1.5)
    else:
        logger.info("Skipping Phase 1.5 (already completed)")
        cuts_path = PATHS["temp"] / f"{video_name}_silence_cuts.json"
        if cuts_path.exists():
            silence_cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
        trimmed_srt_path = PATHS["temp"] / f"{video_name}_trimmed.srt"
        if not trimmed_srt_path.exists():
            trimmed_srt_path = srt_path

    # === Phase 2: AI Director ===
    if not is_completed(progress, 2) and 2 not in skip_phases:
        logger.info("=" * 60)
        logger.info("PHASE 2: AI Director (Gemma4 via Ollama)")
        logger.info("=" * 60)
        progress["current_phase"] = 2
        save_progress(progress)

        srt_content = ""
        if trimmed_srt_path and trimmed_srt_path.exists():
            srt_content = trimmed_srt_path.read_text(encoding="utf-8")

        tl_result = generate_timeline(transcript_text, srt_content, duration_sec, input_path.name, selected_model)
        timeline_data = tl_result["timeline_data"]

        seo_result = generate_seo(transcript_text, video_name, selected_model)
        seo_data = seo_result["seo_data"]

        unload_ollama(selected_model)
        mark_completed(progress, 2)
    else:
        logger.info("Skipping Phase 2 (already completed)")
        timeline_path = PATHS["temp"] / f"{video_name}_timeline.json"
        if not timeline_path.exists():
            timeline_path = PATHS["temp"] / "timeline.json"
        if timeline_path.exists():
            timeline_data = json.loads(timeline_path.read_text(encoding="utf-8"))
        seo_path = PATHS["temp"] / f"{video_name}_seo_metadata.json"
        if not seo_path.exists():
            seo_path = PATHS["temp"] / "seo_metadata.json"
        if seo_path.exists():
            seo_data = json.loads(seo_path.read_text(encoding="utf-8"))

    # === Phase 2.5: Timeline Validation ===
    if not is_completed(progress, 2.5) and 2.5 not in skip_phases:
        from validator_module import validate_timeline, filter_invalid_entries

        logger.info("=" * 60)
        logger.info("PHASE 2.5: Timeline Validation")
        logger.info("=" * 60)
        progress["current_phase"] = 2.5
        save_progress(progress)

        if timeline_data:
            validation = validate_timeline(timeline_data, duration_sec)

            if not validation["valid"]:
                logger.warning(f"Timeline has {len(validation['errors'])} errors, attempting fix...")
                timeline_data = filter_invalid_entries(timeline_data)

                revalidation = validate_timeline(timeline_data, duration_sec)
                if not revalidation["valid"]:
                    logger.error("Timeline still invalid after filtering, continuing with valid entries only")

            if validation["warnings"]:
                for w in validation["warnings"]:
                    logger.warning(f"  {w}")
        else:
            logger.error("No timeline data to validate")

        mark_completed(progress, 2.5)
    else:
        logger.info("Skipping Phase 2.5 (already completed)")

    # === Phase 3: Asset Generation ===
    if not is_completed(progress, 3) and 3 not in skip_phases:
        from asset_module import generate_comfyui_images, download_web_assets, generate_graphics

        logger.info("=" * 60)
        logger.info("PHASE 3: Asset Generation & Retrieval")
        logger.info("=" * 60)
        progress["current_phase"] = 3
        save_progress(progress)

        if timeline_data and "timeline" in timeline_data:
            entries = timeline_data["timeline"]

            if draft:
                logger.info("--- 3A: ComfyUI Images skipped in draft mode ---")
            else:
                logger.info("--- 3A: ComfyUI Images (Z-Image) ---")
                comfyui_result = generate_comfyui_images(entries, asset_prefix=video_name)
                logger.info(f"ComfyUI: {len(comfyui_result['generated'])} generated, {len(comfyui_result['failed'])} failed")

            logger.info("--- 3B: Web Assets (Giphy) ---")
            web_result = download_web_assets(entries, asset_prefix=video_name)
            logger.info(f"Web: {len(web_result['downloaded'])} downloaded, {len(web_result['failed'])} failed")

            logger.info("--- 3C: Static Graphics (Pillow) ---")
            gfx_result = generate_graphics(entries, asset_prefix=video_name)
            logger.info(f"Graphics: {len(gfx_result['generated'])} generated, {len(gfx_result['failed'])} failed")
        else:
            logger.warning("No timeline data, skipping asset generation")

        mark_completed(progress, 3)
    else:
        logger.info("Skipping Phase 3 (already completed)")

    # === Phase 4: Compositing ===
    if not is_completed(progress, 4) and 4 not in skip_phases:
        from compositing_module import composite_video

        logger.info("=" * 60)
        logger.info("PHASE 4: Compositing & Export")
        logger.info("=" * 60)
        progress["current_phase"] = 4
        save_progress(progress)

        effective_srt = trimmed_srt_path or srt_path

        if all_formats:
            for prof in EXPORT_PROFILES:
                logger.info(f"--- Exporting {prof} profile ---")
                composite_video(
                    input_path, timeline_data, effective_srt,
                    silence_cuts=silence_cuts,
                    draft=draft,
                    profile=prof,
                    asset_prefix=video_name,
                )
        else:
            composite_video(
                input_path, timeline_data, effective_srt,
                silence_cuts=silence_cuts,
                draft=draft,
                profile=profile,
                asset_prefix=video_name,
            )

        mark_completed(progress, 4)
    else:
        logger.info("Skipping Phase 4 (already completed)")

    # === Phase 5: SEO & Thumbnail ===
    if not is_completed(progress, 5) and 5 not in skip_phases:
        from compositing_module import generate_thumbnail, write_seo_files

        logger.info("=" * 60)
        logger.info("PHASE 5: SEO Metadata & Thumbnail")
        logger.info("=" * 60)
        progress["current_phase"] = 5
        save_progress(progress)

        try:
            hero_moments = timeline_data.get("hero_moments", []) if timeline_data else []
            hero = hero_moments[0] if hero_moments else None
            generate_thumbnail(input_path, hero)
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")

        if seo_data:
            write_seo_files(seo_data, input_path)
        else:
            logger.warning("No SEO data available")

        mark_completed(progress, 5)
    else:
        logger.info("Skipping Phase 5 (already completed)")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Output directory: {PATHS['output']}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Automated Video Editing Pipeline - Dr. Deepika")
    parser.add_argument("--input", "-i", help="Path to input video file (.mp4)")
    parser.add_argument("--draft", "-d", action="store_true", help="Draft mode: 480p, skip ComfyUI, fast export")
    parser.add_argument("--shorts", "-s", action="store_true", help="Export in 9:16 portrait for Shorts/Reels")
    parser.add_argument("--square", "-sq", action="store_true", help="Export in 1:1 square for Instagram")
    parser.add_argument("--all-formats", "-a", action="store_true", help="Export all formats (landscape + shorts + square)")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--skip", type=str, default="", help="Comma-separated phase numbers to skip (e.g., '1,1.5,2')")
    parser.add_argument("--model", "-m", help="Ollama model name from `ollama list` (local or cloud)")
    parser.add_argument("--list-models", action="store_true", help="Print available Ollama models and exit")
    parser.add_argument("--check", action="store_true", help="Run setup preflight checks and exit")
    parser.add_argument("--skip-check", action="store_true", help="Skip preflight checks before running")
    parser.add_argument("--no-interactive", action="store_true", help="Do not prompt for model selection; use --model, .env, or first listed model")

    args = parser.parse_args()

    if args.list_models:
        print(format_ollama_models(list_ollama_models()))
        return

    if args.check:
        model_for_check = args.model or OLLAMA_MODEL
        report = run_preflight(model_name=model_for_check, require_model=True, print_report=False)
        print(format_preflight_report(report))
        raise SystemExit(1 if report["errors"] else 0)

    if not args.input and not args.resume:
        parser.error("--input is required unless --resume, --check, or --list-models is used")

    profile = "landscape"
    if args.shorts:
        profile = "shorts"
    elif args.square:
        profile = "square"

    skip_phases = []
    if args.skip:
        for p in args.skip.split(","):
            try:
                skip_phases.append(float(p.strip()))
            except ValueError:
                pass

    run_pipeline(
        input_file=args.input,
        draft=args.draft,
        profile=profile,
        resume=args.resume,
        skip_phases=skip_phases,
        all_formats=args.all_formats,
        model_name=args.model,
        interactive=not args.no_interactive,
        skip_check=args.skip_check,
    )


if __name__ == "__main__":
    main()
