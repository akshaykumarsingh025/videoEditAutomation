import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

from config import OLLAMA_URL, OLLAMA_MODEL, PATHS, RETRY_ATTEMPTS, RETRY_BASE_DELAY

logger = logging.getLogger("pipeline.director")

DIRECTOR_SYSTEM_PROMPT = """You are a creative video director for a healthcare YouTube channel run by Dr. Deepika.

You receive a Hinglish transcript with timestamps. Your job is to create a JSON timeline of visual overlays to make the video engaging.

RULES:
1. Output ONLY valid JSON. No markdown, no explanation, no code fences.
2. All text content must be in Hinglish (Romanized Hindi) only.
3. Use these action types ONLY: LOWER_THIRD, COMFYUI_PROMPT, WEB_GIF, TEXT_CARD, WATERMARK
4. Each timeline entry must have: id, time (HH:MM:SS), duration (seconds), action, data, position
5. Optional field: fade ("in", "out", "in-out") for COMFYUI_PROMPT and TEXT_CARD
6. LOWER_THIRD: Use for speaker introductions. Position: "bottom-left"
7. COMFYUI_PROMPT: Detailed English prompts for AI image generation (medical/educational themes)
8. WEB_GIF: Short search terms for reaction GIFs (e.g., "mind blown", "thumbs up")
9. TEXT_CARD: Hinglish text for myth/fact cards, key takeaways. Position: "center"
10. WATERMARK: Always include ONE entry at time "00:00:00" with duration 0 for persistent watermark
11. Do NOT place overlays during the first 2 seconds of video
12. Space overlays at least 5 seconds apart. Aim for one strong visual beat every 12-18 seconds, not constant popups.
13. Make the first overlay a sharp hook or lower third after 00:00:02.500 when the transcript supports it.
14. Prefer clean educational overlays over gimmicks. Use WEB_GIF sparingly (maximum 2) and only when it fits the tone.
15. Do not invent medical claims. Visuals must support what the speaker actually says.
16. Include a "hero_moments" array: 2-3 segments of 20-40 seconds each that would work as YouTube Shorts/Reels
17. Include an "seo" object with: title (Hinglish), description (Hinglish, 2-3 sentences), tags (array of 8-12 Hinglish tags), chapters (array of {time, title})

OUTPUT FORMAT:
{
  "video_info": {"source": "", "duration_sec": 0, "language": "hinglish"},
  "silence_trim": true,
  "timeline": [...],
  "hero_moments": [...],
  "seo": {"title": "", "description": "", "tags": [], "chapters": []}
}"""


def list_ollama_models() -> list[dict]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError("Ollama executable was not found on PATH") from e
    except subprocess.CalledProcessError as e:
        message = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"Could not read Ollama model list: {message}") from e

    models = []
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        cols = re.split(r"\s{2,}", line)
        if len(cols) < 3:
            continue
        name = cols[0]
        model = {
            "name": name,
            "id": cols[1],
            "size": cols[2],
            "modified": cols[3] if len(cols) > 3 else "",
            "kind": "cloud" if cols[2] == "-" or "cloud" in name.lower() else "local",
        }
        models.append(model)
    return models


def format_ollama_models(models: list[dict]) -> str:
    if not models:
        return "No Ollama models found."

    lines = []
    for i, model in enumerate(models, 1):
        modified = f", {model['modified']}" if model.get("modified") else ""
        lines.append(f"{i}. {model['name']} [{model['kind']}, {model['size']}{modified}]")
    return "\n".join(lines)


def resolve_ollama_model(model_name: str | None = None, interactive: bool = True) -> str:
    models = list_ollama_models()
    names = [m["name"] for m in models]

    requested = (model_name or OLLAMA_MODEL or "").strip()
    if requested:
        if requested in names:
            return requested
        raise ValueError(
            f"Ollama model '{requested}' is not installed/listed.\n"
            f"Available models:\n{format_ollama_models(models)}"
        )

    if not models:
        raise RuntimeError("No Ollama models found. Pull or enable a model first.")

    if interactive and sys.stdin.isatty():
        print("\nAvailable Ollama models:")
        print(format_ollama_models(models))
        choice = input("Select AI director model number or name: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]["name"]
        if choice in names:
            return choice
        raise ValueError(f"Invalid model selection: {choice}")

    logger.info("No model selected interactively; using first non-embedding Ollama-listed model")
    for model in models:
        if "embed" not in model["name"].lower():
            return model["name"]
    return models[0]["name"]


def generate_timeline(
    transcript_text: str,
    srt_content: str,
    duration_sec: float,
    input_filename: str,
    model_name: str,
) -> dict:
    logger.info(f"Generating timeline with {model_name} (video duration: {duration_sec}s)")

    user_prompt = f"""TRANSCRIPT (Hinglish):
{transcript_text}

SRT TIMESTAMPS:
{srt_content[:3000]}

VIDEO DURATION: {duration_sec} seconds
INPUT FILE: {input_filename}

Create the timeline JSON now."""

    response = _call_ollama(user_prompt, model_name)
    timeline_data = _parse_ollama_json(response)

    if timeline_data:
        timeline_data.setdefault("video_info", {})["duration_sec"] = duration_sec
        timeline_data.setdefault("video_info", {})["source"] = input_filename
        timeline_data.setdefault("video_info", {})["language"] = "hinglish"
        timeline_data.setdefault("silence_trim", True)

    timeline_path = PATHS["temp"] / f"{Path(input_filename).stem}_timeline.json"
    timeline_path.write_text(json.dumps(timeline_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Timeline saved: {timeline_path}")

    return {"timeline_path": timeline_path, "timeline_data": timeline_data}


def generate_seo(transcript_text: str, input_stem: str, model_name: str) -> dict:
    logger.info(f"Generating SEO metadata with {model_name}")

    seo_prompt = f"""Based on this Hinglish video transcript, generate YouTube SEO metadata.

TRANSCRIPT:
{transcript_text[:2000]}

Output ONLY this JSON format, no other text:
{{
  "title": "Hinglish title, under 70 chars",
  "description": "2-3 sentence Hinglish description with key topics",
  "tags": ["tag1", "tag2", ...],
  "chapters": [{{"time": "00:00", "title": "Chapter name"}}]
}}"""

    response = _call_ollama(seo_prompt, model_name)
    seo_data = _parse_ollama_json(response)

    seo_path = PATHS["temp"] / f"{input_stem}_seo_metadata.json"
    seo_path.write_text(json.dumps(seo_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"SEO metadata saved: {seo_path}")

    return {"seo_path": seo_path, "seo_data": seo_data}


def _call_ollama(prompt: str, model_name: str) -> str:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            logger.info(f"Calling Ollama {model_name} (attempt {attempt})")
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "system": DIRECTOR_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 4096},
                },
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.error(f"Ollama call failed (attempt {attempt}): {e}. Retrying in {delay}s")
            if attempt == RETRY_ATTEMPTS:
                raise RuntimeError(f"Ollama failed after {RETRY_ATTEMPTS} attempts: {e}")
            time.sleep(delay)


def _parse_ollama_json(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Direct JSON parse failed, attempting extraction from text")
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.error(f"Could not parse JSON from Ollama response: {text[:200]}")
        return {}


def unload_ollama(model_name: str):
    try:
        subprocess.run(["ollama", "stop", model_name], timeout=30, capture_output=True)
        logger.info(f"Ollama model {model_name} stopped")
    except Exception as e:
        logger.warning(f"Could not stop Ollama model: {e}")
