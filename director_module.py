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
3. Use these action types ONLY: BROLL_IMAGE, LOWER_THIRD, TEXT_CARD
4. Each timeline entry must have: id, time (HH:MM:SS), duration (seconds), action, data, position
5. Optional field: fade ("in", "out", "in-out") — REQUIRED for BROLL_IMAGE (use "in-out")
6. Optional field: fx — for BROLL_IMAGE, set fx: "ken_burns_in" (slow zoom in) or "ken_burns_out" (slow zoom out). Alternate between them.

BROLL_IMAGE PROMPT RULES (CRITICAL — READ CAREFULLY):
- This is the MOST IMPORTANT action. Use it whenever the speaker mentions a medical concept or topic.
- The image will REPLACE the video FULL SCREEN while audio continues — like real B-roll editing.
- KEEP PROMPTS SIMPLE, CONCRETE, AND PEOPLE-FOCUSED. The ZImage AI model CANNOT do abstract concepts, diagrams, infographics, or 3D renders.

MANDATORY: Every BROLL_IMAGE prompt MUST describe a SCENE with a PERSON (woman/girl/female doctor/nurse/patient) doing something specific.
NEVER generate prompts without a person. NEVER generate prompts for abstract concepts.

TEMPLATE for every prompt: "[person description] [doing action] [in setting], [mood/atmosphere], soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters"

CONCRETE EXAMPLES OF GOOD PROMPTS:
- "a young Indian woman sitting in a doctor's waiting room looking calm, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters"
- "a female doctor in a white coat holding a vaccine vial and smiling reassuringly, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters"
- "a pregnant Indian woman gently holding her belly with a peaceful smile, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters"
- "a young woman lying on a medical bed while a female doctor examines her with care, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters"
- "an Indian woman patient talking to a female doctor across a desk in a cozy clinic, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters"
- "a middle-aged Indian woman looking relieved after a health checkup, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters"
- "a female doctor showing a medicine bottle to a woman patient and explaining softly, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters"
- "a young Indian girl receiving an injection on her arm from a kind female nurse, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters"

NEVER USE THESE (THE MODEL CANNOT RENDER THEM):
- "infographic", "diagram", "chart", "statistics", "3D animation", "cross-section", "anatomical illustration", "medical diagram", "virus particles", "molecular structure", "flowchart", "comparison table", "X-ray", "microscope view", "abstract representation", "screenshot", "UI", "app interface"

DURATION: 10-20 seconds per BROLL_IMAGE. Keep the image on screen for the ENTIRE duration the speaker talks about that topic. Look at the SRT timestamps — if the speaker discusses HPV from 01:11 to 01:35, set duration to 24 seconds. NEVER cut an image short while the topic is still being discussed.
Position: "center". Always set fade: "in-out". Set fx: "ken_burns_in" or "ken_burns_out" (alternate between them).
Generate 8-14 BROLL_IMAGE entries per video — cover as many topic changes as possible.

LOWER_THIRD: Use for speaker introductions only. Position: "bottom-left". Duration: 5-7 seconds.

TEXT_CARD: Hinglish text for myth/fact cards, key takeaways, important warnings, statistics, and emphasis moments. Use these VERY FREQUENTLY — every time the speaker says something important, surprising, or worth remembering, add a TEXT_CARD. Generate 8-15 TEXT_CARD entries per video. Make these IMPACTFUL — short punchy Hinglish statements. Position: "center". Duration: 6-8 seconds. Examples: "HPV Vaccine = Cancer Se Bachav!", "Har Mahila Ko Pap Smear Zaroori Hai", "9-45 Saal Ki Mahilayein Vaccine Le Sakti Hain", "Early Detection Saves Lives!".

Do NOT place overlays during the first 3 seconds of video.
Space overlays at least 2 seconds apart.
Prefer BROLL_IMAGE over everything else — show the viewer what the doctor is talking about.
Do not invent medical claims. Visuals must support what the speaker actually says.
Include a "hero_moments" array: 2-3 segments of 20-40 seconds each that would work as YouTube Shorts/Reels
Include an "seo" object with: title (Hinglish), description (Hinglish, 2-3 sentences), tags (array of 8-12 Hinglish tags), chapters (array of {time, title})

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
    logger.info(f"Unloading Ollama model: {model_name}")

    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.splitlines()[1:]:
                cols = line.strip().split()
                if cols:
                    try:
                        subprocess.run(["ollama", "stop", cols[0]], timeout=15, capture_output=True)
                        logger.info(f"Stopped: {cols[0]}")
                    except Exception:
                        pass
    except Exception:
        pass

    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
            logger.info(f"VRAM after Ollama unload: {vram:.0f} MB")
    except ImportError:
        pass
