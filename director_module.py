import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

from config import OLLAMA_URL, OLLAMA_MODEL, PATHS, RETRY_ATTEMPTS, RETRY_BASE_DELAY, BRAND

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

CRITICAL COVERAGE RULE:
- You MUST place overlays across the ENTIRE video duration, from start to finish.
- Divide the video into equal segments (every 30-45 seconds) and ensure EACH segment has at least one BROLL_IMAGE or TEXT_CARD.
- DO NOT cluster all overlays in the first few minutes. The LAST quarter of the video needs overlays just as much as the first quarter.
- If the video is 10 minutes long, you need overlays at 1:00, 2:00, 3:00, ... 9:00 etc.

BROLL_IMAGE PROMPT RULES (CRITICAL - READ CAREFULLY):
- This is the MOST IMPORTANT action. Use it whenever the speaker mentions a medical concept or topic.
- The image will REPLACE the video FULL SCREEN while audio continues, like real B-roll editing.
- KEEP PROMPTS SIMPLE, REALISTIC, CONCRETE, AND PEOPLE-FOCUSED. The ZImage AI model does poorly with abstract concepts, diagrams, tiny objects, close-up hands, and complex scenes.

MANDATORY: Every BROLL_IMAGE prompt MUST describe a SCENE with a PERSON (woman/girl/female doctor/nurse/patient) doing something specific.
NEVER generate prompts without a person. NEVER generate prompts for abstract concepts.

TEMPLATE for every prompt: "[one person or two people] [simple script-based action] [in simple setting], realistic editorial photo, natural clinic lighting, medium shot, simple clean background, hands relaxed or out of frame, no text, no words, no letters"

HAND SAFETY:
- Avoid prompts focused on hands, fingers, injections, syringes, vials, reports, papers, phones, or small objects.
- Do not say "holding", "showing", "pointing", "touching", "folded hands", or "close up of hands".
- Prefer "doctor talking to patient", "patient listening", "woman sitting calmly", "doctor standing beside patient".
- Use medium shot or waist-up shot, never close-up hands.

CONCRETE EXAMPLES OF GOOD PROMPTS:
- "a young Indian woman sitting in a clean clinic waiting room looking calm, realistic editorial photo, natural clinic lighting, medium shot, hands relaxed or out of frame, no text, no words, no letters"
- "a female doctor talking to an Indian woman patient in a modern clinic, realistic editorial photo, natural clinic lighting, medium shot, simple clean background, hands relaxed or out of frame, no text, no words, no letters"
- "a pregnant Indian woman sitting peacefully in a bright clinic room, realistic editorial photo, natural clinic lighting, medium shot, hands relaxed or out of frame, no text, no words, no letters"
- "an Indian woman patient listening carefully to a female doctor across a clinic desk, realistic editorial photo, natural clinic lighting, medium shot, simple clean background, hands relaxed or out of frame, no text, no words, no letters"
- "a middle-aged Indian woman looking relieved after a health checkup in a clinic, realistic editorial photo, natural clinic lighting, medium shot, hands relaxed or out of frame, no text, no words, no letters"

NEVER USE THESE (THE MODEL CANNOT RENDER THEM):
- "infographic", "diagram", "chart", "statistics", "3D animation", "cross-section", "anatomical illustration", "medical diagram", "virus particles", "molecular structure", "flowchart", "comparison table", "X-ray", "microscope view", "abstract representation", "screenshot", "UI", "app interface", "text overlay", "title card", "label", "close up hands", "fingers"

DURATION: 5-6 seconds per BROLL_IMAGE. Keep images short and impactful. Change them frequently to keep viewer engaged.
Position: "center". Always set fade: "in-out". Set fx: "ken_burns_in" or "ken_burns_out" (alternate between them).
Generate AT LEAST 12-18 BROLL_IMAGE entries per video. Use one every 30-45 seconds. Cover the ENTIRE video from start to finish.

LOWER_THIRD: Use for speaker introductions only. Position: "bottom-left". Duration: 5-7 seconds.

TEXT_CARD: Quick pop-up overlays for key points the doctor mentions. Use these VERY FREQUENTLY — every time the doctor says a specific point like "pehla problem hai periods ka pain" or "HPV ka matlab hai Human Papillomavirus", add a quick 3-4 second TEXT_CARD that pops in and out fast. These should feel like quick emphasis pops, not lingering cards. Generate AT LEAST 15-25 TEXT_CARD entries per video — one every 20-30 seconds. Short punchy Hinglish statements. Position: "center". Duration: 3-4 seconds. Examples: "Periods Ka Pain?", "HPV = Human Papillomavirus", "9-45 Saal Vaccine Le Sakti Hain", "Pap Smear Zaroori Hai!".

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

        # Post-process: sanitize BROLL_IMAGE prompts and check density
        timeline_data = _sanitize_broll_prompts(timeline_data)
        timeline_data = _enforce_timeline_density(timeline_data, duration_sec)

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


# --- BROLL prompt sanitizer ---
BANNED_PROMPT_TERMS = [
    "infographic", "diagram", "chart", "statistics", "3D animation",
    "cross-section", "anatomical illustration", "medical diagram",
    "virus particles", "molecular structure", "flowchart",
    "comparison table", "X-ray", "x-ray", "microscope view",
    "abstract representation", "screenshot", "UI", "app interface",
    "text overlay", "title card", "label", "3d render", "3D render",
    "cutaway", "split screen", "before and after",
]

WATERCOLOR_SUFFIX = ", soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters"
NO_TEXT_SUFFIX = ", no text, no words, no letters, no watermark, no signature, no writing"


def _sanitize_broll_prompts(timeline_data: dict) -> dict:
    """Clean up BROLL_IMAGE prompts: strip banned terms, ensure style suffix."""
    if "timeline" not in timeline_data:
        return timeline_data

    sanitized_count = 0
    for entry in timeline_data["timeline"]:
        if entry.get("action") != "BROLL_IMAGE":
            continue

        prompt = entry["data"]
        original = prompt

        # Remove banned terms
        for term in BANNED_PROMPT_TERMS:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            prompt = pattern.sub("", prompt)

        # Clean up double commas/spaces left after removal
        prompt = re.sub(r",\s*,", ",", prompt)
        prompt = re.sub(r"\s{2,}", " ", prompt).strip().strip(",").strip()

        # Ensure watercolor style suffix
        if "watercolor" not in prompt.lower():
            prompt = prompt.rstrip(",. ") + WATERCOLOR_SUFFIX

        # Ensure no-text suffix
        if "no text" not in prompt.lower():
            prompt = prompt.rstrip(",. ") + NO_TEXT_SUFFIX

        # Ensure fade and fx defaults
        entry.setdefault("fade", "in-out")
        if "fx" not in entry:
            entry["fx"] = "ken_burns_in" if sanitized_count % 2 == 0 else "ken_burns_out"

        if prompt != original:
            sanitized_count += 1
            logger.info(f"Sanitized BROLL prompt for entry {entry.get('id')}: removed banned terms")

        entry["data"] = prompt

    if sanitized_count:
        logger.info(f"Sanitized {sanitized_count} BROLL_IMAGE prompts")
    return timeline_data


def _enforce_timeline_density(timeline_data: dict, duration_sec: float) -> dict:
    """Check timeline density and AUTO-FILL gaps where no overlays exist.
    
    If the AI director left segments >45s without any BROLL_IMAGE or TEXT_CARD,
    this function inserts filler entries to ensure full video coverage.
    """
    if "timeline" not in timeline_data or duration_sec <= 0:
        return timeline_data

    timeline = timeline_data["timeline"]
    broll_entries = [e for e in timeline if e.get("action") == "BROLL_IMAGE"]
    text_entries = [e for e in timeline if e.get("action") == "TEXT_CARD"]
    all_visual = broll_entries + text_entries

    if not all_visual:
        logger.warning("DENSITY CHECK: No BROLL_IMAGE or TEXT_CARD entries in timeline!")
        return timeline_data

    # Parse times and sort
    def _t(e):
        parts = e["time"].split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return 0

    all_visual.sort(key=_t)
    times = [_t(e) for e in all_visual]

    # Build list of gap intervals (>45s) that need filling
    GAP_THRESHOLD = 45  # seconds
    gaps = []

    # Check gap from start
    if times[0] > GAP_THRESHOLD:
        gaps.append((3, times[0]))

    # Check gaps between entries
    for i in range(len(times) - 1):
        gap_start = times[i] + 5  # account for entry duration
        gap_end = times[i + 1]
        if gap_end - gap_start > GAP_THRESHOLD:
            gaps.append((gap_start, gap_end))

    # Check gap from last entry to video end
    last_overlay_end = times[-1] + 5
    if duration_sec - last_overlay_end > GAP_THRESHOLD:
        gaps.append((last_overlay_end, duration_sec - 5))

    # Summary before filling
    last_overlay_time = times[-1] if times else 0
    logger.info(
        f"Timeline density: {len(broll_entries)} BROLL_IMAGE, {len(text_entries)} TEXT_CARD "
        f"across {duration_sec:.0f}s video. "
        f"Coverage: 0s to {last_overlay_time:.0f}s ({last_overlay_time/duration_sec*100:.0f}%)"
    )

    if not gaps:
        logger.info("Timeline density OK — no large gaps detected")
        return timeline_data

    logger.warning(f"DENSITY CHECK: Found {len(gaps)} gap(s) to auto-fill")

    # Generate filler entries for each gap
    max_existing_id = 0
    for e in timeline:
        try:
            max_existing_id = max(max_existing_id, int(e.get("id", 0)))
        except (ValueError, TypeError):
            pass

    filler_id = max_existing_id + 100  # start filler IDs at a safe offset
    inserted_count = 0

    # Safe generic BROLL prompts for filler (rotate through these)
    FILLER_BROLL_PROMPTS = [
        "a female doctor in a white coat smiling warmly at the camera in a modern clinic, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters",
        "a young Indian woman sitting comfortably and listening attentively in a doctor's office, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters",
        "a kind female doctor writing notes while talking to a woman patient, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters",
        "a young Indian couple holding hands supportively in a hospital waiting area, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters",
        "a female doctor showing care and empathy while examining a patient gently, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters",
        "an Indian woman looking hopeful and relaxed after visiting a female doctor, soft watercolor painting style, warm pastel colors, gentle lighting, no text, no words, no letters",
    ]

    FILLER_TEXT_CARDS = [
        "Doctor Se Zaroor Milein!",
        "Apni Health Ka Dhyan Rakhein",
        "Sahi Jaankari Zaroori Hai",
        "Regular Checkup Karwayein",
        "Dr. Deepika Se Expert Advice",
        "Aapki Sehat, Humari Zimmedari",
        "Timely Treatment Se Fayda",
        "Questions? Doctor Se Poochein!",
    ]

    broll_idx = 0
    text_idx = 0

    for gap_start, gap_end in gaps:
        gap_duration = gap_end - gap_start
        logger.info(f"Auto-filling gap: {gap_start:.0f}s → {gap_end:.0f}s ({gap_duration:.0f}s)")

        # Fill the gap by placing entries every ~30s
        t = gap_start + 5  # small offset from gap start
        while t < gap_end - 10:
            # Alternate: BROLL then TEXT_CARD
            if inserted_count % 2 == 0:
                # Insert a BROLL_IMAGE
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                entry = {
                    "id": filler_id,
                    "time": f"{h:02d}:{m:02d}:{s:02d}",
                    "duration": 5,
                    "action": "BROLL_IMAGE",
                    "data": FILLER_BROLL_PROMPTS[broll_idx % len(FILLER_BROLL_PROMPTS)],
                    "position": "center",
                    "fade": "in-out",
                    "fx": "ken_burns_in" if broll_idx % 2 == 0 else "ken_burns_out",
                    "_auto_filled": True,
                }
                broll_idx += 1
                t += 30  # next entry ~30s later
            else:
                # Insert a TEXT_CARD
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                entry = {
                    "id": filler_id,
                    "time": f"{h:02d}:{m:02d}:{s:02d}",
                    "duration": 3,
                    "action": "TEXT_CARD",
                    "data": FILLER_TEXT_CARDS[text_idx % len(FILLER_TEXT_CARDS)],
                    "position": "center",
                    "fade": "in-out",
                    "_auto_filled": True,
                }
                text_idx += 1
                t += 25  # next entry ~25s later

            timeline.append(entry)
            filler_id += 1
            inserted_count += 1

    if inserted_count:
        logger.info(f"Auto-filled {inserted_count} entries to cover timeline gaps")
        # Re-sort timeline by time
        timeline.sort(key=_t)

    return timeline_data


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
