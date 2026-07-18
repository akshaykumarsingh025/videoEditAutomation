import os
import json
import copy
import time
import logging
import subprocess
import random
import re
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

from config import (
    COMFYUI_DIR,
    COMFYUI_KEEP_ALIVE,
    COMFYUI_URL,
    COMFYUI_WORKFLOW,
    COMFYUI_STARTUP_TIMEOUT,
    COMFYUI_POLL_INTERVAL,
    PATHS,
    BRAND,
    RETRY_ATTEMPTS,
    RETRY_BASE_DELAY,
)

logger = logging.getLogger("pipeline.asset")

comfyui_process = None
comfyui_started_by_pipeline = False


def generate_comfyui_images(timeline_entries: list, asset_prefix: str = "") -> dict:
    comfyui_entries = [e for e in timeline_entries if e["action"] in ("COMFYUI_PROMPT", "BROLL_IMAGE")]
    if not comfyui_entries:
        logger.info("No ComfyUI images to generate")
        return {"generated": [], "failed": []}

    generated = []

    for entry in comfyui_entries:
        entry_id = entry["id"]
        dst = _generated_image_path(entry_id, asset_prefix)
        if dst.exists():
            dst.unlink()
            logger.info(f"Deleted old image for entry {entry_id}: {dst}")

    global comfyui_process, comfyui_started_by_pipeline
    comfyui_started_by_pipeline = False

    if _wait_for_comfyui(timeout=3, quiet=True):
        logger.info("Using existing ComfyUI server")
    else:
        _force_free_vram_for_comfyui()
        comfyui_process = _start_comfyui()
        comfyui_started_by_pipeline = comfyui_process is not None

    if not comfyui_process and not _wait_for_comfyui(timeout=3, quiet=True):
        logger.error("ComfyUI failed to start, skipping all image generation")
        return {"generated": [], "failed": [e["id"] for e in comfyui_entries]}

    if not _wait_for_comfyui(timeout=COMFYUI_STARTUP_TIMEOUT):
        _finish_comfyui_session()
        return {"generated": [], "failed": [e["id"] for e in comfyui_entries]}

    if not COMFYUI_WORKFLOW.exists():
        logger.error(f"ComfyUI workflow not found: {COMFYUI_WORKFLOW}")
        _finish_comfyui_session()
        return {"generated": [], "failed": [e["id"] for e in comfyui_entries]}

    try:
        workflow_template = json.loads(COMFYUI_WORKFLOW.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Could not load ComfyUI workflow: {e}")
        _finish_comfyui_session()
        return {"generated": [], "failed": [e["id"] for e in comfyui_entries]}

    failed = []

    for entry in comfyui_entries:
        entry_id = entry["id"]
        raw_prompt = entry["data"]
        is_broll = entry["action"] == "BROLL_IMAGE"

        if is_broll:
            prompt = raw_prompt
            prompt = _force_women_only(prompt)
            if "ultra-realistic" not in prompt.lower() and "photorealistic" not in prompt.lower():
                prompt = f"{prompt}, ultra-realistic photograph, professional studio lighting, shallow depth of field, 85mm lens, natural skin texture, cinematic color grading, 8K resolution, photorealistic, sharp focus"
            if "no text" not in prompt.lower():
                prompt = f"{prompt}, no text, no words, no letters, no watermark, no signature, no writing"
        else:
            prompt = raw_prompt

        logger.info(f"Generating {'B-roll' if is_broll else 'ComfyUI'} image for entry {entry_id}: {prompt[:80]}...")

        try:
            output_path = _generate_single_image(workflow_template, prompt, entry_id, asset_prefix, is_broll=is_broll)
            if output_path:
                generated.append({"id": entry_id, "path": str(output_path)})
            else:
                failed.append(entry_id)
        except Exception as e:
            logger.error(f"ComfyUI generation failed for entry {entry_id}: {e}")
            failed.append(entry_id)

    _finish_comfyui_session()
    logger.info(f"ComfyUI generation complete: {len(generated)} generated, {len(failed)} failed")
    return {"generated": generated, "failed": failed}


def download_web_assets(timeline_entries: list, asset_prefix: str = "") -> dict:
    gif_entries = [e for e in timeline_entries if e["action"] == "WEB_GIF"]
    if not gif_entries:
        logger.info("No web assets to download")
        return {"downloaded": [], "failed": []}

    downloaded = []
    failed = []

    for entry in gif_entries:
        search_term = entry["data"]
        output_base = PATHS["web"] / f"{_asset_prefix(asset_prefix)}gif_{_entry_token(entry['id'])}"

        try:
            result = _download_giphy_gif(search_term, output_base)
            if result:
                downloaded.append({"id": entry["id"], "path": str(result)})
            else:
                failed.append(entry["id"])
        except Exception as e:
            logger.error(f"Giphy download failed for entry {entry['id']}: {e}")
            failed.append(entry["id"])

    logger.info(f"Web assets: {len(downloaded)} downloaded, {len(failed)} failed")
    return {"downloaded": downloaded, "failed": failed}


def generate_graphics(timeline_entries: list, asset_prefix: str = "") -> dict:
    supported_actions = ("TEXT_CARD", "LOWER_THIRD", "QUOTE_CARD", "STAT_CARD", "LIST_CARD", "CTA_CARD", "CHAPTER_TITLE")
    graphics_entries = [e for e in timeline_entries if e["action"] in supported_actions]
    if not graphics_entries:
        logger.info("No static graphics to generate")
        return {"generated": [], "failed": []}

    generated = []
    failed = []

    card_renderers = {
        "TEXT_CARD": _create_text_card,
        "LOWER_THIRD": _create_lower_third,
        "QUOTE_CARD": _create_quote_card,
        "STAT_CARD": _create_stat_card,
        "LIST_CARD": _create_list_card,
        "CTA_CARD": _create_cta_card,
        "CHAPTER_TITLE": _create_chapter_title,
    }

    for entry in graphics_entries:
        try:
            renderer = card_renderers.get(entry["action"])
            if renderer:
                output_path = renderer(entry, asset_prefix)
            else:
                continue

            if output_path:
                generated.append({"id": entry["id"], "path": str(output_path)})
            else:
                failed.append(entry["id"])
        except Exception as e:
            logger.error(f"Graphic generation failed for entry {entry['id']}: {e}")
            failed.append(entry["id"])

    logger.info(f"Graphics: {len(generated)} generated, {len(failed)} failed")
    return {"generated": generated, "failed": failed}


def _start_comfyui():
    logger.info("Starting ComfyUI server...")
    if not COMFYUI_DIR.exists():
        logger.error(f"ComfyUI directory not found: {COMFYUI_DIR}")
        return None
    try:
        python_exe = COMFYUI_DIR.parent / "python_embeded" / "python.exe"
        if not python_exe.exists():
            python_exe = "python"

        proc = subprocess.Popen(
            [str(python_exe), "main.py", "--listen", "127.0.0.1", "--port", "8188"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(COMFYUI_DIR),
        )
        logger.info(f"ComfyUI process started (PID: {proc.pid})")
        return proc
    except Exception as e:
        logger.error(f"Failed to start ComfyUI: {e}")
        return None


def _wait_for_comfyui(timeout: int | float | None = None, quiet: bool = False) -> bool:
    timeout = COMFYUI_STARTUP_TIMEOUT if timeout is None else timeout
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if resp.status_code == 200:
                logger.info("ComfyUI is ready")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(min(COMFYUI_POLL_INTERVAL, 5))

    if not quiet:
        logger.error(f"ComfyUI did not become ready within {timeout}s")
    return False


def _stop_comfyui():
    global comfyui_process, comfyui_started_by_pipeline
    if comfyui_process and comfyui_started_by_pipeline:
        logger.info("Stopping ComfyUI...")
        comfyui_process.terminate()
        try:
            comfyui_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            comfyui_process.kill()
        comfyui_process = None
        comfyui_started_by_pipeline = False
        logger.info("ComfyUI stopped, VRAM freed")
    else:
        logger.info("ComfyUI was started externally or already stopped — skipping process kill")
        logger.info("If ComfyUI is still running, it will hold VRAM. Stop it manually if needed.")


def _finish_comfyui_session():
    global comfyui_process, comfyui_started_by_pipeline
    if COMFYUI_KEEP_ALIVE:
        if comfyui_started_by_pipeline:
            logger.info("Leaving ComfyUI running for the rest of the pipeline (COMFYUI_KEEP_ALIVE=true)")
        else:
            logger.info("ComfyUI was external/already running; leaving it running")
        comfyui_process = None
        comfyui_started_by_pipeline = False
        return

    _stop_comfyui()


def _generate_single_image(workflow_template: dict, prompt: str, entry_id, asset_prefix: str = "", is_broll: bool = False) -> Path | None:
    clean_prompt = _clean_prompt_for_comfyui(prompt)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        workflow = copy.deepcopy(workflow_template)
        current_prompt = clean_prompt if attempt == 1 else _simplify_prompt(clean_prompt)

        workflow["30:19"]["inputs"]["value"] = current_prompt
        workflow["30:3"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
        workflow["29"]["inputs"]["filename_prefix"] = f"{_asset_prefix(asset_prefix)}gen_{_entry_token(entry_id)}"

        logger.info(f"ComfyUI attempt {attempt} for entry {entry_id}: {current_prompt[:100]}...")

        result = _submit_and_poll(workflow, entry_id, asset_prefix, is_broll)
        if result is not None:
            return result

        if attempt < RETRY_ATTEMPTS:
            logger.warning(f"Retrying entry {entry_id} with simplified prompt (attempt {attempt + 1})")
            time.sleep(2)

    logger.error(f"All {RETRY_ATTEMPTS} attempts failed for entry {entry_id}")
    return None


def _force_women_only(prompt: str) -> str:
    male_terms = [
        (r'\bman\b', 'woman'), (r'\bMan\b', 'Woman'),
        (r'\bmen\b', 'women'), (r'\bMen\b', 'Women'),
        (r'\bboy\b', 'girl'), (r'\bBoy\b', 'Girl'),
        (r'\bboys\b', 'girls'), (r'\bBoys\b', 'Girls'),
        (r'\bmale\b', 'female'), (r'\bMale\b', 'Female'),
        (r'\bgentleman\b', 'lady'), (r'\bGentleman\b', 'Lady'),
        (r'\bguy\b', 'woman'), (r'\bGuy\b', 'Woman'),
        (r'\bguy\b', 'woman'), (r'\bguys\b', 'women'),
        (r'\bhe\b', 'she'), (r'\bHe\b', 'She'),
        (r'\bhim\b', 'her'), (r'\bhis\b', 'her'),
    ]
    for pattern, replacement in male_terms:
        prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
    has_female = any(w in prompt.lower() for w in ['woman', 'women', 'girl', 'lady', 'female', 'mother', 'sister', 'daughter', 'wife', 'aunt', 'pregnant'])
    if not has_female:
        prompt = f"an Indian woman {prompt.lstrip('aAanAn ')}"
    return prompt


def _clean_prompt_for_comfyui(prompt: str) -> str:
    cleaned = prompt
    for ch in ['"', "'", "`", "\\", "<", ">", "{", "}", "[", "]", "|"]:
        cleaned = cleaned.replace(ch, "")
    cleaned = re.sub(r'[^\x20-\x7E\n,]', ' ', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    if len(cleaned) > 450:
        cleaned = cleaned[:447].rsplit(',', 1)[0].rstrip(', ')
    return cleaned


def _simplify_prompt(prompt: str) -> str:
    simple = prompt
    for phrase in [
        "ultra-realistic photograph,", "professional studio lighting,",
        "shallow depth of field,", "85mm lens,", "natural skin texture,",
        "cinematic color grading,", "8K resolution,", "photorealistic,",
        "sharp focus,", "professional lighting,", "volumetric lighting,",
        "rim lighting,", "soft shadows,",
    ]:
        simple = simple.replace(phrase, "")
    simple = re.sub(r',\s*,', ',', simple)
    simple = re.sub(r'\s{2,}', ' ', simple).strip().strip(',').strip()
    if len(simple) > 250:
        simple = simple[:247].rsplit(',', 1)[0].rstrip(', ')
    simple += ", realistic photo, no text, no watermark"
    return simple


def _submit_and_poll(workflow: dict, entry_id, asset_prefix: str, is_broll: bool) -> Path | None:
    try:
        resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=30)
    except Exception as e:
        logger.error(f"Failed to submit prompt to ComfyUI: {e}")
        return None

    resp_json = resp.json()

    if resp.status_code != 200 or "error" in resp_json:
        error_msg = resp_json.get("error", {})
        node_errors = resp_json.get("node_errors", {})
        logger.error(f"ComfyUI prompt rejected for entry {entry_id}: {error_msg}")
        if node_errors:
            for node_id, errs in node_errors.items():
                logger.error(f"  Node {node_id}: {errs}")
        return None

    prompt_id = resp_json.get("prompt_id")
    if not prompt_id:
        logger.error(f"No prompt_id in ComfyUI response for entry {entry_id}: {resp_json}")
        return None

    logger.info(f"ComfyUI prompt submitted: {prompt_id}")

    start = time.time()
    max_wait = 300
    while time.time() - start < max_wait:
        try:
            hist_resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            hist_data = hist_resp.json()

            if prompt_id in hist_data:
                status = hist_data[prompt_id].get("status", {})
                if status.get("completed", False) or status.get("status_str") == "success":
                    outputs = hist_data[prompt_id].get("outputs", {})
                    logger.info(f"ComfyUI outputs keys: {list(outputs.keys())}")

                    image_data = None
                    for node_id, node_output in outputs.items():
                        if "images" in node_output and node_output["images"]:
                            image_data = node_output
                            image_node_id = node_id
                            break

                    if image_data:
                        images = image_data["images"]
                        img_info = images[0]
                        filename = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")

                        logger.info(f"ComfyUI image found: node={image_node_id}, file={filename}, subfolder={subfolder}, type={img_type}")

                        if img_type == "temp":
                            comfyui_output = COMFYUI_DIR / "temp"
                        else:
                            output_candidates = [
                                COMFYUI_DIR / "output",
                                Path("D:/Software/ComfyUI_windows_portable/ComfyUI/output"),
                                Path.home() / "ComfyUI" / "output",
                            ]

                            comfyui_output = Path("")
                            for candidate in output_candidates:
                                if candidate.exists():
                                    comfyui_output = candidate
                                    logger.info(f"Found ComfyUI output dir: {comfyui_output}")
                                    break

                        if not comfyui_output.exists():
                            logger.error(f"ComfyUI output directory not found. Tried: {[str(c) for c in output_candidates]}")
                            return None

                        src = comfyui_output / subfolder / filename
                        dst = _generated_image_path(entry_id, asset_prefix)

                        logger.info(f"Looking for ComfyUI output: {src}")
                        if src.exists():
                            img = Image.open(src)
                            img = _fit_image(img, (1920, 1080))
                            img.save(dst)
                            logger.info(f"Image saved: {dst}")
                            return dst
                        else:
                            logger.error(f"ComfyUI image file not found at: {src}")
                            try:
                                for f in comfyui_output.iterdir():
                                    if f.is_file() and f.suffix in ('.png', '.jpg', '.webp'):
                                        logger.info(f"  {f.name} ({f.stat().st_size // 1024} KB)")
                            except Exception:
                                pass
                            return None
                    else:
                        logger.error(f"ComfyUI completed but no image data in outputs for entry {entry_id}. Outputs: {list(outputs.keys())}")
                        return None

                if status.get("status_str") == "error":
                    error_msg = status.get("message", "unknown error")
                    logger.error(f"ComfyUI generation error for entry {entry_id}: {error_msg}")
                    return None

        except requests.ConnectionError:
            pass
        except Exception as e:
            logger.warning(f"Error polling ComfyUI: {e}")

        time.sleep(COMFYUI_POLL_INTERVAL)

    logger.error(f"ComfyUI generation timed out for entry {entry_id}")
    return None


def _download_giphy_gif(search_term: str, output_base: Path) -> Path | None:
    giphy_api_key = os.environ.get("GIPHY_API_KEY", "")
    if not giphy_api_key:
        logger.warning("GIPHY_API_KEY not set, skipping GIF download")
        return None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(
                "https://api.giphy.com/v1/gifs/search",
                params={"api_key": giphy_api_key, "q": search_term, "limit": 1, "rating": "pg"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data.get("data"):
                logger.warning(f"No GIF results for '{search_term}'")
                return None

            images = data["data"][0]["images"]
            asset_url = (
                images.get("original_mp4", {}).get("mp4")
                or images.get("original", {}).get("mp4")
                or images.get("downsized_medium", {}).get("url")
                or images.get("original", {}).get("url")
            )
            if not asset_url:
                logger.warning(f"No downloadable media URL for '{search_term}'")
                return None

            suffix = ".mp4" if ".mp4" in asset_url.split("?", 1)[0].lower() else ".gif"
            output_path = output_base.with_suffix(suffix)

            gif_resp = requests.get(asset_url, timeout=30, stream=True)
            gif_resp.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in gif_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"GIF downloaded: {output_path}")
            return output_path

        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(f"Giphy download attempt {attempt} failed: {e}. Retrying in {delay}s")
            time.sleep(delay)

    return None


def _create_text_card(entry: dict, asset_prefix: str = "") -> Path | None:
    width, height = 1920, 1080
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    style = BRAND.get("text_card_style", {})
    bg_hex = style.get("bg_color", BRAND["brand"].get("card_bg_color", "#0D1B2A"))
    accent_hex = style.get("accent_bar_color", BRAND["brand"].get("card_accent_color", "#E8734A"))
    text_hex = style.get("text_color", BRAND["brand"].get("card_text_color", "#FFFFFF"))
    sub_hex = style.get("subtext_color", BRAND["brand"].get("card_subtext_color", "#94A3B8"))
    corner_r = style.get("corner_radius", 16)
    pad_x = style.get("padding_x", 60)
    pad_y = style.get("padding_y", 30)
    border_w = style.get("border_width", 2)
    border_hex = style.get("border_color", accent_hex)

    text = entry["data"]
    font_size = BRAND["fonts"]["text_card_size"]
    font = _load_font(BRAND["fonts"]["text_card"], font_size)
    sub_font = _load_font(BRAND["fonts"]["text_card"], int(font_size * 0.45))

    lines = text.split("\n") if "\n" in text else [text]
    main_text = lines[0]
    sub_text = lines[1] if len(lines) > 1 else ""

    main_bbox = draw.textbbox((0, 0), main_text, font=font)
    main_tw = main_bbox[2] - main_bbox[0]
    main_th = main_bbox[3] - main_bbox[1]

    sub_tw, sub_th = 0, 0
    if sub_text:
        sub_bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
        sub_tw = sub_bbox[2] - sub_bbox[0]
        sub_th = sub_bbox[3] - sub_bbox[1]

    card_w = max(main_tw, sub_tw) + pad_x * 2
    card_h = main_th + sub_th + pad_y * 3 + 10
    card_w = max(card_w, 500)
    card_h = max(card_h, 160)

    cx = (width - card_w) // 2
    y_ratio = style.get("y_position_ratio", 0.72)
    cy = int((height - card_h) * y_ratio)

    bg_rgb = _hex_to_rgba(bg_hex, 225)
    draw.rounded_rectangle(
        [(cx, cy), (cx + card_w, cy + card_h)],
        radius=corner_r,
        fill=bg_rgb,
    )

    border_rgb = _hex_to_rgba(border_hex, 180)
    draw.rounded_rectangle(
        [(cx, cy), (cx + card_w, cy + card_h)],
        radius=corner_r,
        fill=None,
        outline=border_rgb,
        width=border_w,
    )

    accent_rgb = _hex_to_rgba(accent_hex, 255)
    accent_bar_h = 5
    draw.rounded_rectangle(
        [(cx + corner_r, cy + card_h - accent_bar_h - 8), (cx + card_w - corner_r, cy + card_h - 8)],
        radius=3,
        fill=accent_rgb,
    )

    top_accent_w = 6
    draw.rounded_rectangle(
        [(cx, cy), (cx + top_accent_w, cy + card_h)],
        radius=3,
        fill=accent_rgb,
    )

    text_rgb = _hex_to_rgb(text_hex)
    sub_rgb = _hex_to_rgb(sub_hex)

    text_x = cx + pad_x
    text_y = cy + pad_y
    draw.text((text_x, text_y), main_text, fill=text_rgb, font=font)

    if sub_text:
        draw.text((text_x, text_y + main_th + 14), sub_text, fill=sub_rgb, font=sub_font)

    output_path = PATHS["graphics"] / f"{_asset_prefix(asset_prefix)}card_{_entry_token(entry['id'])}.png"
    img.save(output_path)
    logger.info(f"Text card saved: {output_path}")
    return output_path


def _create_lower_third(entry: dict, asset_prefix: str = "") -> Path | None:
    style = BRAND.get("lower_third_style", {})
    bg_hex = style.get("bg_color", BRAND["brand"].get("card_bg_color", "#0D1B2A"))
    accent_hex = style.get("accent_bar_color", BRAND["brand"].get("card_accent_color", "#E8734A"))
    text_hex = style.get("text_color", BRAND["brand"].get("card_text_color", "#FFFFFF"))
    sub_hex = style.get("subtext_color", BRAND["brand"].get("card_subtext_color", "#94A3B8"))
    bar_height = style.get("height", 100)
    corner_r = style.get("corner_radius", 8)
    accent_w = style.get("accent_bar_width", 6)

    width, height = 1920, bar_height + 30
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bar_y = 15
    bg_rgb = _hex_to_rgba(bg_hex, 220)
    draw.rounded_rectangle(
        [(40, bar_y), (width - 40, bar_y + bar_height)],
        radius=corner_r,
        fill=bg_rgb,
    )

    accent_rgb = _hex_to_rgba(accent_hex, 255)
    draw.rounded_rectangle(
        [(40, bar_y), (40 + accent_w, bar_y + bar_height)],
        radius=3,
        fill=accent_rgb,
    )

    bottom_accent_y = bar_y + bar_height - 4
    draw.rounded_rectangle(
        [(40 + accent_w, bottom_accent_y), (width - 40, bar_y + bar_height)],
        radius=2,
        fill=accent_rgb,
    )

    text = entry["data"]
    lines = text.split("\n") if "\n" in text else [text]
    main_text = lines[0]
    sub_text = lines[1] if len(lines) > 1 else ""

    main_font_size = BRAND["fonts"]["lower_third_size"]
    main_font = _load_font(BRAND["fonts"]["lower_third"], main_font_size)
    sub_font = _load_font(BRAND["fonts"]["lower_third"], int(main_font_size * 0.6))

    text_rgb = _hex_to_rgb(text_hex)
    sub_rgb = _hex_to_rgb(sub_hex)

    x = 40 + accent_w + 20

    if sub_text:
        draw.text((x, bar_y + 14), main_text, fill=text_rgb, font=main_font)
        draw.text((x, bar_y + 14 + main_font_size + 4), sub_text, fill=sub_rgb, font=sub_font)
    else:
        main_bbox = draw.textbbox((0, 0), main_text, font=main_font)
        main_th = main_bbox[3] - main_bbox[1]
        draw.text((x, bar_y + (bar_height - main_th) // 2), main_text, fill=text_rgb, font=main_font)

    output_path = PATHS["graphics"] / f"{_asset_prefix(asset_prefix)}lower_third_{_entry_token(entry['id'])}.png"
    img.save(output_path)
    logger.info(f"Lower third saved: {output_path}")
    return output_path


def _create_quote_card(entry: dict, asset_prefix: str = "") -> Path | None:
    width, height = 1920, 1080
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    style = BRAND.get("quote_card_style", {})
    bg_hex = style.get("bg_color", "#0D1B2A")
    accent_hex = style.get("accent_color", "#E8734A")
    text_hex = style.get("text_color", "#FFFFFF")
    quote_mark_hex = style.get("quote_mark_color", "#E8734A")
    corner_r = style.get("corner_radius", 18)
    pad_x = style.get("padding_x", 60)
    pad_y = style.get("padding_y", 40)

    text = entry["data"]
    lines = text.split("\n") if "\n" in text else [text]
    quote_text = lines[0]
    attribution = lines[1] if len(lines) > 1 else ""

    font_size = int(BRAND["fonts"]["text_card_size"] * style.get("font_size_ratio", 0.7))
    font = _load_font(BRAND["fonts"]["text_card"], font_size)
    attr_font = _load_font(BRAND["fonts"]["lower_third"], int(font_size * 0.5))
    quote_font = _load_font(BRAND["fonts"]["text_card"], int(font_size * 1.8))

    quote_mark = "\u201C"
    quote_mark_bbox = draw.textbbox((0, 0), quote_mark, font=quote_font)
    quote_mark_w = quote_mark_bbox[2] - quote_mark_bbox[0]
    quote_mark_h = quote_mark_bbox[3] - quote_mark_bbox[1]

    wrapped = _wrap_text(draw, quote_text, font, int(width * 0.7))
    line_height = draw.textbbox((0, 0), "Ay", font=font)[3] + 10
    total_text_h = len(wrapped) * line_height
    if attribution:
        total_text_h += int(font_size * 0.5) + 20

    card_h = total_text_h + pad_y * 2 + quote_mark_h // 2
    card_w = width - 200
    cx = (width - card_w) // 2
    cy = (height - card_h) // 2

    bg_rgb = _hex_to_rgba(bg_hex, 230)
    draw.rounded_rectangle([(cx, cy), (cx + card_w, cy + card_h)], radius=corner_r, fill=bg_rgb)

    border_rgb = _hex_to_rgba(accent_hex, 180)
    draw.rounded_rectangle([(cx, cy), (cx + card_w, cy + card_h)], radius=corner_r, fill=None, outline=border_rgb, width=2)

    accent_rgb = _hex_to_rgba(accent_hex, 255)
    draw.rounded_rectangle([(cx, cy), (cx + 6, cy + card_h)], radius=3, fill=accent_rgb)

    quote_mark_rgb = _hex_to_rgb(quote_mark_hex)
    text_rgb = _hex_to_rgb(text_hex)

    text_x = cx + pad_x + 6
    draw.text((text_x, cy + pad_y - quote_mark_h // 3), quote_mark, fill=quote_mark_rgb, font=quote_font)

    for i, line in enumerate(wrapped):
        draw.text((text_x, cy + pad_y + quote_mark_h // 3 + i * line_height), line, fill=text_rgb, font=font)

    if attribution:
        attr_y = cy + pad_y + quote_mark_h // 3 + len(wrapped) * line_height + 20
        attr_rgb = _hex_to_rgb(accent_hex)
        draw.text((text_x, attr_y), f"\u2014 {attribution}", fill=attr_rgb, font=attr_font)

    output_path = PATHS["graphics"] / f"{_asset_prefix(asset_prefix)}quote_card_{_entry_token(entry['id'])}.png"
    img.save(output_path)
    logger.info(f"Quote card saved: {output_path}")
    return output_path


def _create_stat_card(entry: dict, asset_prefix: str = "") -> Path | None:
    width, height = 1920, 1080
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    style = BRAND.get("stat_card_style", {})
    bg_hex = style.get("bg_color", "#0D1B2A")
    accent_hex = style.get("accent_color", "#E8734A")
    number_hex = style.get("number_color", "#E8734A")
    label_hex = style.get("label_color", "#CBD5E1")
    corner_r = style.get("corner_radius", 18)
    pad_x = style.get("padding_x", 60)
    pad_y = style.get("padding_y", 40)

    text = entry["data"]
    lines = text.split("\n") if "\n" in text else [text]
    stat_number = lines[0]
    stat_label = lines[1] if len(lines) > 1 else ""

    number_font = _load_font(BRAND["fonts"]["text_card"], 120)
    label_font = _load_font(BRAND["fonts"]["text_card"], int(BRAND["fonts"]["text_card_size"] * 0.5))

    num_bbox = draw.textbbox((0, 0), stat_number, font=number_font)
    num_w = num_bbox[2] - num_bbox[0]
    num_h = num_bbox[3] - num_bbox[1]

    lbl_bbox = draw.textbbox((0, 0), stat_label, font=label_font) if stat_label else (0, 0, 0, 0)
    lbl_w = lbl_bbox[2] - lbl_bbox[0]
    lbl_h = lbl_bbox[3] - lbl_bbox[1] if stat_label else 0

    card_w = max(num_w, lbl_w) + pad_x * 2
    card_h = num_h + lbl_h + pad_y * 3 + 20
    card_w = max(card_w, 400)

    cx = (width - card_w) // 2
    cy = int((height - card_h) * 0.72)

    bg_rgb = _hex_to_rgba(bg_hex, 230)
    draw.rounded_rectangle([(cx, cy), (cx + card_w, cy + card_h)], radius=corner_r, fill=bg_rgb)

    border_rgb = _hex_to_rgba(accent_hex, 180)
    draw.rounded_rectangle([(cx, cy), (cx + card_w, cy + card_h)], radius=corner_r, fill=None, outline=border_rgb, width=2)

    accent_rgb = _hex_to_rgba(accent_hex, 255)
    draw.rounded_rectangle([(cx, cy), (cx + 6, cy + card_h)], radius=3, fill=accent_rgb)

    bottom_accent_y = cy + card_h - 5
    draw.rounded_rectangle([(cx + 10, bottom_accent_y), (cx + card_w - 10, cy + card_h)], radius=2, fill=accent_rgb)

    number_rgb = _hex_to_rgb(number_hex)
    label_rgb = _hex_to_rgb(label_hex)

    draw.text(((width - num_w) // 2, cy + pad_y), stat_number, fill=number_rgb, font=number_font)

    if stat_label:
        draw.text(((width - lbl_w) // 2, cy + pad_y + num_h + 16), stat_label, fill=label_rgb, font=label_font)

    output_path = PATHS["graphics"] / f"{_asset_prefix(asset_prefix)}stat_card_{_entry_token(entry['id'])}.png"
    img.save(output_path)
    logger.info(f"Stat card saved: {output_path}")
    return output_path


def _create_list_card(entry: dict, asset_prefix: str = "") -> Path | None:
    width, height = 1920, 1080
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    style = BRAND.get("list_card_style", {})
    bg_hex = style.get("bg_color", "#0D1B2A")
    accent_hex = style.get("accent_color", "#E8734A")
    text_hex = style.get("text_color", "#FFFFFF")
    bullet_hex = style.get("bullet_color", "#E8734A")
    corner_r = style.get("corner_radius", 18)
    pad_x = style.get("padding_x", 60)
    pad_y = style.get("padding_y", 40)

    text = entry["data"]
    items = [line.strip() for line in text.split("\n") if line.strip()]

    title = ""
    list_items = items
    if items and not items[0].startswith(("•", "-", "*", "→")):
        title = items[0]
        list_items = items[1:]

    item_font = _load_font(BRAND["fonts"]["text_card"], int(BRAND["fonts"]["text_card_size"] * 0.5))
    title_font = _load_font(BRAND["fonts"]["text_card"], BRAND["fonts"]["text_card_size"])

    line_height = draw.textbbox((0, 0), "Ay", font=item_font)[3] + 16
    title_h = 0
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_h = title_bbox[3] - title_bbox[1] + 24

    total_h = title_h + len(list_items) * line_height + pad_y * 2

    max_item_w = 0
    for item in list_items:
        clean = item.lstrip("•-*→ ").strip()
        bbox = draw.textbbox((0, 0), clean, font=item_font)
        max_item_w = max(max_item_w, bbox[2] - bbox[0])

    if title:
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        max_item_w = max(max_item_w, title_bbox[2] - title_bbox[0])

    card_w = max_item_w + pad_x * 2 + 40
    card_h = total_h

    cx = (width - card_w) // 2
    cy = int((height - card_h) * 0.72)

    bg_rgb = _hex_to_rgba(bg_hex, 230)
    draw.rounded_rectangle([(cx, cy), (cx + card_w, cy + card_h)], radius=corner_r, fill=bg_rgb)

    border_rgb = _hex_to_rgba(accent_hex, 180)
    draw.rounded_rectangle([(cx, cy), (cx + card_w, cy + card_h)], radius=corner_r, fill=None, outline=border_rgb, width=2)

    accent_rgb = _hex_to_rgba(accent_hex, 255)
    draw.rounded_rectangle([(cx, cy), (cx + 6, cy + card_h)], radius=3, fill=accent_rgb)

    text_rgb = _hex_to_rgb(text_hex)
    bullet_rgb = _hex_to_rgb(bullet_hex)

    y = cy + pad_y
    if title:
        draw.text((cx + pad_x + 6, y), title, fill=text_rgb, font=title_font)
        y += title_h

    for item in list_items:
        clean = item.lstrip("•-*→ ").strip()
        draw.text((cx + pad_x + 6, y), "→", fill=bullet_rgb, font=item_font)
        draw.text((cx + pad_x + 36, y), clean, fill=text_rgb, font=item_font)
        y += line_height

    output_path = PATHS["graphics"] / f"{_asset_prefix(asset_prefix)}list_card_{_entry_token(entry['id'])}.png"
    img.save(output_path)
    logger.info(f"List card saved: {output_path}")
    return output_path


def _create_cta_card(entry: dict, asset_prefix: str = "") -> Path | None:
    width, height = 1920, 1080
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    style = BRAND.get("cta_card_style", {})
    bg_hex = style.get("bg_color", "#0D1B2A")
    accent_hex = style.get("accent_color", "#E8734A")
    text_hex = style.get("text_color", "#FFFFFF")
    sub_hex = style.get("subtext_color", "#CBD5E1")
    btn_hex = style.get("button_color", "#E8734A")
    btn_text_hex = style.get("button_text_color", "#FFFFFF")
    corner_r = style.get("corner_radius", 18)
    pad_x = style.get("padding_x", 60)
    pad_y = style.get("padding_y", 40)

    text = entry["data"]
    lines = text.split("\n") if "\n" in text else [text]
    headline = lines[0]
    sub_text = lines[1] if len(lines) > 1 else ""
    btn_text = lines[2] if len(lines) > 2 else "Subscribe"

    headline_font = _load_font(BRAND["fonts"]["text_card"], BRAND["fonts"]["text_card_size"])
    sub_font = _load_font(BRAND["fonts"]["lower_third"], int(BRAND["fonts"]["text_card_size"] * 0.45))
    btn_font = _load_font(BRAND["fonts"]["text_card"], int(BRAND["fonts"]["text_card_size"] * 0.5))

    hl_bbox = draw.textbbox((0, 0), headline, font=headline_font)
    hl_w = hl_bbox[2] - hl_bbox[0]
    hl_h = hl_bbox[3] - hl_bbox[1]

    sub_w = 0
    sub_h = 0
    if sub_text:
        sb_bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
        sub_w = sb_bbox[2] - sb_bbox[0]
        sub_h = sb_bbox[3] - sb_bbox[1]

    btn_bbox = draw.textbbox((0, 0), btn_text, font=btn_font)
    btn_w = btn_bbox[2] - btn_bbox[0]
    btn_h = btn_bbox[3] - btn_bbox[1]

    btn_pad_x = 40
    btn_pad_y = 16
    btn_box_w = btn_w + btn_pad_x * 2
    btn_box_h = btn_h + btn_pad_y * 2

    card_w = max(hl_w, sub_w, btn_box_w) + pad_x * 2
    card_h = hl_h + (sub_h + 16 if sub_text else 0) + btn_box_h + 30 + pad_y * 2

    cx = (width - card_w) // 2
    cy = (height - card_h) // 2

    bg_rgb = _hex_to_rgba(bg_hex, 230)
    draw.rounded_rectangle([(cx, cy), (cx + card_w, cy + card_h)], radius=corner_r, fill=bg_rgb)

    border_rgb = _hex_to_rgba(accent_hex, 180)
    draw.rounded_rectangle([(cx, cy), (cx + card_w, cy + card_h)], radius=corner_r, fill=None, outline=border_rgb, width=2)

    text_rgb = _hex_to_rgb(text_hex)
    sub_rgb = _hex_to_rgb(sub_hex)
    btn_text_rgb = _hex_to_rgb(btn_text_hex)
    btn_rgb = _hex_to_rgb(btn_hex)

    y = cy + pad_y
    draw.text(((width - hl_w) // 2, y), headline, fill=text_rgb, font=headline_font)
    y += hl_h + 16

    if sub_text:
        draw.text(((width - sub_w) // 2, y), sub_text, fill=sub_rgb, font=sub_font)
        y += sub_h + 16

    y += 14
    btn_x = (width - btn_box_w) // 2
    draw.rounded_rectangle([(btn_x, y), (btn_x + btn_box_w, y + btn_box_h)], radius=12, fill=btn_rgb)
    draw.text(((width - btn_w) // 2, y + btn_pad_y), btn_text, fill=btn_text_rgb, font=btn_font)

    output_path = PATHS["graphics"] / f"{_asset_prefix(asset_prefix)}cta_card_{_entry_token(entry['id'])}.png"
    img.save(output_path)
    logger.info(f"CTA card saved: {output_path}")
    return output_path


def _create_chapter_title(entry: dict, asset_prefix: str = "") -> Path | None:
    width, height = 1920, 1080
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    style = BRAND.get("chapter_title_style", {})
    bg_hex = style.get("bg_color", "#0D1B2A")
    accent_hex = style.get("accent_color", "#E8734A")
    text_hex = style.get("text_color", "#FFFFFF")
    sub_hex = style.get("subtext_color", "#CBD5E1")
    pad_x = style.get("padding_x", 80)
    pad_y = style.get("padding_y", 60)

    text = entry["data"]
    lines = text.split("\n") if "\n" in text else [text]
    chapter_title = lines[0]
    chapter_sub = lines[1] if len(lines) > 1 else ""

    title_font = _load_font(BRAND["fonts"]["text_card"], int(BRAND["fonts"]["text_card_size"] * 1.2))
    sub_font = _load_font(BRAND["fonts"]["lower_third"], int(BRAND["fonts"]["text_card_size"] * 0.45))

    draw.rounded_rectangle([(0, 0), (width, height)], radius=0, fill=_hex_to_rgba(bg_hex, 240))

    accent_rgb = _hex_to_rgba(accent_hex, 255)
    line_y = height // 2 - 40
    draw.rectangle([(pad_x, line_y), (width - pad_x, line_y + 3)], fill=accent_rgb)

    title_rgb = _hex_to_rgb(text_hex)
    sub_rgb = _hex_to_rgb(sub_hex)

    t_bbox = draw.textbbox((0, 0), chapter_title, font=title_font)
    t_w = t_bbox[2] - t_bbox[0]
    draw.text(((width - t_w) // 2, line_y - t_bbox[3] - 24), chapter_title, fill=title_rgb, font=title_font)

    if chapter_sub:
        s_bbox = draw.textbbox((0, 0), chapter_sub, font=sub_font)
        s_w = s_bbox[2] - s_bbox[0]
        draw.text(((width - s_w) // 2, line_y + 24), chapter_sub, fill=sub_rgb, font=sub_font)

    output_path = PATHS["graphics"] / f"{_asset_prefix(asset_prefix)}chapter_title_{_entry_token(entry['id'])}.png"
    img.save(output_path)
    logger.info(f"Chapter title saved: {output_path}")
    return output_path


def _load_font(font_name: str, size: int) -> ImageFont.FreeTypeFont:
    font_path = PATHS["fonts"] / font_name
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    logger.warning(f"Font not found: {font_path}, using default")
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_wrapped_text(draw, text, font, color, cx, cy, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test

    if current_line:
        lines.append(current_line)

    line_height = draw.textbbox((0, 0), "Ay", font=font)[3] + 10
    total_h = len(lines) * line_height
    start_y = cy - total_h // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = cx - w // 2
        y = start_y + i * line_height
        draw.text((x, y), line, fill=color, font=font)


def _hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _hex_to_rgba(hex_color: str, alpha: int) -> tuple:
    return _hex_to_rgb(hex_color) + (alpha,)


def _wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test
    if current_line:
        lines.append(current_line)
    return lines


def _asset_prefix(asset_prefix: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(asset_prefix or "").strip())
    return f"{clean}_" if clean else ""


def _entry_token(entry_id) -> str:
    try:
        return f"{int(entry_id):03d}"
    except (TypeError, ValueError):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(entry_id))


def _generated_image_path(entry_id, asset_prefix: str = "") -> Path:
    return PATHS["gen_images"] / f"{_asset_prefix(asset_prefix)}gen_{_entry_token(entry_id)}.png"


def _crop_to_16_9(img: Image.Image, target_size: tuple) -> Image.Image:
    tw, th = target_size
    target_ratio = tw / th
    w, h = img.size
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    return img.resize(target_size, Image.LANCZOS)


def _fit_image(img: Image.Image, target_size: tuple) -> Image.Image:
    tw, th = target_size
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    w, h = img.size
    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    x = (tw - new_w) // 2
    y = (th - new_h) // 2
    canvas.paste(img_resized, (x, y))
    return canvas


def _force_free_vram_for_comfyui():
    import gc as _gc
    import subprocess as _sp
    import time as _time

    logger.info("Force-freeing VRAM before ComfyUI starts...")

    try:
        result = _sp.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.splitlines()[1:]:
                cols = line.strip().split()
                if cols:
                    model_name = cols[0]
                    try:
                        _sp.run(["ollama", "stop", model_name], timeout=15, capture_output=True)
                        logger.info(f"Stopped Ollama model: {model_name}")
                    except Exception:
                        pass
    except Exception:
        pass

    _gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            vram = torch.cuda.memory_allocated(0) / (1024 ** 2)
            logger.info(f"CUDA cache cleared for ComfyUI. VRAM allocated: {vram:.0f} MB")
    except ImportError:
        pass

    _time.sleep(3)
