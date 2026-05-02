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
    comfyui_entries = [e for e in timeline_entries if e["action"] == "COMFYUI_PROMPT"]
    if not comfyui_entries:
        logger.info("No ComfyUI images to generate")
        return {"generated": [], "failed": []}

    global comfyui_process, comfyui_started_by_pipeline
    comfyui_started_by_pipeline = False

    if _wait_for_comfyui(timeout=3, quiet=True):
        logger.info("Using existing ComfyUI server")
    else:
        comfyui_process = _start_comfyui()
        comfyui_started_by_pipeline = comfyui_process is not None

    if not comfyui_process and not _wait_for_comfyui(timeout=3, quiet=True):
        logger.error("ComfyUI failed to start, skipping all image generation")
        return {"generated": [], "failed": [e["id"] for e in comfyui_entries]}

    if not _wait_for_comfyui(timeout=COMFYUI_STARTUP_TIMEOUT):
        _stop_comfyui()
        return {"generated": [], "failed": [e["id"] for e in comfyui_entries]}

    if not COMFYUI_WORKFLOW.exists():
        logger.error(f"ComfyUI workflow not found: {COMFYUI_WORKFLOW}")
        _stop_comfyui()
        return {"generated": [], "failed": [e["id"] for e in comfyui_entries]}

    workflow_template = json.loads(COMFYUI_WORKFLOW.read_text(encoding="utf-8"))

    generated = []
    failed = []

    for entry in comfyui_entries:
        entry_id = entry["id"]
        prompt = entry["data"]
        logger.info(f"Generating ComfyUI image for entry {entry_id}: {prompt[:60]}...")

        try:
            output_path = _generate_single_image(workflow_template, prompt, entry_id, asset_prefix)
            if output_path:
                generated.append({"id": entry_id, "path": str(output_path)})
            else:
                failed.append(entry_id)
        except Exception as e:
            logger.error(f"ComfyUI generation failed for entry {entry_id}: {e}")
            failed.append(entry_id)

    _stop_comfyui()
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
    graphics_entries = [e for e in timeline_entries if e["action"] in ("TEXT_CARD", "LOWER_THIRD")]
    if not graphics_entries:
        logger.info("No static graphics to generate")
        return {"generated": [], "failed": []}

    generated = []
    failed = []

    for entry in graphics_entries:
        try:
            if entry["action"] == "TEXT_CARD":
                output_path = _create_text_card(entry, asset_prefix)
            elif entry["action"] == "LOWER_THIRD":
                output_path = _create_lower_third(entry, asset_prefix)
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
        proc = subprocess.Popen(
            ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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


def _generate_single_image(workflow_template: dict, prompt: str, entry_id, asset_prefix: str = "") -> Path | None:
    workflow = copy.deepcopy(workflow_template)

    workflow["67"]["inputs"]["text"] = prompt
    workflow["69"]["inputs"]["seed"] = random.randint(0, 2**32 - 1)
    workflow["9"]["inputs"]["filename_prefix"] = f"{_asset_prefix(asset_prefix)}gen_{_entry_token(entry_id)}"

    try:
        resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow}, timeout=30)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
        logger.info(f"ComfyUI prompt submitted: {prompt_id}")
    except Exception as e:
        logger.error(f"Failed to submit prompt to ComfyUI: {e}")
        return None

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
                    if "9" in outputs:
                        images = outputs["9"].get("images", [])
                        if images:
                            img_info = images[0]
                            filename = img_info["filename"]
                            subfolder = img_info.get("subfolder", "")

                            comfyui_output = Path(workflow_template.get("_comfyui_output_dir", ""))
                            if not comfyui_output.exists():
                                for candidate in [
                                    COMFYUI_DIR / "output",
                                    Path.home() / "ComfyUI" / "output",
                                ]:
                                    if candidate.exists():
                                        comfyui_output = candidate
                                        break

                            src = comfyui_output / subfolder / filename
                            dst = PATHS["gen_images"] / f"{_asset_prefix(asset_prefix)}gen_{_entry_token(entry_id)}.png"

                            if src.exists():
                                img = Image.open(src)
                                img = img.resize((1920, 1080), Image.LANCZOS)
                                img.save(dst)
                                logger.info(f"Image saved: {dst}")
                                return dst

                    logger.error(f"ComfyUI completed but no image in output for entry {entry_id}")
                    return None

                if status.get("status_str") == "error":
                    logger.error(f"ComfyUI generation error for entry {entry_id}: {status}")
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

    bg_color = _hex_to_rgba(BRAND["brand"]["primary_color"], 220)
    draw.rounded_rectangle([(100, 300), (width - 100, height - 300)], radius=20, fill=bg_color)

    text = entry["data"]
    font_size = BRAND["fonts"]["text_card_size"]
    font = _load_font(BRAND["fonts"]["text_card"], font_size)

    text_color = _hex_to_rgb(BRAND["brand"]["bg_color"])
    _draw_wrapped_text(draw, text, font, text_color, width // 2, height // 2, width - 300)

    output_path = PATHS["graphics"] / f"{_asset_prefix(asset_prefix)}card_{_entry_token(entry['id'])}.png"
    img.save(output_path)
    logger.info(f"Text card saved: {output_path}")
    return output_path


def _create_lower_third(entry: dict, asset_prefix: str = "") -> Path | None:
    width, height = 1920, 100
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg_color = _hex_to_rgba(BRAND["brand"]["primary_color"], 200)
    draw.rounded_rectangle([(0, 0), (width, height)], radius=10, fill=bg_color)

    accent_color = _hex_to_rgba(BRAND["brand"]["accent_color"], 255)
    draw.rounded_rectangle([(0, 0), (8, height)], radius=4, fill=accent_color)

    text = entry["data"]
    font_size = BRAND["fonts"]["lower_third_size"]
    font = _load_font(BRAND["fonts"]["lower_third"], font_size)

    text_color = _hex_to_rgb(BRAND["brand"]["bg_color"])
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = 30
    y = (height - text_h) // 2
    draw.text((x, y), text, fill=text_color, font=font)

    output_path = PATHS["graphics"] / f"{_asset_prefix(asset_prefix)}lower_third_{_entry_token(entry['id'])}.png"
    img.save(output_path)
    logger.info(f"Lower third saved: {output_path}")
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


def _asset_prefix(asset_prefix: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(asset_prefix or "").strip())
    return f"{clean}_" if clean else ""


def _entry_token(entry_id) -> str:
    try:
        return f"{int(entry_id):03d}"
    except (TypeError, ValueError):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(entry_id))
