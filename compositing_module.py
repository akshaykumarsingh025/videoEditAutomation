import json
import logging
import os
import re
import subprocess
import warnings
import math
import time
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

warnings.filterwarnings("ignore", category=ResourceWarning)
os.environ["IMAGEMAGICK_BINARY"] = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"

try:
    import moviepy.config
    moviepy.config.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
    moviepy.config_defaults.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
except Exception:
    pass

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
from moviepy.video.fx.all import colorx, crop, loop as video_loop, resize
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from config import PATHS, BRAND, EXPORT_PROFILES, DRAFT_RESOLUTION

logger = logging.getLogger("pipeline.compositing")


def composite_video(
    input_video,
    timeline_data,
    srt_path,
    silence_cuts=None,
    draft=False,
    profile="landscape",
    asset_prefix="",
):
    input_video = Path(input_video)
    srt_path = Path(srt_path)
    profile_config = EXPORT_PROFILES.get(profile, EXPORT_PROFILES["landscape"])

    logger.info(f"Loading source video: {input_video}")
    video = VideoFileClip(str(input_video))

    video = _apply_silence_cuts(video, silence_cuts)

    if draft:
        target_w, target_h = DRAFT_RESOLUTION
        logger.info(f"Draft mode: rendering at {target_w}x{target_h}")
    else:
        target_w, target_h = profile_config["resolution"]

    video = _prepare_base_video(video, target_w, target_h, profile)

    # Smart Zoom: subtle face-tracking zoom on talking-head segments
    video = _apply_smart_zoom(video, timeline_data, draft)

    clips = [video]
    clips.extend(_create_timeline_clips(video, timeline_data, draft, asset_prefix))
    clips.extend(_create_watermark_clip(video, timeline_data, draft))
    clips.extend(_create_info_card_clip(video, draft))
    clips.extend(_create_subtitle_clips(video, srt_path, draft, asset_prefix))

    logger.info("Compositing all layers...")
    final = CompositeVideoClip(clips, size=(target_w, target_h))
    final = final.set_duration(video.duration)

    final = _add_music(final, input_video, draft)

    output_name = f"{input_video.stem}_{profile}"
    if draft:
        output_name += "_draft"
    output_path = PATHS["output"] / f"{output_name}.mp4"

    logger.info(f"Exporting to {output_path}...")
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast" if draft else BRAND["export"]["preset"],
        bitrate="1M" if draft else None,
        threads=4,
        logger=None,
    )

    video.close()
    final.close()

    _normalize_audio(output_path)

    logger.info(f"Export complete: {output_path}")
    return output_path


def generate_thumbnail(input_video, hero_moment=None):
    input_video = Path(input_video)
    video = VideoFileClip(str(input_video))

    if hero_moment and "time" in hero_moment:
        time_parts = hero_moment["time"].split(":")
        t = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
        t = min(t, video.duration - 1)
    else:
        t = video.duration * 0.25

    frame = video.get_frame(t)
    img = Image.fromarray(frame)
    img = img.resize((1280, 720), Image.LANCZOS)

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle([(0, 520), (1280, 720)], radius=0, fill=(26, 115, 232, 200))
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    brand_name = BRAND["brand"]["name"]
    try:
        font = ImageFont.truetype(str(PATHS["fonts"] / BRAND["fonts"]["text_card"]), 48)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), brand_name, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((1280 - tw) // 2, 560), brand_name, fill=(255, 255, 255), font=font)

    output_path = PATHS["output"] / f"{input_video.stem}_thumb.jpg"
    img.convert("RGB").save(output_path, "JPEG", quality=90)
    video.close()

    logger.info(f"Thumbnail saved: {output_path}")
    return output_path


def write_seo_files(seo_data, input_video):
    input_video = Path(input_video)
    results = {}

    seo_path = PATHS["output"] / f"{input_video.stem}_seo.json"
    seo_path.write_text(json.dumps(seo_data, indent=2, ensure_ascii=False), encoding="utf-8")
    results["seo_path"] = seo_path

    if "chapters" in seo_data and seo_data["chapters"]:
        chapters_path = PATHS["output"] / f"{input_video.stem}_chapters.txt"
        lines = []
        for ch in seo_data["chapters"]:
            lines.append(f"{ch['time']} - {ch['title']}")
        chapters_path.write_text("\n".join(lines), encoding="utf-8")
        results["chapters_path"] = chapters_path

    logger.info(f"SEO files written: {list(results.keys())}")
    return results


def _apply_silence_cuts(video, silence_cuts):
    if not silence_cuts:
        return video

    logger.info(f"Applying {len(silence_cuts)} silence cuts...")
    clips = []
    prev_end = 0.0

    for cut in silence_cuts:
        start_sec = cut.get("start_sec", _time_to_sec(cut["start"]))
        end_sec = cut.get("end_sec", _time_to_sec(cut["end"]))

        if start_sec > prev_end:
            clips.append(video.subclip(prev_end, start_sec))
        prev_end = end_sec

    if prev_end < video.duration:
        clips.append(video.subclip(prev_end, video.duration))

    if clips:
        return concatenate_videoclips(clips)
    return video


def _create_subtitle_clips(video, srt_path, draft, asset_prefix=""):
    if not srt_path or not Path(srt_path).exists():
        logger.warning(f"SRT file not found: {srt_path}")
        return []

    words_path = PATHS["temp"] / f"{Path(srt_path).stem.replace('_trimmed', '')}_words.json"
    words_data = None
    if words_path.exists():
        try:
            words_data = json.loads(words_path.read_text(encoding="utf-8"))
            logger.info(f"Loaded word-level data for karaoke subtitles: {len(words_data)} words")
        except Exception as e:
            logger.warning(f"Failed to load words data: {e}")

    srt_content = Path(srt_path).read_text(encoding="utf-8")
    blocks = srt_content.strip().split("\n\n")

    scale = video.w / 1920
    if draft:
        scale = DRAFT_RESOLUTION[0] / 1920

    font_size = max(18, int(BRAND["fonts"]["subtitle_size"] * scale))

    font_path = None
    for candidate in [
        PATHS["fonts"] / BRAND["fonts"].get("subtitle", "Poppins-Medium.ttf"),
        PATHS["fonts"] / "Poppins-Medium.ttf",
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/NotoSans-Regular.ttf"),
        Path("C:/Windows/Fonts/mangal.ttf"),
        Path("C:/Windows/Fonts/devanagari.ttf"),
    ]:
        if candidate.exists():
            font_path = candidate
            break

    sub_style = BRAND.get("subtitle_style", {})
    text_color = sub_style.get("text_color", "#FFFFFF")
    highlight_color = sub_style.get("highlight_color", "#E8734A")
    y_pos_ratio = sub_style.get("y_position_ratio", 0.84)

    if words_data:
        clips = _create_karaoke_subtitles_pillow(video, words_data, blocks, font_size, font_path, text_color, highlight_color, y_pos_ratio)
    else:
        clips = _create_simple_subtitles_pillow(video, blocks, font_size, font_path, text_color, y_pos_ratio)

    logger.info(f"Created {len(clips)} subtitle clips")
    return clips


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _load_font_safe(font_path, size):
    """Load a TrueType font with fallback to Pillow's default."""
    try:
        return ImageFont.truetype(str(font_path), size)
    except (IOError, OSError):
        logger.warning(f"Could not load font {font_path}, trying system fallback")
        for fallback in [
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]:
            if fallback.exists():
                try:
                    return ImageFont.truetype(str(fallback), size)
                except (IOError, OSError):
                    continue
        return ImageFont.load_default()


def _render_subtitle_image(text, video_w, video_h, font_size, font_path, text_color_hex, y_pos_ratio):
    """Render subtitle text as a transparent PNG using Pillow — no ImageMagick needed."""
    font = _load_font_safe(font_path, font_size) if font_path else _load_font_safe(Path("arial.ttf"), font_size)
    max_text_w = int(video_w * 0.88)

    canvas = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Word-wrap
    words = text.split()
    wrapped_lines = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_text_w and current_line:
            wrapped_lines.append(current_line)
            current_line = word
        else:
            current_line = test
    if current_line:
        wrapped_lines.append(current_line)

    if not wrapped_lines:
        return None

    line_height = draw.textbbox((0, 0), "Ay", font=font)[3] + 6

    actual_max_w = 0
    for line in wrapped_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        actual_max_w = max(actual_max_w, bbox[2] - bbox[0])

    pad_x = 28
    pad_y = 14
    block_h = len(wrapped_lines) * line_height + pad_y * 2
    block_w = actual_max_w + pad_x * 2

    y_start = int(video_h * y_pos_ratio)
    x_start = (video_w - block_w) // 2

    draw.rounded_rectangle(
        [(x_start, y_start - pad_y), (x_start + block_w, y_start - pad_y + block_h)],
        radius=10,
        fill=(0, 0, 0, 170),
    )

    text_rgb = _hex_to_rgb(text_color_hex)
    for i, line in enumerate(wrapped_lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        lx = (video_w - tw) // 2
        ly = y_start + i * line_height
        draw.text((lx + 2, ly + 2), line, fill=(0, 0, 0, 180), font=font)
        draw.text((lx, ly), line, fill=text_rgb + (255,), font=font)

    return canvas


def _create_karaoke_subtitles_pillow(video, words_data, srt_blocks, font_size, font_path, text_color, highlight_color, y_pos_ratio):
    clips = []
    words_per_segment = _map_words_to_srt(words_data, srt_blocks)
    sub_cache_dir = PATHS["temp"] / "subtitle_frames"
    sub_cache_dir.mkdir(parents=True, exist_ok=True)

    clip_idx = 0
    for seg_idx, block in enumerate(srt_blocks):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            parts = lines[1].split(" --> ")
            start = _srt_to_sec(parts[0].strip())
            end = _srt_to_sec(parts[1].strip())
        except (IndexError, ValueError):
            continue

        duration = end - start
        if duration <= 0:
            continue

        text = " ".join(lines[2:]).strip()
        if not text:
            continue

        seg_words = words_per_segment.get(seg_idx, [])

        if seg_words:
            chunk_size = 8
            for i in range(0, len(seg_words), chunk_size):
                chunk = seg_words[i:i + chunk_size]
                chunk_start = chunk[0]["start"]
                chunk_end = chunk[-1]["end"]
                chunk_text = " ".join(w["word"].strip() for w in chunk)
                if not chunk_text:
                    continue
                chunk_duration = max(chunk_end - chunk_start, 0.3)

                try:
                    img = _render_subtitle_image(chunk_text, video.w, video.h, font_size, font_path, text_color, y_pos_ratio)
                    if img is None:
                        continue
                    img_path = sub_cache_dir / f"sub_{clip_idx:05d}.png"
                    img.save(str(img_path))
                    clip = ImageClip(str(img_path)).set_start(chunk_start).set_duration(chunk_duration)
                    clips.append(clip)
                    clip_idx += 1
                except Exception as e:
                    logger.warning(f"Failed to create karaoke subtitle: {e}")
        else:
            try:
                img = _render_subtitle_image(text, video.w, video.h, font_size, font_path, text_color, y_pos_ratio)
                if img is None:
                    continue
                img_path = sub_cache_dir / f"sub_{clip_idx:05d}.png"
                img.save(str(img_path))
                clip = ImageClip(str(img_path)).set_start(start).set_duration(duration)
                clips.append(clip)
                clip_idx += 1
            except Exception as e:
                logger.warning(f"Failed to create subtitle: {e}")

    return clips


def _map_words_to_srt(words_data, srt_blocks):
    words_per_segment = {}
    word_idx = 0

    for seg_idx, block in enumerate(srt_blocks):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            parts = lines[1].split(" --> ")
            seg_start = _srt_to_sec(parts[0].strip())
            seg_end = _srt_to_sec(parts[1].strip())
        except (IndexError, ValueError):
            continue

        seg_words = []
        while word_idx < len(words_data):
            w = words_data[word_idx]
            w_start = w["start"]
            w_end = w["end"]
            w_mid = (w_start + w_end) / 2

            if w_mid < seg_start - 0.5:
                word_idx += 1
                continue
            elif w_mid > seg_end + 0.5:
                break
            else:
                seg_words.append(w)
                word_idx += 1

        if seg_words:
            words_per_segment[seg_idx] = seg_words

    return words_per_segment


def _create_simple_subtitles_pillow(video, srt_blocks, font_size, font_path, text_color, y_pos_ratio):
    clips = []
    sub_cache_dir = PATHS["temp"] / "subtitle_frames"
    sub_cache_dir.mkdir(parents=True, exist_ok=True)
    clip_idx = 0

    for block in srt_blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            parts = lines[1].split(" --> ")
            start = _srt_to_sec(parts[0].strip())
            end = _srt_to_sec(parts[1].strip())
        except (IndexError, ValueError):
            continue

        duration = end - start
        if duration <= 0:
            continue

        text = " ".join(lines[2:]).strip()
        if not text:
            continue

        try:
            img = _render_subtitle_image(text, video.w, video.h, font_size, font_path, text_color, y_pos_ratio)
            if img is None:
                continue
            img_path = sub_cache_dir / f"sub_simple_{clip_idx:05d}.png"
            img.save(str(img_path))
            clip = ImageClip(str(img_path)).set_start(start).set_duration(duration)
            clips.append(clip)
            clip_idx += 1
        except Exception as e:
            logger.warning(f"Failed to create subtitle clip: {e}")

    return clips


def _create_timeline_clips(video, timeline_data, draft, asset_prefix=""):
    if not timeline_data or "timeline" not in timeline_data:
        return []

    clips = []

    for entry in timeline_data["timeline"]:
        if entry["action"] == "WATERMARK":
            continue

        start = _time_to_sec(entry["time"])
        duration = entry.get("duration", 3)
        position = entry.get("position", "center")
        fade = entry.get("fade")
        fx = entry.get("fx", "")

        try:
            if entry["action"] == "BROLL_IMAGE":
                img_path = _find_asset(PATHS["gen_images"], asset_prefix, "gen", entry["id"], [".png"])
                if not img_path.exists():
                    if not draft:
                        logger.warning(f"B-roll image not found: {img_path}")
                    continue

                clip = ImageClip(str(img_path)).set_duration(duration).set_start(start)
                clip = clip.resize((video.w, video.h))
                clip = clip.set_position(("center", "center"))
                clip = _apply_fade(clip, fade or "in-out", duration)

                if fx.startswith("ken_burns"):
                    clip = _apply_ken_burns(clip, fx, duration)

            elif entry["action"] == "COMFYUI_PROMPT":
                img_path = _find_asset(PATHS["gen_images"], asset_prefix, "gen", entry["id"], [".png"])
                if not img_path.exists():
                    if not draft:
                        logger.warning(f"ComfyUI image not found: {img_path}")
                    continue

                clip = ImageClip(str(img_path)).set_duration(duration).set_start(start)
                clip = clip.resize(height=int(video.h * 0.6))
                clip = _apply_position(clip, video, position)
                clip = _apply_fade(clip, fade, duration)

            elif entry["action"] == "WEB_GIF":
                gif_path = _find_asset(PATHS["web"], asset_prefix, "gif", entry["id"], [".mp4", ".gif"])
                if not gif_path.exists():
                    if not draft:
                        logger.warning(f"GIF not found for entry {entry['id']}")
                    continue

                clip = _create_web_media_clip(gif_path, duration).set_start(start)
                clip = clip.resize(height=video.h // 4)
                clip = _apply_position(clip, video, position)

            elif entry["action"] == "TEXT_CARD":
                img_path = _find_asset(PATHS["graphics"], asset_prefix, "card", entry["id"], [".png"])
                if not img_path.exists():
                    if not draft:
                        logger.warning(f"Text card not found: {img_path}")
                    continue

                clip = ImageClip(str(img_path)).set_duration(duration).set_start(start)
                clip = clip.resize((video.w, video.h))
                clip = _animate_text_card(clip, video, duration)

            elif entry["action"] == "LOWER_THIRD":
                img_path = _find_asset(PATHS["graphics"], asset_prefix, "lower_third", entry["id"], [".png"])
                if not img_path.exists():
                    if not draft:
                        logger.warning(f"Lower third not found: {img_path}")
                    continue

                clip = ImageClip(str(img_path)).set_duration(duration).set_start(start)
                clip = clip.resize(width=int(video.w * 0.5))
                clip = _apply_position(clip, video, "bottom-left")

            else:
                continue

            clips.append(clip)

        except Exception as e:
            logger.error(f"Failed to create clip for entry {entry['id']}: {e}")

    logger.info(f"Created {len(clips)} overlay clips")
    return clips


def _create_watermark_clip(video, timeline_data, draft):
    watermark_cfg = BRAND.get("watermark", {})
    watermark_entries = []
    if timeline_data and "timeline" in timeline_data:
        watermark_entries = [e for e in timeline_data["timeline"] if e.get("action") == "WATERMARK"]

    configured_logo = watermark_cfg.get("logo")
    logo_name = watermark_entries[0].get("data") if watermark_entries else None
    logo_name = logo_name or configured_logo or "brand_logo.png"
    logo_path = Path(logo_name)
    if not logo_path.is_absolute():
        logo_path = PATHS["templates"] / logo_path.name
    if not logo_path.exists():
        logger.warning(f"Watermark logo not found: {logo_path}")
        return _create_text_watermark_clip(video)

    try:
        opacity = watermark_cfg.get("opacity", 0.1)
        clip = ImageClip(str(logo_path)).set_duration(video.duration).set_start(0)
        clip = clip.resize(height=video.h // 8)
        clip = clip.set_opacity(opacity)
        margin = watermark_cfg.get("margin", 20)
        pos = watermark_cfg.get("position", "bottom-right")
        clip = _apply_position(clip, video, pos, margin)
        return [clip]
    except Exception as e:
        logger.error(f"Failed to create watermark clip: {e}")
        return []


# ─── Smart Zoom (Face-Tracking Subtle Zoom) ───────────────────────────

def _apply_smart_zoom(video, timeline_data, draft):
    """Apply subtle centered zoom during talking-head segments.
    
    The zoom oscillates slowly (zoom-in -> zoom-out -> repeat) centered on
    the frame so the video feels alive without visible panning artifacts.
    """
    zoom_cfg = BRAND.get("smart_zoom", {})
    if not zoom_cfg.get("enabled", False):
        return video

    max_zoom = zoom_cfg.get("max_zoom", 1.04)
    if max_zoom <= 1.0:
        return video

    cycle_sec = max(float(zoom_cfg.get("cycle_seconds", 18)), 0.1)

    broll_intervals = []
    if timeline_data and "timeline" in timeline_data:
        for entry in timeline_data["timeline"]:
            if entry.get("action") in ("BROLL_IMAGE", "COMFYUI_PROMPT"):
                t = _time_to_sec(entry["time"])
                d = entry.get("duration", 5)
                broll_intervals.append((t, t + d))

    def zoom_filter(get_frame, t):
        for bstart, bend in broll_intervals:
            if bstart <= t <= bend:
                return get_frame(t)

        phase = (t % cycle_sec) / cycle_sec
        zoom = 1.0 + (max_zoom - 1.0) * (0.5 - 0.5 * math.cos(phase * 2 * math.pi))

        frame = get_frame(t)
        try:
            import numpy as np
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype("uint8")
            frame = np.ascontiguousarray(frame)
        except Exception:
            pass
        fh, fw = frame.shape[:2]

        crop_w = max(1, int(round(fw / zoom)))
        crop_h = max(1, int(round(fh / zoom)))

        x1 = (fw - crop_w) // 2
        y1 = (fh - crop_h) // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame

        try:
            import cv2 as _cv2
            resized = _cv2.resize(cropped, (fw, fh), interpolation=_cv2.INTER_LINEAR)
        except Exception:
            from PIL import Image as _PILImage
            import numpy as np
            pil_img = _PILImage.fromarray(cropped)
            pil_img = pil_img.resize((fw, fh), _PILImage.LANCZOS)
            resized = np.array(pil_img)

        return resized

    try:
        zoomed = video.fl(zoom_filter)
        zoomed = zoomed.set_duration(video.duration)
        if video.audio:
            zoomed = zoomed.set_audio(video.audio)
        logger.info(f"Smart zoom applied: max {max_zoom:.0%}, cycle {cycle_sec}s, centered")
        return zoomed
    except Exception as e:
        logger.warning(f"Smart zoom failed, using original video: {e}")
        return video


def _detect_face_center(video, num_samples=5):
    """Sample frames from the video and detect the average face center position.
    
    Returns (cx, cy) as fractions of video dimensions (0..1), or (None, None)
    if no face is detected.
    """
    if not HAS_CV2:
        logger.warning("Smart zoom: OpenCV not available, skipping face detection")
        return None, None

    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.warning("Smart zoom: failed to load face cascade classifier")
            return None, None
    except Exception as e:
        logger.warning(f"Smart zoom: cascade init failed: {e}")
        return None, None

    # Sample frames at evenly spaced points in the first 60% of the video
    # (speakers are usually visible in the first half)
    sample_times = []
    if video.duration <= 0:
        return None, None
    sample_start = min(2.0, max(0.0, video.duration * 0.1))
    sample_end = min(video.duration * 0.6, max(sample_start, video.duration - 0.1))
    for i in range(num_samples):
        t = sample_start + (sample_end - sample_start) * i / max(num_samples - 1, 1)
        t = max(0.0, min(t, max(0.0, video.duration - 0.01)))
        sample_times.append(t)

    face_centers = []
    for t in sample_times:
        try:
            frame = video.get_frame(t)
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Downscale for speed
            scale = 480.0 / max(gray.shape[:2])
            if scale < 1.0:
                small = cv2.resize(gray, None, fx=scale, fy=scale)
            else:
                small = gray
                scale = 1.0

            faces = face_cascade.detectMultiScale(
                small,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            if len(faces) > 0:
                # Pick the largest face
                areas = [w_f * h_f for (_, _, w_f, h_f) in faces]
                best_idx = areas.index(max(areas))
                x, y, w_f, h_f = faces[best_idx]

                # Convert back to original coordinates, normalized to 0..1
                cx = (x + w_f / 2) / scale / frame.shape[1]
                cy = (y + h_f / 2) / scale / frame.shape[0]
                face_centers.append((cx, cy))
        except Exception:
            continue

    if not face_centers:
        return None, None

    # Average all detected face centers
    avg_cx = sum(c[0] for c in face_centers) / len(face_centers)
    avg_cy = sum(c[1] for c in face_centers) / len(face_centers)

    return avg_cx, avg_cy


# ─── Animated Text Cards ──────────────────────────────────────────────

def _animate_text_card(clip, video, duration):
    """Apply entrance/exit animation to a text card clip.
    
    Supported animations (configured in brand_profile.yaml → text_card_style.animation):
      - "slide_up": slides up from below with fade
      - "scale_bounce": scales from 0.8 to 1.0 with an elastic overshoot
      - "fade_scale": fades in while scaling from 0.95 to 1.0 (subtle)
      - "none": just applies standard crossfade (legacy behavior)
    """
    style = BRAND.get("text_card_style", {})
    anim_type = style.get("animation", "slide_up")
    anim_dur = style.get("animation_duration", 0.35)

    if anim_type == "none" or duration < anim_dur * 3:
        # Too short for animation or disabled — fallback to simple fade
        return _apply_fade(clip, "in-out", duration)

    # Clamp animation duration to something reasonable
    anim_dur = min(anim_dur, duration / 4)

    try:
        if anim_type == "slide_up":
            return _anim_slide_up(clip, video, duration, anim_dur)
        elif anim_type == "scale_bounce":
            return _anim_scale_bounce(clip, video, duration, anim_dur)
        elif anim_type == "fade_scale":
            return _anim_fade_scale(clip, video, duration, anim_dur)
        else:
            return _apply_fade(clip, "in-out", duration)
    except Exception as e:
        logger.warning(f"Text card animation '{anim_type}' failed: {e}, using fade fallback")
        return _apply_fade(clip, "in-out", duration)


def _anim_slide_up(clip, video, duration, anim_dur):
    """Slide up from below + fade in, then slide down + fade out."""
    slide_distance = video.h * 0.08  # 8% of video height

    def position_func(t):
        if t < anim_dur:
            # Entrance: slide up from below
            progress = _ease_out_cubic(t / anim_dur)
            y_offset = slide_distance * (1.0 - progress)
            return ("center", y_offset)
        elif t > duration - anim_dur:
            # Exit: slide down
            progress = _ease_in_cubic((t - (duration - anim_dur)) / anim_dur)
            y_offset = slide_distance * progress
            return ("center", y_offset)
        else:
            return ("center", 0)

    def opacity_func(t):
        if t < anim_dur:
            return _ease_out_cubic(t / anim_dur)
        elif t > duration - anim_dur:
            return 1.0 - _ease_in_cubic((t - (duration - anim_dur)) / anim_dur)
        return 1.0

    clip = clip.set_position(position_func)
    return _apply_opacity_curve(clip, opacity_func)


def _anim_scale_bounce(clip, video, duration, anim_dur):
    """Scale from 0.85→1.02→1.0 (bounce) on entrance, scale down on exit, with opacity."""
    bounce_dur = anim_dur * 1.4  # bounce takes a bit longer

    def scale_func(t):
        if t < bounce_dur:
            progress = t / bounce_dur
            # Overshoot curve: goes from 0.85 to 1.02 then settles at 1.0
            if progress < 0.7:
                s = 0.85 + 0.17 * _ease_out_cubic(progress / 0.7)
            else:
                s = 1.02 - 0.02 * _ease_in_out_cubic((progress - 0.7) / 0.3)
            return s
        elif t > duration - anim_dur:
            progress = (t - (duration - anim_dur)) / anim_dur
            s = 1.0 - 0.15 * _ease_in_cubic(progress)
            return s
        return 1.0

    def opacity_func(t):
        if t < anim_dur:
            return min(1.0, _ease_out_cubic(t / anim_dur))
        elif t > duration - anim_dur:
            return 1.0 - _ease_in_cubic((t - (duration - anim_dur)) / anim_dur)
        return 1.0

    clip = _apply_opacity_curve(clip, opacity_func)
    clip = clip.resize(scale_func)
    clip = clip.set_position(("center", "center"))
    return clip


def _anim_fade_scale(clip, video, duration, anim_dur):
    """Subtle scale from 0.96→1.0 with fade (most minimal animation)."""

    def scale_func(t):
        if t < anim_dur:
            progress = _ease_out_cubic(t / anim_dur)
            return 0.96 + 0.04 * progress
        elif t > duration - anim_dur:
            progress = _ease_in_cubic((t - (duration - anim_dur)) / anim_dur)
            return 1.0 - 0.04 * progress
        return 1.0

    def opacity_func(t):
        if t < anim_dur:
            return _ease_out_cubic(t / anim_dur)
        elif t > duration - anim_dur:
            return 1.0 - _ease_in_cubic((t - (duration - anim_dur)) / anim_dur)
        return 1.0

    clip = _apply_opacity_curve(clip, opacity_func)
    clip = clip.resize(scale_func)
    clip = clip.set_position(("center", "center"))
    return clip


# ─── Easing Functions ─────────────────────────────────────────────────

def _ease_out_cubic(t):
    """Decelerating: fast start, slow end."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def _ease_in_cubic(t):
    """Accelerating: slow start, fast end."""
    t = max(0.0, min(1.0, t))
    return t ** 3


def _ease_in_out_cubic(t):
    """Smooth S-curve."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t ** 3
    return 1.0 - (-2 * t + 2) ** 3 / 2


def _apply_opacity_curve(clip, opacity_func):
    try:
        base_mask = clip.mask
        if base_mask is None:
            base_mask = ColorClip(clip.size, color=1.0, ismask=True).set_duration(clip.duration)

        def mask_filter(get_frame, t):
            opacity = max(0.0, min(1.0, float(opacity_func(t))))
            return get_frame(t) * opacity

        return clip.set_mask(base_mask.fl(mask_filter))
    except Exception as e:
        logger.warning(f"Animated opacity failed, keeping clip fully visible: {e}")
        return clip


def _apply_ken_burns(clip, fx_type, duration):
    broll_fx = BRAND.get("broll_fx", {})
    zoom_start = broll_fx.get("zoom_start", 1.0)
    zoom_end = broll_fx.get("zoom_end", 1.15)

    if "out" in fx_type:
        zoom_start, zoom_end = zoom_end, zoom_start

    def zoom_func(t):
        progress = t / max(duration, 0.01)
        return zoom_start + (zoom_end - zoom_start) * progress

    clip = clip.resize(zoom_func)
    return clip


def _create_info_card_clip(video, draft):
    info_cfg = BRAND.get("info_card", {})
    if not info_cfg.get("enabled", False):
        return []

    interval = info_cfg.get("interval_seconds", 90)
    show_duration = info_cfg.get("show_duration", 8)
    fade_dur = 0.5

    try:
        card_img = _generate_info_card_image(video.w, video.h, draft)
        clips = []

        # Show at start (after 2 seconds)
        t = 2.0
        while t < video.duration - show_duration:
            clip = ImageClip(str(card_img)).set_duration(show_duration).set_start(t)
            clip = clip.crossfadein(fade_dur).crossfadeout(fade_dur)
            clips.append(clip)
            t += interval

        logger.info(f"Created {len(clips)} periodic info card appearances (every {interval}s, {show_duration}s each)")
        return clips
    except Exception as e:
        logger.error(f"Failed to create info card clip: {e}")
        return []


def _generate_info_card_image(video_w, video_h, draft):
    info_cfg = BRAND.get("info_card", {})
    card_w = info_cfg.get("width", 520)
    padding = info_cfg.get("padding", 28)
    corner_r = info_cfg.get("corner_radius", 16)
    bg_hex = info_cfg.get("bg_color", "#0A1628")
    accent_hex = info_cfg.get("accent_color", "#E8734A")
    text_hex = info_cfg.get("text_color", "#FFFFFF")
    sub_hex = info_cfg.get("subtext_color", "#D1D5DB")
    border_hex = info_cfg.get("border_color", "#E8734A")
    border_width = info_cfg.get("border_width", 2)
    accent_bar_w = info_cfg.get("accent_bar_width", 8)
    lines = info_cfg.get("lines", [])
    position = info_cfg.get("position", "top-left")

    name_font_base = info_cfg.get("name_font_size", 42)
    title_font_base = info_cfg.get("title_font_size", 26)
    line_font_base = info_cfg.get("line_font_size", 24)
    phone_font_base = info_cfg.get("phone_font_size", 28)

    scale_factor = video_w / 1920
    if draft:
        scale_factor = DRAFT_RESOLUTION[0] / 1920

    card_w = int(card_w * scale_factor)
    padding = int(padding * scale_factor)

    name_font_size = max(16, int(name_font_base * scale_factor))
    title_font_size = max(12, int(title_font_base * scale_factor))
    line_font_size = max(11, int(line_font_base * scale_factor))
    phone_font_size = max(13, int(phone_font_base * scale_factor))

    bold_font_path = PATHS["fonts"] / BRAND["fonts"].get("text_card", "Poppins-Bold.ttf")
    semi_font_path = PATHS["fonts"] / BRAND["fonts"].get("lower_third", "Poppins-SemiBold.ttf")

    font_name = _load_font_safe(bold_font_path, name_font_size)
    font_title = _load_font_safe(semi_font_path, title_font_size)
    font_line = _load_font_safe(semi_font_path, line_font_size)
    font_phone = _load_font_safe(bold_font_path, phone_font_size)

    line_fonts = [
        font_name,
        font_title,
        font_line,
        font_line,
        font_phone,
        font_line,
    ]
    while len(line_fonts) < len(lines):
        line_fonts.append(font_line)

    temp_img = Image.new("RGBA", (max(card_w, 200), 200), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    max_text_w = 0
    total_text_h = padding
    line_heights = []

    for i, line in enumerate(lines):
        f = line_fonts[i]
        bbox = temp_draw.textbbox((0, 0), line, font=f)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        max_text_w = max(max_text_w, tw)
        gap = 14 if i == 0 else 8
        line_heights.append(th + gap)
        total_text_h += th + gap

    total_text_h += padding
    card_h = total_text_h + padding
    card_w = max(card_w, max_text_w + padding * 2 + 20)

    canvas = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    if position == "top-left":
        cx, cy = 24, 24
    elif position == "top-right":
        cx, cy = video_w - card_w - 24, 24
    elif position == "bottom-left":
        cx, cy = 24, video_h - card_h - 24
    else:
        cx, cy = 24, 24

    bg_rgb = _hex_to_rgb(bg_hex)
    draw.rounded_rectangle(
        [(cx, cy), (cx + card_w, cy + card_h)],
        radius=corner_r,
        fill=bg_rgb + (225,),
    )

    border_rgb = _hex_to_rgb(border_hex)
    draw.rounded_rectangle(
        [(cx, cy), (cx + card_w, cy + card_h)],
        radius=corner_r,
        fill=None,
        outline=border_rgb + (180,),
        width=border_width,
    )

    accent_rgb = _hex_to_rgb(accent_hex)
    draw.rounded_rectangle(
        [(cx, cy), (cx + accent_bar_w, cy + card_h)],
        radius=4,
        fill=accent_rgb + (255,),
    )

    text_rgb = _hex_to_rgb(text_hex)
    sub_rgb = _hex_to_rgb(sub_hex)
    accent_text_rgb = _hex_to_rgb(accent_hex)

    y_offset = cy + padding
    for i, line in enumerate(lines):
        f = line_fonts[i]
        if i == 0:
            color = text_rgb
        elif i == len(lines) - 2:
            color = accent_text_rgb
        else:
            color = sub_rgb
        draw.text((cx + padding + accent_bar_w + 8, y_offset), line, fill=color, font=f)
        y_offset += line_heights[i]

    output_path = PATHS["temp"] / "info_card_overlay.png"
    canvas.save(output_path)
    return output_path


def _load_font_safe(font_path, size):
    if font_path.exists():
        try:
            return ImageFont.truetype(str(font_path), size)
        except Exception:
            pass
    for sys_font in ["arial.ttf", "Arial.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/calibri.ttf"]:
        try:
            return ImageFont.truetype(sys_font, size)
        except OSError:
            continue
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _apply_position(clip, video, position, margin=0):
    if margin == 0:
        margin = 20

    pos_map = {
        "center": ("center", "center"),
        "bottom-left": (margin, video.h - clip.h - margin),
        "bottom-right": (video.w - clip.w - margin, video.h - clip.h - margin),
        "top-left": (margin, margin),
        "top-right": (video.w - clip.w - margin, margin),
        "top-center": ("center", margin),
        "bottom-center": ("center", video.h - clip.h - margin),
    }

    pos = pos_map.get(position, ("center", "center"))
    return clip.set_position(pos)


def _apply_fade(clip, fade, duration):
    if not fade:
        return clip
    fade_dur = min(0.5, duration / 4)
    if "in" in fade:
        clip = clip.crossfadein(fade_dur)
    if "out" in fade:
        clip = clip.crossfadeout(fade_dur)
    return clip


def _add_music(final_clip, input_video, draft):
    configured_track = BRAND.get("music", {}).get("default_track", "background.mp3")
    music_path = PATHS["music"] / Path(configured_track).name
    if not music_path.exists():
        logger.warning("Background music not found, skipping")
        return final_clip

    try:
        music = AudioFileClip(str(music_path))
        if music.duration < final_clip.duration:
            from moviepy.audio.fx.all import audio_loop
            music = audio_loop(music, duration=final_clip.duration)
        else:
            music = music.subclip(0, final_clip.duration)

        vol_normal = 10 ** (BRAND["music"]["volume_normal"] / 20)
        music = music.volumex(vol_normal)

        fade_out_sec = BRAND["music"].get("fade_out_seconds", 3)
        if fade_out_sec > 0:
            music = music.audio_fadeout(fade_out_sec)

        original_audio = final_clip.audio
        if original_audio:
            from moviepy.editor import CompositeAudioClip
            final_clip = final_clip.set_audio(CompositeAudioClip([original_audio, music]))
        else:
            final_clip = final_clip.set_audio(music)

    except Exception as e:
        logger.error(f"Failed to add background music: {e}")

    return final_clip


def _normalize_audio(output_path):
    logger.info("Normalizing audio loudness to -14 LUFS...")
    temp_path = output_path.parent / f"{output_path.stem}_normalized{output_path.suffix}"

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(output_path),
            "-af", "loudnorm=I=-14:TP=-1:LRA=11",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            str(temp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0 and temp_path.exists():
            replaced = False
            for attempt in range(5):
                try:
                    temp_path.replace(output_path)
                    replaced = True
                    break
                except PermissionError:
                    time.sleep(0.5 * (attempt + 1))

            if replaced:
                logger.info("Audio normalization complete")
            else:
                logger.warning("Audio normalization completed but output file was locked; keeping unnormalized export")
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except PermissionError:
                        logger.warning(f"Could not remove locked normalized temp file: {temp_path}")
        else:
            logger.warning(f"Audio normalization failed: {result.stderr[:200]}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except PermissionError:
                    logger.warning(f"Could not remove locked normalized temp file: {temp_path}")
    except FileNotFoundError:
        logger.warning("ffmpeg not found, skipping audio normalization")
    except Exception as e:
        logger.warning(f"Audio normalization error: {e}")

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Final output: {output_path} ({size_mb:.1f} MB)")
    else:
        logger.error(f"Output file missing after normalization: {output_path}")


def _time_to_sec(time_str):
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(time_str)


def _hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _srt_to_sec(time_str):
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def _prepare_base_video(video, target_w, target_h, profile):
    source_ratio = video.w / video.h
    target_ratio = target_w / target_h
    if profile in {"shorts", "square"} and abs(source_ratio - target_ratio) > 0.08:
        bg = _resize_crop(video, target_w, target_h).fx(colorx, 0.5)
        fg = _resize_contain(video, int(target_w * 0.94), int(target_h * 0.72))
        return CompositeVideoClip([bg, fg.set_position("center")], size=(target_w, target_h)).set_duration(video.duration)
    return _resize_crop(video, target_w, target_h)


def _resize_crop(video, target_w, target_h):
    source_ratio = video.w / video.h
    target_ratio = target_w / target_h
    resized = video.resize(height=target_h) if source_ratio > target_ratio else video.resize(width=target_w)
    return crop(
        resized,
        x_center=resized.w / 2,
        y_center=resized.h / 2,
        width=target_w,
        height=target_h,
    )


def _resize_contain(video, max_w, max_h):
    source_ratio = video.w / video.h
    box_ratio = max_w / max_h
    if source_ratio > box_ratio:
        return video.resize(width=max_w)
    return video.resize(height=max_h)


def _create_web_media_clip(path: Path, duration: float):
    clip = VideoFileClip(str(path), audio=False)
    if clip.duration < duration:
        clip = video_loop(clip, duration=duration)
    else:
        clip = clip.subclip(0, duration)
    return clip.set_duration(duration)


def _create_text_watermark_clip(video):
    try:
        clip = TextClip(
            BRAND["brand"]["name"],
            fontsize=max(18, video.h // 32),
            font="Arial",
            color="white",
            method="label",
        ).set_duration(video.duration).set_start(0)
        watermark_cfg = BRAND.get("watermark", {})
        clip = clip.set_opacity(watermark_cfg.get("opacity", 0.1))
        margin = watermark_cfg.get("margin", 20)
        position = watermark_cfg.get("position", "bottom-right")
        return [_apply_position(clip, video, position, margin)]
    except Exception as e:
        logger.warning(f"Text watermark fallback failed: {e}")
        return []


def _find_asset(folder: Path, asset_prefix: str, kind: str, entry_id, suffixes: list[str]) -> Path:
    token = _entry_token(entry_id)
    prefixes = [_asset_prefix(asset_prefix), ""]
    for prefix in prefixes:
        for suffix in suffixes:
            candidate = folder / f"{prefix}{kind}_{token}{suffix}"
            if candidate.exists():
                return candidate
    return folder / f"{prefixes[0]}{kind}_{token}{suffixes[0]}"


def _asset_prefix(asset_prefix: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(asset_prefix or "").strip())
    return f"{clean}_" if clean else ""


def _entry_token(entry_id) -> str:
    try:
        return f"{int(entry_id):03d}"
    except (TypeError, ValueError):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(entry_id))
