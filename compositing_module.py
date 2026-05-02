import json
import logging
import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("IMAGEMAGICK_BINARY", r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe")

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
from moviepy.video.fx.all import colorx, crop, loop as video_loop
from PIL import Image

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

    clips = [video]
    clips.extend(_create_subtitle_clips(video, srt_path, draft))
    clips.extend(_create_timeline_clips(video, timeline_data, draft, asset_prefix))
    clips.extend(_create_watermark_clip(video, timeline_data, draft))

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


def _create_subtitle_clips(video, srt_path, draft):
    if not srt_path.exists():
        logger.warning(f"SRT file not found: {srt_path}")
        return []

    srt_content = srt_path.read_text(encoding="utf-8")
    blocks = srt_content.strip().split("\n\n")

    clips = []
    font_path = PATHS["fonts"] / BRAND["fonts"]["subtitle"]
    font_size = BRAND["fonts"]["subtitle_size"]
    if draft:
        font_size = int(font_size * DRAFT_RESOLUTION[1] / 1080)

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        time_line = lines[1]
        text = " ".join(lines[2:]).strip()
        if not text:
            continue

        try:
            parts = time_line.split(" --> ")
            start = _srt_to_sec(parts[0].strip())
            end = _srt_to_sec(parts[1].strip())
        except (IndexError, ValueError):
            continue

        duration = end - start
        if duration <= 0:
            continue

        try:
            txt_clip = TextClip(
                text,
                fontsize=font_size,
                font=str(font_path) if font_path.exists() else "Arial",
                color="white",
                bg_color="rgba(0,0,0,0.6)",
                size=(video.w * 0.9, None),
                method="caption",
            )
            txt_clip = txt_clip.set_start(start).set_duration(duration)
            txt_clip = txt_clip.set_position(("center", video.h * 0.82))
            clips.append(txt_clip)
        except Exception as e:
            logger.warning(f"Failed to create subtitle clip: {e}")

    logger.info(f"Created {len(clips)} subtitle clips")
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

        try:
            if entry["action"] == "COMFYUI_PROMPT":
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
                clip = _apply_fade(clip, fade or "in-out", duration)

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
    if not timeline_data or "timeline" not in timeline_data:
        return []

    watermark_entries = [e for e in timeline_data["timeline"] if e["action"] == "WATERMARK"]
    if not watermark_entries:
        return []

    logo_name = watermark_entries[0].get("data", "brand_logo.png")
    configured_logo = BRAND.get("watermark", {}).get("logo")
    logo_name = logo_name or configured_logo or "brand_logo.png"
    logo_path = Path(logo_name)
    if not logo_path.is_absolute():
        logo_path = PATHS["templates"] / logo_path.name
    if not logo_path.exists():
        logger.warning(f"Watermark logo not found: {logo_path}")
        return _create_text_watermark_clip(video)

    try:
        opacity = BRAND["watermark"]["opacity"]
        clip = ImageClip(str(logo_path)).set_duration(video.duration).set_start(0)
        clip = clip.resize(height=video.h // 8)
        clip = clip.set_opacity(opacity)
        margin = BRAND["watermark"].get("margin", 20)
        pos = BRAND["watermark"].get("position", "bottom-right")
        clip = _apply_position(clip, video, pos, margin)
        return [clip]
    except Exception as e:
        logger.error(f"Failed to create watermark clip: {e}")
        return []


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
    temp_path = output_path.with_suffix(".normalized.mp4")

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
            temp_path.replace(output_path)
            logger.info("Audio normalization complete")
        else:
            logger.warning(f"Audio normalization failed: {result.stderr[:200]}")
            if temp_path.exists():
                temp_path.unlink()
    except FileNotFoundError:
        logger.warning("ffmpeg not found, skipping audio normalization")
    except Exception as e:
        logger.warning(f"Audio normalization error: {e}")


def _time_to_sec(time_str):
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(time_str)


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
        clip = clip.set_opacity(BRAND["watermark"].get("opacity", 0.1))
        margin = BRAND["watermark"].get("margin", 20)
        position = BRAND["watermark"].get("position", "bottom-right")
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
