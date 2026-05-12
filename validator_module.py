import json
import logging
from pathlib import Path

from config import VALID_ACTIONS, REQUIRED_TIMELINE_FIELDS

logger = logging.getLogger("pipeline.validator")


def validate_timeline(timeline_data: dict, video_duration: float) -> dict:
    errors = []
    warnings = []

    if not timeline_data:
        errors.append("Timeline data is empty or None")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if "timeline" not in timeline_data:
        errors.append("Missing 'timeline' key in timeline data")
        return {"valid": False, "errors": errors, "warnings": warnings}

    timeline = timeline_data["timeline"]
    if not isinstance(timeline, list):
        errors.append("'timeline' must be a list")
        return {"valid": False, "errors": errors, "warnings": warnings}

    seen_ids = set()
    position_time_map = {}
    overlay_times = []
    web_gif_count = 0

    for i, entry in enumerate(timeline):
        entry_label = f"entry {i} (id={entry.get('id', '??')})"

        missing = REQUIRED_TIMELINE_FIELDS - set(entry.keys())
        if missing:
            errors.append(f"{entry_label}: missing required fields: {missing}")
            continue

        if entry["id"] in seen_ids:
            warnings.append(f"{entry_label}: duplicate id {entry['id']}")
        seen_ids.add(entry["id"])

        action = entry["action"]
        if action not in VALID_ACTIONS:
            errors.append(f"{entry_label}: invalid action '{action}'. Must be one of {VALID_ACTIONS}")
            continue
        if action == "WEB_GIF":
            web_gif_count += 1

        time_sec = _parse_time_to_sec(entry["time"])
        if time_sec < 0:
            errors.append(f"{entry_label}: invalid time '{entry['time']}'")
        elif action != "WATERMARK":
            overlay_times.append({"id": entry["id"], "time": time_sec})
            if time_sec < 2:
                warnings.append(f"{entry_label}: overlay starts in first 2 seconds; move it after the opening beat")

        duration = entry.get("duration", 0)
        if action != "WATERMARK" and duration <= 0:
            errors.append(f"{entry_label}: duration must be > 0 for action '{action}'")

        if action != "WATERMARK" and time_sec + duration > video_duration + 1:
            errors.append(
                f"{entry_label}: time + duration ({time_sec + duration:.1f}s) "
                f"exceeds video duration ({video_duration:.1f}s)"
            )

        position = entry.get("position", "")
        valid_positions = {"bottom-left", "bottom-right", "top-left", "top-right", "center", "top-center", "bottom-center"}
        if position and position not in valid_positions:
            warnings.append(f"{entry_label}: unusual position '{position}'")

        if action != "WATERMARK" and position:
            key = position
            if key not in position_time_map:
                position_time_map[key] = []
            for existing in position_time_map[key]:
                if _overlaps(time_sec, duration, existing["start"], existing["duration"]):
                    errors.append(
                        f"{entry_label}: overlaps with entry {existing['id']} at position '{position}'"
                    )
            position_time_map[key].append({"id": entry["id"], "start": time_sec, "duration": duration})

        if action == "COMFYUI_PROMPT" and len(entry["data"]) < 10:
            warnings.append(f"{entry_label}: COMFYUI_PROMPT data is very short, may produce poor results")

        if action == "WATERMARK" and time_sec != 0:
            warnings.append(f"{entry_label}: WATERMARK should start at 00:00:00")

    overlay_times.sort(key=lambda item: item["time"])
    for prev, current in zip(overlay_times, overlay_times[1:]):
        gap = current["time"] - prev["time"]
        if gap < 2:
            warnings.append(
                f"entry id={current['id']}: starts {gap:.1f}s after entry {prev['id']}; overlays too close together"
            )

    if web_gif_count > 2:
        warnings.append(f"Timeline uses {web_gif_count} WEB_GIF entries; keep medical videos cleaner with 0-2 GIFs")

    valid = len(errors) == 0
    result = {"valid": valid, "errors": errors, "warnings": warnings}

    if valid:
        logger.info(f"Timeline validation passed ({len(timeline)} entries)")
    else:
        logger.error(f"Timeline validation failed: {len(errors)} errors, {len(warnings)} warnings")

    for w in warnings:
        logger.warning(f"  Warning: {w}")
    for e in errors:
        logger.error(f"  Error: {e}")

    return result


def filter_invalid_entries(timeline_data: dict) -> dict:
    if not timeline_data or "timeline" not in timeline_data:
        return timeline_data

    valid_entries = []
    for entry in timeline_data["timeline"]:
        if not REQUIRED_TIMELINE_FIELDS.issubset(set(entry.keys())):
            continue
        if entry["action"] not in VALID_ACTIONS:
            continue
        valid_entries.append(entry)

    removed = len(timeline_data["timeline"]) - len(valid_entries)
    if removed > 0:
        logger.warning(f"Filtered out {removed} invalid timeline entries")

    timeline_data["timeline"] = valid_entries
    return timeline_data


def _parse_time_to_sec(time_str: str) -> float:
    try:
        parts = time_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except (ValueError, IndexError):
        return -1.0


def _overlaps(start1: float, dur1: float, start2: float, dur2: float) -> bool:
    end1 = start1 + dur1
    end2 = start2 + dur2
    return start1 < end2 and start2 < end1
