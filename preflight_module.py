import importlib.metadata
import os
import shutil
from pathlib import Path

from config import (
    BRAND,
    COMFYUI_DIR,
    COMFYUI_WORKFLOW,
    PATHS,
)
from director_module import format_ollama_models, list_ollama_models


REQUIRED_PACKAGES = {
    "moviepy": "moviepy",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "requests": "requests",
    "openai-whisper": "whisper",
}


def run_preflight(model_name: str | None = None, require_model: bool = True, print_report: bool = True) -> dict:
    report = {
        "errors": [],
        "warnings": [],
        "info": [],
        "models": [],
    }

    _check_commands(report, require_model)
    _check_ollama(report, model_name, require_model)
    _check_packages(report)
    _check_assets(report)
    _check_comfyui(report)
    _check_optional_integrations(report)

    if print_report:
        print(format_preflight_report(report))

    return report


def format_preflight_report(report: dict) -> str:
    lines = ["Preflight report"]

    if report["errors"]:
        lines.append("\nErrors:")
        lines.extend(f"- {item}" for item in report["errors"])

    if report["warnings"]:
        lines.append("\nWarnings:")
        lines.extend(f"- {item}" for item in report["warnings"])

    if report["info"]:
        lines.append("\nInfo:")
        lines.extend(f"- {item}" for item in report["info"])

    if report["models"]:
        lines.append("\nOllama models:")
        lines.append(format_ollama_models(report["models"]))

    if not report["errors"] and not report["warnings"]:
        lines.append("\nEverything important looks ready.")

    return "\n".join(lines)


def _check_commands(report: dict, require_model: bool):
    if not shutil.which("ffmpeg"):
        report["errors"].append("ffmpeg was not found on PATH. MoviePy and audio normalization need it.")
    else:
        report["info"].append("ffmpeg is available.")

    if require_model and not shutil.which("ollama"):
        report["errors"].append("ollama was not found on PATH. The AI director needs Ollama.")
    elif shutil.which("ollama"):
        report["info"].append("ollama is available.")


def _check_ollama(report: dict, model_name: str | None, require_model: bool):
    if not shutil.which("ollama"):
        return

    try:
        models = list_ollama_models()
    except Exception as e:
        if require_model:
            report["errors"].append(str(e))
        else:
            report["warnings"].append(str(e))
        return

    report["models"] = models
    if not models and require_model:
        report["errors"].append("No Ollama models are listed. Pull or enable a model first.")
        return

    if model_name and model_name not in {model["name"] for model in models}:
        message = f"Selected Ollama model '{model_name}' is not in `ollama list`."
        if require_model:
            report["errors"].append(message)
        else:
            report["warnings"].append(message)


def _check_packages(report: dict):
    for package_name in REQUIRED_PACKAGES:
        try:
            version = importlib.metadata.version(package_name)
            report["info"].append(f"{package_name} {version} is installed.")
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"Python package '{package_name}' is not installed.")

    numpy_version = _package_version("numpy")
    torch_version = _package_version("torch")
    if numpy_version:
        report["info"].append(f"numpy {numpy_version} is installed.")
    if torch_version:
        report["info"].append(f"torch {torch_version} is installed.")
    if numpy_version and torch_version and _major(numpy_version) >= 2 and _torch_minor(torch_version) <= (2, 2):
        report["warnings"].append(
            "Torch/Whisper may warn or behave unstably with NumPy 2.x. Pin `numpy<2` in this project environment."
        )


def _check_assets(report: dict):
    for font_key in ("subtitle", "text_card", "lower_third"):
        font_name = BRAND.get("fonts", {}).get(font_key)
        if font_name and not (PATHS["fonts"] / font_name).exists():
            report["warnings"].append(f"Configured font missing: assets/fonts/{font_name}. Arial/default font will be used.")

    logo_value = BRAND.get("watermark", {}).get("logo", "brand_logo.png")
    logo_path = PATHS["templates"] / Path(logo_value).name
    if not logo_path.exists():
        report["warnings"].append("Watermark logo missing. A text watermark fallback will be attempted.")

    track_value = BRAND.get("music", {}).get("default_track", "background.mp3")
    track_path = PATHS["music"] / Path(track_value).name
    if not track_path.exists():
        report["warnings"].append("Background music missing. Export will continue without music.")


def _check_comfyui(report: dict):
    if not COMFYUI_WORKFLOW.exists():
        report["warnings"].append(f"ComfyUI workflow missing: {COMFYUI_WORKFLOW}")
    if not COMFYUI_DIR.exists():
        report["warnings"].append(
            f"ComfyUI directory missing: {COMFYUI_DIR}. COMFYUI_PROMPT entries will be skipped unless a server is already running."
        )


def _check_optional_integrations(report: dict):
    if not os.environ.get("GIPHY_API_KEY"):
        report["warnings"].append("GIPHY_API_KEY is not set. WEB_GIF assets will be skipped.")


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _major(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return 0


def _torch_minor(version: str) -> tuple[int, int]:
    base = version.split("+", 1)[0]
    parts = base.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0
