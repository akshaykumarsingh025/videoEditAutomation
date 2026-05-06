import os
import yaml
import logging
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()


def load_env_file(env_path: Path | None = None):
    env_path = env_path or BASE_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file()

PATHS = {
    "input": BASE_DIR / "input",
    "output": BASE_DIR / "output",
    "temp": BASE_DIR / "temp",
    "models": BASE_DIR / "models",
    "workflows": BASE_DIR / "workflows",
    "templates": BASE_DIR / "assets" / "templates",
    "gen_images": BASE_DIR / "assets" / "gen_images",
    "web": BASE_DIR / "assets" / "web",
    "graphics": BASE_DIR / "assets" / "graphics",
    "fonts": BASE_DIR / "assets" / "fonts",
    "music": BASE_DIR / "assets" / "music",
}

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_WORKFLOW = PATHS["workflows"] / "ZImageBaseModelWorkFlow.json"
COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", BASE_DIR / "ComfyUI")).resolve()
COMFYUI_STARTUP_TIMEOUT = 60
COMFYUI_POLL_INTERVAL = 2

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = os.environ.get("VIDEO_AI_MODEL", "").strip() or None

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
WHISPER_LANGUAGE = "hi"

SILENCE_THRESHOLD_MS = 2000

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1

EXPORT_PROFILES = {
    "landscape": {"resolution": (1920, 1080), "aspect": "16:9"},
    "shorts": {"resolution": (1080, 1920), "aspect": "9:16"},
    "square": {"resolution": (1080, 1080), "aspect": "1:1"},
}

DRAFT_RESOLUTION = (854, 480)

VALID_ACTIONS = {"BROLL_IMAGE", "LOWER_THIRD", "TEXT_CARD"}

REQUIRED_TIMELINE_FIELDS = {"id", "time", "duration", "action", "data", "position"}

LOG_FILE = BASE_DIR / "pipeline.log"
PROGRESS_FILE = BASE_DIR / "progress.json"


def load_brand_profile():
    profile_path = BASE_DIR / "brand_profile.yaml"
    with open(profile_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


BRAND = load_brand_profile()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("pipeline")


def ensure_dirs():
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
