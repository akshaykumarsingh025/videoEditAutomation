# Video Edit Automation

Automated local video editing pipeline for Dr. Deepika style healthcare videos. It extracts audio, transcribes with Whisper, asks an Ollama model to create an edit timeline and SEO metadata, generates supporting assets, then renders YouTube, Shorts/Reels, or square exports with MoviePy and ffmpeg.

## Quick Setup

1. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

2. Make sure external tools are available:

```powershell
ffmpeg -version
ollama list
```

3. Copy `.env.example` to `.env` and choose a model from `ollama list`.

Local and cloud Ollama models both work. The model name must match exactly, for example `gemma4:e4b`, `gemma4:26b`, or `gemma4:31b-cloud`.

4. Optional assets:

- Put fonts in `assets/fonts/` matching `brand_profile.yaml`.
- Put watermark logo in `assets/templates/brand_logo.png`.
- Put background music in `assets/music/background.mp3`.
- Set `GIPHY_API_KEY` if you want WEB_GIF assets.
- Set `COMFYUI_DIR` if ComfyUI is installed outside this project.
- `COMFYUI_KEEP_ALIVE=true` keeps ComfyUI running after image generation so later steps continue without the pipeline shutting it down.

## Useful Commands

List available Ollama models:

```powershell
python orchestrator.py --list-models
```

Check setup before a long render:

```powershell
python orchestrator.py --check
```

Run a fast draft:

```powershell
python orchestrator.py --input input\video.mp4 --draft --model gemma4:e4b
```

Export all formats:

```powershell
python orchestrator.py --input input\video.mp4 --all-formats --model gemma4:e4b
```

If `--model` is not provided and `VIDEO_AI_MODEL` is not set, the script will show `ollama list` and ask you to pick a model.

## Video Quality Notes

- Draft mode skips ComfyUI image generation for speed.
- Generated ComfyUI images are cached by video/entry id, so a resumed run skips images that already exist.
- Shorts and square exports use a background-plus-foreground reframe instead of stretching the source video.
- The AI director prompt favors clean educational overlays, strong hook timing, and fewer reaction GIFs.
- Timeline validation warns when overlays are too close together or start too early.
