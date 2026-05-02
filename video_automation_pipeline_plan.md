# Automated Video Editing Pipeline Architecture
## Project: Healthcare Educational Media (Dr. Deepika)

This document outlines the complete architecture and execution plan for an automated, fully local video processing pipeline. The system processes raw medical/educational video footage, leverages local AI for creative direction and asset generation, and automatically composites a final, publication-ready MP4.

**Language constraint:** All content is **Hinglish only** (Romanized Hindi). No Devanagari script, no Hindi fonts needed. Standard Latin fonts with proper Unicode support are sufficient.

---

## 1. System Architecture & Resource Allocation

The pipeline executes **strictly one phase at a time** — no two VRAM-heavy models ever run simultaneously. Each phase fully completes and unloads before the next begins. This is critical on a 32GB RAM machine with a shared GPU.

### Core Stack
* **Audio/Transcription:** OpenAI Whisper (Local) — Hinglish language code `hi`
* **AI Director:** Gemma4 (via Ollama)
* **Image Generation:** ComfyUI with Z-Image base model workflow (`ZImageBaseModelWorkFlow.json`)
* **Graphics Generation:** Pillow (PIL)
* **Video Compositing:** MoviePy + FFmpeg
* **Web Assets:** Python Requests (Giphy/Unsplash APIs)
* **Audio Processing:** pydub + FFmpeg loudnorm filter

### VRAM & Memory Management Strategy (Strict Sequential)
Each model loads, runs, and **unloads completely** before the next starts:

| Phase | Model Loaded | VRAM Usage | Unload Action |
|-------|-------------|------------|----------------|
| 1 | Whisper | ~2-3 GB | `del model`, `gc.collect()`, `torch.cuda.empty_cache()` |
| 2 | Gemma4 (Ollama) | ~4-6 GB | `ollama stop gemma4` via subprocess |
| 3 | ComfyUI (Z-Image) | ~6-8 GB | Shutdown ComfyUI process, clear temp |
| 4 | None (CPU only) | 0 GB | MoviePy + FFmpeg run on CPU/RAM only |

**Hard rule:** Before loading any model, verify VRAM is free via `nvidia-smi`. If not, wait 10s and retry (max 3 attempts). Abort with clear error if VRAM never frees.

---

## 2. Pipeline Execution Flow

The orchestrator runs **6 sequential phases**. Each phase writes a checkpoint to `progress.json` on success. If the pipeline crashes, it resumes from the last completed phase.

### progress.json Structure
```json
{
  "current_phase": 3,
  "completed_phases": [1, 1.5, 2, 2.5],
  "input_file": "dr_deepika_ep01.mp4",
  "started_at": "2026-04-30T10:00:00",
  "last_updated": "2026-04-30T10:45:00"
}
```

**Resume logic:** On startup, read `progress.json`. Skip any phase already in `completed_phases`. If a phase is partially done, re-run it from scratch (phases are idempotent by design).

---

### Phase 1: Ingestion & Transcription (Whisper)

1. **Input:** Raw `.mp4` file placed in `/input` directory.
2. **Extraction:** Python extracts `.wav` audio using MoviePy's `AudioFileClip`.
3. **Transcription:** Whisper `large-v3` model transcribes the `.wav` file with `language="hi"` (this captures Hinglish best).
4. **Word-Level Timestamps:** Whisper outputs word-level timing data, saved as `words.json`.
5. **Output:**
   - `transcript.txt` — raw Hinglish text
   - `subtitles.srt` — timestamped Hinglish subtitles
   - `words.json` — word-level timestamps for silence detection
6. **Cleanup:** Unload Whisper model, free VRAM, run `gc.collect()` + `torch.cuda.empty_cache()`.

---

### Phase 1.5: Silence Detection & Auto-Trim

1. **Input:** `words.json` from Phase 1.
2. **Detection:** Scan word-level timestamps for gaps > 2 seconds where no speech is detected.
3. **Output:** `silence_cuts.json`:
   ```json
   [
     {"start": "00:00:12.500", "end": "00:00:15.100", "duration_ms": 2600},
     {"start": "00:03:45.200", "end": "00:03:48.000", "duration_ms": 2800}
   ]
   ```
4. **SRT Adjustment:** The `subtitles.srt` file is re-generated with silence gaps removed and all timestamps shifted accordingly. Saved as `subtitles_trimmed.srt`.
5. **Note:** The actual video trimming happens in Phase 4 (compositing). This phase only identifies what to cut and prepares adjusted subtitles.

---

### Phase 2: AI Director (Gemma4 via Ollama)

1. **Input:** `transcript.txt`, `subtitles_trimmed.srt`, and `silence_cuts.json`.
2. **System Prompt:** A strict prompt forces Gemma to act as a creative video director. Output MUST be valid JSON matching the timeline schema. No explanation text outside JSON.
3. **Processing:** Gemma analyzes the Hinglish transcript for:
   - Key medical concepts needing visual illustration
   - Emotional cues for GIF reactions
   - Myth/fact segments for text cards
   - Intro/outro lower third placement
   - Hero moments for Shorts/Reels clipping
4. **Output:** `timeline.json` with this structure:
   ```json
   {
     "video_info": {
       "source": "input.mp4",
       "duration_sec": 420,
       "language": "hinglish"
     },
     "silence_trim": true,
     "timeline": [
       {
         "id": 1,
         "time": "00:00:03",
         "duration": 5,
         "action": "LOWER_THIRD",
         "data": "Dr. Deepika - Medical Expert",
         "position": "bottom-left"
       },
       {
         "id": 2,
         "time": "00:01:15",
         "duration": 6,
         "action": "COMFYUI_PROMPT",
         "data": "High quality 3D render of a glowing human cell, 4k, educational medical illustration",
         "position": "center",
         "fade": "in-out"
       },
       {
         "id": 3,
         "time": "00:02:30",
         "duration": 3,
         "action": "WEB_GIF",
         "data": "mind blown reaction",
         "position": "top-right"
       },
       {
         "id": 4,
         "time": "00:03:00",
         "duration": 4,
         "action": "TEXT_CARD",
         "data": "Myth: Sugar directly causes diabetes.",
         "position": "center"
       },
       {
         "id": 5,
         "time": "00:00:00",
         "duration": 0,
         "action": "WATERMARK",
         "data": "brand_logo.png",
         "position": "bottom-right"
       }
     ],
     "hero_moments": [
       {"time": "00:01:15", "duration": 30, "title": "What really happens to your cells"},
       {"time": "00:03:00", "duration": 25, "title": "Diabetes myth busted"}
     ],
     "seo": {
       "title": "",
       "description": "",
       "tags": [],
       "chapters": []
     }
   }
   ```
5. **SEO Generation:** A second Gemma call generates YouTube SEO metadata (title, description, tags, chapter timestamps) from the transcript. Fills in the `seo` field.
6. **Cleanup:** Stop Gemma via `ollama stop gemma4`, free VRAM.

---

### Phase 2.5: Timeline Validation

Before generating any assets, validate `timeline.json`:

1. **Schema Validation:** Check all required fields exist (`id`, `time`, `duration`, `action`, `data`, `position`).
2. **Action Whitelist:** Only allow these actions: `LOWER_THIRD`, `COMFYUI_PROMPT`, `WEB_GIF`, `TEXT_CARD`, `WATERMARK`.
3. **Timestamp Bounds:** Every `time` + `duration` must not exceed the video's total duration.
4. **Overlap Check:** No two overlays with the same `position` should overlap in time.
5. **On Failure:** Re-run Phase 2 with the validation errors appended to the Gemma prompt. Max 3 retries. After that, skip invalid entries and log warnings to `pipeline.log`.

---

### Phase 3: Asset Generation & Retrieval (One Sub-Phase at a Time)

Python parses `timeline.json` and generates assets. **Each sub-phase loads and unloads its own tools.** No two model-dependent tools run simultaneously.

#### 3A: ComfyUI Images (Z-Image Workflow)

**Model:** Z-Image base model via `ZImageBaseModelWorkFlow.json`

Workflow node map:
- Node `76` — `UnetLoaderGGUF`: Loads `z-image-Q5_K_M.gguf`
- Node `66` — `UNETLoader`: Loads `z_image_bf16.safetensors` (alternative, not used with GGUF)
- Node `62` — `CLIPLoader`: Loads `qwen_3_4b.safetensors` (Lumina2 type)
- Node `63` — `VAELoader`: Loads `ae.safetensors`
- Node `67` — `CLIPTextEncode` (positive): **INJECT PROMPT HERE**
- Node `71` — `CLIPTextEncode` (negative): Empty string by default
- Node `68` — `EmptySD3LatentImage`: 1024x1024, batch_size 1
- Node `70` — `ModelSamplingAuraFlow`: shift=3
- Node `69` — `KSampler`: res_multistep, simple scheduler, 25 steps, CFG 4, denoise 1
- Node `65` — `VAEDecode`
- Node `9` — `SaveImage`: **SET filename_prefix HERE**

**Process:**
1. Start ComfyUI server: `python main.py --listen 127.0.0.1 --port 8188`
2. Wait for `http://127.0.0.1:8188/system_stats` to return 200 (max 60s, retry every 5s).
3. For each `COMFYUI_PROMPT` entry in `timeline.json`:
   - Deep-copy `ZImageBaseModelWorkFlow.json` as template
   - Inject the prompt text into node `67` `inputs.text` field
   - Set a random seed on node `69` `inputs.seed`
   - Set `filename_prefix` on node `9` `inputs.filename_prefix` to `gen_{id:03d}`
   - POST the modified workflow to `http://127.0.0.1:8188/prompt`
   - Poll `http://127.0.0.1:8188/history/{prompt_id}` every 2s until status is complete
   - Copy the output from ComfyUI's output directory to `/assets/gen_images/gen_{id:03d}.png`
   - Resize to appropriate overlay size (1920x1080 for full-screen, 960x540 for inset) using Pillow
4. Shutdown ComfyUI process completely. Free VRAM.
5. **Error handling:** If ComfyUI fails to start after 3 attempts, log error and skip all `COMFYUI_PROMPT` entries. Pipeline continues without generated images.

#### 3B: Web Assets (GIFs/Stock)

1. For each `WEB_GIF` entry, query Giphy API with the `data` field as search term.
2. Download the top result. Save to `/assets/web/gif_{id:03d}.mp4` or `.gif`.
3. For stock images (if any `WEB_STOCK` entries added later), query Unsplash API, download, save to `/assets/web/stock_{id:03d}.jpg`.
4. **Retry logic:** Exponential backoff (1s, 2s, 4s) on API failures. Skip on 3rd failure and log warning.

#### 3C: Static Graphics (Pillow)

1. For each `TEXT_CARD` entry: Load branded template from `/assets/templates/text_card_base.png`, overlay text using brand font (from `brand_profile.yaml`), save to `/assets/graphics/card_{id:03d}.png`.
2. For each `LOWER_THIRD` entry: Load lower-third template, overlay name/title text, save to `/assets/graphics/lower_third_{id:03d}.png`.
3. **Hinglish rendering:** Use `Poppins-Medium.ttf` or similar clean Latin font. Since content is Hinglish (Roman script only), no Devanagari font is needed.
4. All graphics output as transparent PNGs for compositing.

---

### Phase 4: Compositing & Export (MoviePy + FFmpeg)

**CPU-only phase.** No GPU models loaded.

1. **Load Source:** MoviePy loads the original raw `.mp4`.
2. **Silence Trimming:** Apply cuts from `silence_cuts.json` — remove silent segments and concatenate remaining clips. Adjust all `timeline.json` timestamps to account for removed segments.
3. **Subtitles:** MoviePy parses `subtitles_trimmed.srt` and burns hardcoded subtitles into the lower-center of the frame (semi-transparent dark background box for readability, Hinglish text in Poppins font).
4. **Watermark:** Persistent brand logo overlay at the position specified in `timeline.json` (usually bottom-right, 10% opacity).
5. **Timeline Assembly:** Python iterates through `timeline.json` and layers assets:
   - Lower thirds: Slide in from left at exact timestamps, hold for `duration` seconds.
   - ComfyUI images: Fade in/out over footage at specified timestamps.
   - Web GIFs: Pop up at specified position using `CompositeVideoClip`.
   - Text cards: Full-screen overlay with fade transitions.
6. **Audio Mastering:**
   - Background music added at -20dB under speech.
   - **Ducking:** When speech is detected (from `.srt` timing), music volume drops to -30dB.
   - **Loudness normalization:** Apply `ffmpeg loudnorm` filter targeting -14 LUFS (YouTube standard).
   - Fade out music 3 seconds before video end.
7. **Export Profiles:**

   | Profile | Resolution | Aspect | Platform | Flag |
   |---------|-----------|--------|----------|------|
   | Landscape | 1920x1080 | 16:9 | YouTube | default |
   | Portrait | 1080x1920 | 9:16 | Reels/Shorts | `--shorts` |
   | Square | 1080x1080 | 1:1 | Instagram | `--square` |

8. **Draft/Preview Mode:** With `--draft` flag:
   - Render at 480p instead of 1080p
   - Skip ComfyUI image generation (use placeholder rectangles)
   - Skip GIF downloads (use placeholder text)
   - Fast encode with lower bitrate
   - Output filename: `output_draft.mp4`
   - Purpose: Rapid iteration on timeline edits before final render

9. **Final Export:** FFmpeg encodes to `/output/{filename}_{profile}.mp4` with H.264 codec.

---

### Phase 5: SEO Metadata & Thumbnail Generation

1. **SEO File:** Write `seo_metadata.json` with the title, description, tags, and chapters generated by Gemma in Phase 2.
2. **Thumbnail:** Extract the best frame from a hero moment (or use a ComfyUI-generated image) and overlay branded text using Pillow. Save as `/output/{filename}_thumb.jpg` (1280x720).
3. **Chapters File:** Write `chapters.txt` in YouTube-compatible format.

---

## 3. Error Handling & Retry Strategy

| Component | Failure Mode | Retry Strategy | Fallback |
|-----------|-------------|----------------|----------|
| Whisper | Model load fails | Retry 3x, 10s gap | Abort pipeline |
| Ollama/Gemma | API timeout / invalid JSON | Retry 3x with exponential backoff | Skip timeline, manual mode |
| ComfyUI | Server won't start | Retry 3x, 30s gap | Skip all COMFYUI_PROMPT entries |
| ComfyUI | Generation fails mid-way | Retry individual prompt 2x | Skip that image, log warning |
| Giphy/Unsplash | API timeout | Retry 3x exponential backoff | Skip that GIF, log warning |
| Pillow | Font/template missing | Check at startup before pipeline runs | Abort with clear message |
| MoviePy/FFmpeg | Render crash | Resume from progress.json checkpoint | Re-run from last completed phase |

---

## 4. Directory Structure

```
/video_automation_pipeline
│
├── /input                    # Drop raw Dr. Deepika footage here
├── /output                   # Final rendered videos + thumbnails + SEO
├── /assets
│   ├── /templates            # Pillow base images (logos, lower third bases)
│   ├── /gen_images           # Output from ComfyUI (Z-Image)
│   ├── /web                  # Downloaded GIFs/Stock
│   ├── /graphics             # Generated Pillow text cards + lower thirds
│   └── /fonts                # Poppins-Medium.ttf, etc.
│
├── /models                   # Local model weights (Whisper)
├── /workflows                # ComfyUI workflow JSONs
│   └── ZImageBaseModelWorkFlow.json
│
├── /temp                     # Intermediate files (audio, etc.)
│
├── orchestrator.py           # Main execution script + progress tracking
├── audio_module.py           # Whisper integration + silence detection
├── director_module.py        # Ollama/Gemma4 API calls + SEO generation
├── validator_module.py       # Timeline JSON validation
├── asset_module.py           # ComfyUI API, Pillow, Web requests
├── compositing_module.py     # MoviePy + FFmpeg logic
├── config.py                 # Shared config, paths, retry settings
├── brand_profile.yaml        # Brand colors, fonts, logo paths, templates
├── progress.json             # Auto-generated checkpoint file
└── pipeline.log              # Auto-generated execution log
```

---

## 5. Brand Profile (brand_profile.yaml)

```yaml
brand:
  name: "Dr. Deepika"
  primary_color: "#1A73E8"
  accent_color: "#FF6D00"
  bg_color: "#FFFFFF"
  text_color: "#212121"

fonts:
  subtitle: "Poppins-Medium.ttf"
  text_card: "Poppins-Bold.ttf"
  lower_third: "Poppins-SemiBold.ttf"
  subtitle_size: 42
  text_card_size: 64
  lower_third_size: 36

watermark:
  logo: "assets/templates/brand_logo.png"
  opacity: 0.1
  position: "bottom-right"
  margin: 20

lower_third:
  template: "assets/templates/lower_third_base.png"
  height: 80
  animation: "slide_left"

text_card:
  template: "assets/templates/text_card_base.png"
  animation: "fade"

music:
  default_track: "assets/music/background.mp3"
  volume_under_speech: -30
  volume_normal: -20
  fade_out_seconds: 3

export:
  youtube_lufs: -14
  default_resolution: "1920x1080"
  draft_resolution: "854x480"
  codec: "libx264"
  preset: "medium"
  crf: 23
```

---

## 6. CLI Usage

```bash
# Full pipeline (landscape YouTube)
python orchestrator.py --input input/dr_deepika_ep01.mp4

# Draft/preview mode (fast iteration)
python orchestrator.py --input input/dr_deepika_ep01.mp4 --draft

# Shorts/Reels export (9:16 portrait)
python orchestrator.py --input input/dr_deepika_ep01.mp4 --shorts

# Instagram square export
python orchestrator.py --input input/dr_deepika_ep01.mp4 --square

# Resume from last checkpoint
python orchestrator.py --input input/dr_deepika_ep01.mp4 --resume

# Skip specific phases (e.g., re-render after manual timeline edit)
python orchestrator.py --input input/dr_deepika_ep01.mp4 --skip 1,1.5,2,2.5,3

# Generate all formats at once
python orchestrator.py --input input/dr_deepika_ep01.mp4 --all-formats
```

---

## 7. Execution Order Summary

```
START
  │
  ▼
┌─────────────────────────────┐
│ Phase 1:  Whisper            │  Load model → transcribe → unload
│   → transcript.txt           │
│   → subtitles.srt            │
│   → words.json               │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 1.5: Silence Detect    │  No model loaded (pure Python)
│   → silence_cuts.json        │
│   → subtitles_trimmed.srt    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 2:  Gemma4 (Ollama)    │  Load model → generate timeline → unload
│   → timeline.json            │
│   → seo_metadata.json        │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 2.5: Validate          │  No model loaded (pure Python)
│   → validated timeline       │  Retry Phase 2 on failure
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 3A: ComfyUI (Z-Image) │  Load model → gen images → unload
│   → /assets/gen_images/      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 3B: Web Assets         │  No model loaded (HTTP requests)
│   → /assets/web/             │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 3C: Pillow Graphics    │  No model loaded (CPU rendering)
│   → /assets/graphics/        │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 4:  Compositing        │  CPU only (MoviePy + FFmpeg)
│   → output.mp4               │  + silence trim, subs, overlays
│   → output_shorts.mp4        │  + audio mastering, loudnorm
│   → output_thumb.jpg          │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Phase 5:  SEO & Thumbnail    │  No model loaded (file writes)
│   → seo_metadata.json        │
│   → chapters.txt              │
│   → thumbnail.jpg             │
└─────────────────────────────┘
  │
  ▼
DONE
```

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Strict sequential execution | 32GB RAM + shared GPU cannot run Whisper, Ollama, and ComfyUI simultaneously |
| Hinglish only, no Devanagari | Target audience reads Romanized Hindi; avoids font/rendering complexity |
| Z-Image via ComfyUI API | Reuses existing workflow JSON; prompt injection is clean and predictable |
| Silence cutting as separate phase | Keeps Phase 1 simple; trimming logic is independent of transcription |
| Timeline validation before asset gen | Prevents wasting GPU hours on hallucinated timestamps or invalid actions |
| progress.json checkpoint | A 10-minute render crash shouldn't mean restarting from scratch |
| --draft flag | Timeline iteration is the bottleneck; fast previews save hours |
| YouTube loudnorm (-14 LUFS) | Prevents viewers from having to adjust volume vs. other videos |
| Brand profile as YAML | One file to change all branding; no hardcoded paths in modules |
