# Feature Plan — Video Edit Automation

A roadmap of features to make the pipeline produce richer, more engaging videos.

## Where the pipeline is today

The pipeline (`orchestrator.py`) runs these phases:

1. **Ingest & Transcribe** — ffmpeg audio extract + Whisper (`audio_module.py`), auto-transliterated to Hinglish.
2. **Silence detection** — implemented but effectively **disabled** (`SILENCE_THRESHOLD_MS=0` in `config.py`).
3. **AI Director** — Ollama generates a JSON timeline + SEO (`director_module.py`).
4. **Validation** — timeline sanity checks (`validator_module.py`).
5. **Asset generation** — ComfyUI Krea2 Turbo images, Giphy GIFs, Pillow graphics (`asset_module.py`).
6. **Compositing** — MoviePy layering + export (`compositing_module.py`).
7. **SEO + thumbnail**.

**Visual elements the director actually emits today:** `BROLL_IMAGE` and `TEXT_CARD` (per the current `DIRECTOR_SYSTEM_PROMPT`). `LOWER_THIRD`, `WATERMARK`, `WEB_GIF`, and `COMFYUI_PROMPT` action types still exist in code but the director no longer produces them. Subtitles, silence-cut editing, and smart-zoom code all exist but are dormant/removed from the active flow.

So the honest state is: **images + text cards + a static info card + background music + a watermark.** Everything below is what we can add on top.

---

## Tier 1 — High impact, low effort (build first)

### 1.1 Re-enable karaoke subtitles (already 90% built)
The entire Pillow-based karaoke subtitle renderer exists in `compositing_module.py` (`_create_subtitle_clips`, `_create_karaoke_subtitles_pillow`) but is **not called** in `composite_video()`. Word-level timing JSON is already produced by Whisper.
- **Do:** wire `_create_subtitle_clips` back into the `clips` list in `composite_video()`, gated by a `brand_profile.yaml` flag (`subtitles.enabled`).
- **Add:** true word-by-word highlight color (the code renders chunks but doesn't yet color the "active" word). Retention on Shorts/Reels lives and dies on burned-in captions — this is the single highest-ROI item.

### 1.2 B-roll transition variety
Today B-roll only cross-fades in/out (`_apply_fade`). Add a small library of transitions the director can pick per entry via the existing `fx` field:
- slide-in (L/R/up/down), whip-pan blur, dip-to-black/white, zoom-punch, quick-cut.
- Implement as functions in `compositing_module.py`; extend the `fx` vocabulary in the director prompt.

### 1.3 Sound design / SFX layer
`_add_music` only lays one looped bed track. Add a lightweight SFX system:
- "whoosh" on B-roll cut-in, "ding"/"pop" on text-card entrance, subtle riser into hero moments.
- Store SFX in `assets/sfx/`, trigger off timeline entry start times. Big perceived-quality jump for near-zero render cost.

### 1.4 Ducking (music under speech)
`brand_profile.yaml` already declares `volume_under_speech: -30` but nothing implements it. Use the Whisper word timestamps (already on disk) to auto-duck the music bed during speech and lift it during pauses. Sidechain-style ducking via ffmpeg `sidechaincompress` or manual volume automation in MoviePy.

### 1.5 Re-enable silence trimming as an option
`detect_silence` / `adjust_srt_for_silence` are fully built. The problem that got it disabled was over-aggressive cutting of dialogue. Fix instead of delete:
- Add a `silence.min_gap_ms` (e.g. 700ms) and `silence.keep_padding_ms` (e.g. 150ms) so only genuine dead air is removed, never mid-sentence breath.
- Expose as `--trim-silence` CLI flag, off by default.

---

## Tier 2 — New visual variety (medium effort)

### 2.1 More card types from the director
The renderer already supports `LOWER_THIRD`. Reintroduce and add:
- **QUOTE_CARD** — pull-quote styling for the doctor's key sentence.
- **STAT_CARD** — big number + label (e.g. "9–45 saal", "2L paani/din").
- **LIST_CARD** — animated bullet reveal for "3 cheezein yaad rakhein".
- **CTA_CARD** — end-screen subscribe / clinic contact.
- **CHAPTER_TITLE** — full-screen section divider between topics.

Each is a Pillow template in `asset_module.py` + a render branch in `_create_timeline_clips` + prompt vocabulary in the director.

### 2.2 Animated progress/topic bar
A thin top bar showing chapter progress (uses the `chapters` already generated for SEO). Cheap, and makes long-form feel structured.

### 2.3 Intro hook + outro end-card
- **Intro (0–3s):** auto-built animated title card from the SEO title over a blurred first frame.
- **Outro:** end-card with subscribe CTA + clinic info + "next video" thumbnail slot.
- Both assembled with `concatenate_videoclips` around the main body.

### 2.4 Emoji / sticker accents
Overlay relevant emoji or simple vector stickers keyed to keywords ("⚠️" on warnings, "✅" on advice). Pull from a local sticker pack; director tags entries with a `sticker` field.

### 2.5 B-roll Ken Burns upgrade
`_apply_ken_burns` exists but current B-roll uses a plain resize. Give each image a randomized pan+zoom vector (not just center zoom) for a more "edited" feel, with direction alternating.

---

## Tier 3 — Smarter AI director (medium/high effort)

### 3.1 Two-pass director
Pass 1: outline chapters + hero moments from transcript. Pass 2: fill overlays per chapter. Produces better pacing and coverage than the current single 4096-token dump (which is why `_enforce_timeline_density` has to auto-fill gaps).

### 3.2 Emphasis-driven text cards
Use word timestamps + simple heuristics (numbers, negation, imperatives, pitch/volume peaks from the audio) to decide *where* text cards land, instead of the model guessing. More accurate, fewer generic filler cards.

### 3.3 B-roll relevance scoring / regeneration
After ComfyUI generates images, run a quick CLIP or vision-model check that the image matches the spoken sentence; auto-regenerate the worst matches. Reduces "generic doctor in clinic" filler.

### 3.4 Per-video style presets
`brand_profile.yaml` is one fixed brand. Add named presets (calm/educational vs punchy/shorts) selectable via CLI, changing colors, animation speed, card density, and music.

---

## Tier 4 — Distribution & workflow (high leverage, low glamour)

### 4.1 Auto-generate Shorts/Reels from hero moments
`hero_moments` is already produced but only used to pick a thumbnail frame. Auto-cut each hero moment into a standalone 9:16 clip with punchy captions — turns one long video into 3–4 Shorts automatically.

### 4.2 Thumbnail variants + A/B
`generate_thumbnail` makes one basic thumb. Generate 3 variants (different frame, headline, layout) so the user can pick/test.

### 4.3 Multi-language / dubbing track
Whisper already segments; add a translation pass (Ollama) + TTS to produce a second audio track or a fully dubbed export for wider reach.

### 4.4 Web/desktop review UI
Right now everything is CLI + editing `timeline.json` by hand. A small local UI (Gradio/Streamlit) to preview the timeline, tweak card text, re-order, and re-render single phases would massively speed iteration. `progress.json` already exposes phase state to build on.

### 4.5 Direct YouTube upload
Phase 5 writes `_seo.json` and `_chapters.txt`. Add an optional YouTube Data API upload step that pushes the video + title + description + chapters + thumbnail in one command.

---

## Tier 5 — Quality & robustness

- **Caption safe-area / face-avoidance** — position cards/subtitles so they never cover the speaker's face (OpenCV face detect already available via `_detect_face_center`).
- **Color grading pass** — a subtle LUT/contrast/saturation lift on the source for a more "produced" look.
- **Loudness already handled** (-14 LUFS) — extend to per-segment normalization for consistent speech level.
- **Render performance** — MoviePy is slow; consider an ffmpeg-filter-graph fast path for the final composite, or GPU encode (`h264_nvenc`).
- **Config-driven feature flags** — every feature above should be a toggle in `brand_profile.yaml` so nothing is hard-wired.

---

## Suggested build order

1. **Karaoke subtitles** (wire up existing code) — biggest retention win, least new code.
2. **SFX + music ducking** — cheap, large perceived-quality jump.
3. **Transition variety + Ken Burns upgrade** — makes B-roll feel edited.
4. **New card types** (stat/quote/list/CTA) — visual variety.
5. **Auto-Shorts from hero moments** — multiplies output per video.
6. **Two-pass director** — fixes pacing at the source.
7. **Review UI + YouTube upload** — workflow speed.

## Open questions for you

- Are subtitles wanted back on, or were they removed on purpose for this channel's style?
- Primary target — long-form YouTube, Shorts/Reels, or both equally?
- Is faster rendering a priority (current MoviePy path is slow), or is quality the only axis that matters right now?
