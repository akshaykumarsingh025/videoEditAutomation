import os
import sys
import glob
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
VIDEO_EXTS = ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm", "*.wmv", "*.flv", "*.m4v", "*.ts", "*.mts", "*.3gp"]

BANNER = r"""
  ============================================================
   VIDEO AUTOMATION PIPELINE - Dr. Deepika
  ============================================================

   [1] Full Pipeline         (1080p, all assets)
   [2] Draft / Preview       (480p, fast, skip ComfyUI)
   [3] Shorts / Reels        (9:16 portrait)
   [4] Instagram Square      (1:1 square)
   [5] All Formats           (landscape + shorts + square)
   [6] Resume Last Run       (continue from checkpoint)
   [7] Skip Phases           (choose which phases to skip)
   [8] Open Output Folder
   [9] View Pipeline Log
   [0] Exit

  ============================================================
"""


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def find_videos():
    files = []
    for ext in VIDEO_EXTS:
        files.extend(sorted(INPUT_DIR.glob(ext)))
        files.extend(sorted(INPUT_DIR.glob(ext.upper())))
    return list(dict.fromkeys(files))


def pick_file():
    print()
    print("  ============================================================")
    print("   SELECT INPUT VIDEO")
    print("  ============================================================")
    print()
    print("   [1] Pick from /input folder")
    print("   [2] Type path manually")
    print("   [3] Back to menu")
    print()

    choice = input("  Select [1-3]: ").strip()

    if choice == "1":
        return pick_from_input()
    elif choice == "2":
        return type_path()
    return None


def pick_from_input():
    videos = find_videos()

    if not videos:
        print()
        print("  No video files found in input/ folder!")
        print(f"  Drop any video file there and try again.")
        print(f"  Supported: {', '.join(e.replace('*','') for e in VIDEO_EXTS)}")
        input("\n  Press Enter to continue...")
        return None

    print()
    for i, v in enumerate(videos, 1):
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"   [{i}] {v.name}  ({size_mb:.1f} MB)")

    print()
    sel = input("  Select file number (0 to cancel): ").strip()

    try:
        idx = int(sel)
        if idx == 0:
            return None
        if 1 <= idx <= len(videos):
            selected = videos[idx - 1]
            print(f"\n  Selected: {selected}")
            return str(selected)
    except ValueError:
        pass

    print("  Invalid selection.")
    input("  Press Enter to continue...")
    return None


def type_path():
    print()
    path = input("  Enter full path to video: ").strip().strip('"')

    if not path:
        return None

    if not Path(path).exists():
        print(f"  File not found: {path}")
        input("  Press Enter to continue...")
        return None

    print(f"  Selected: {path}")
    return path


def run_pipeline(filepath, extra_args=None):
    cmd = [sys.executable, str(BASE_DIR / "orchestrator.py"), "--input", filepath]
    if extra_args:
        cmd.extend(extra_args)

    print()
    print(f"  Running: {' '.join(cmd)}")
    print("  " + "-" * 58)
    print()

    try:
        subprocess.run(cmd, cwd=str(BASE_DIR))
    except KeyboardInterrupt:
        print("\n  Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n  Error: {e}")

    print()
    input("  Press Enter to continue...")


def view_log():
    print()
    print("  ============================================================")
    print("   PIPELINE LOG (last 80 lines)")
    print("  ============================================================")
    print()

    log_path = BASE_DIR / "pipeline.log"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-80:]:
            print(f"  {line}")
    else:
        print("  No log file found yet.")

    print()
    input("  Press Enter to continue...")


def main():
    while True:
        clear()
        print(BANNER)

        choice = input("  Select option [0-9]: ").strip()

        if choice == "0":
            print("\n  Bye!")
            sys.exit(0)

        elif choice == "1":
            filepath = pick_file()
            if filepath:
                run_pipeline(filepath)

        elif choice == "2":
            filepath = pick_file()
            if filepath:
                run_pipeline(filepath, ["--draft"])

        elif choice == "3":
            filepath = pick_file()
            if filepath:
                run_pipeline(filepath, ["--shorts"])

        elif choice == "4":
            filepath = pick_file()
            if filepath:
                run_pipeline(filepath, ["--square"])

        elif choice == "5":
            filepath = pick_file()
            if filepath:
                run_pipeline(filepath, ["--all-formats"])

        elif choice == "6":
            print("\n  Looking for last checkpoint...\n")
            run_pipeline("dummy", ["--resume"])

        elif choice == "7":
            print()
            print("  ============================================================")
            print("   SKIP PHASES")
            print("  ============================================================")
            print()
            print("   Available phases:")
            print("     1   = Ingestion & Transcription (Whisper)")
            print("     1.5 = Silence Detection & Auto-Trim")
            print("     2   = AI Director (Gemma4)")
            print("     2.5 = Timeline Validation")
            print("     3   = Asset Generation (ComfyUI + Pillow + Web)")
            print("     4   = Compositing & Export")
            print("     5   = SEO & Thumbnail")
            print()
            print("   Example: 1,1.5,2  (skip straight to asset generation)")
            print()

            skip = input("  Phases to skip: ").strip()
            filepath = pick_file()
            if filepath:
                run_pipeline(filepath, ["--skip", skip])

        elif choice == "8":
            print("\n  Opening output folder...")
            os.startfile(str(OUTPUT_DIR)) if os.name == "nt" else subprocess.run(["xdg-open", str(OUTPUT_DIR)])

        elif choice == "9":
            view_log()


if __name__ == "__main__":
    main()
