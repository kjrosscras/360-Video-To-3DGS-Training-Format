import os
import sys
import json
import cv2
import time
import threading
import subprocess
from pathlib import Path

from tkinter import (
    Label, Entry, StringVar, Frame, Checkbutton, BooleanVar,
    filedialog, Button, Text, END, BOTH, DISABLED, NORMAL
)
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import ttk
from PIL import Image, ImageTk

import numpy as np  # NEW: for mask processing

# =========================
# Angle profiles & helpers
# =========================

# With masking: your angles (ref at index 0)
MASKING_PITCH_YAW_PAIRS = [
    (0, 90),   # Reference Pose (ref_idx = 0)
    (34, 0),
    (-42, 0),
    (0, 42),
    (0, -42),
    (42, 180),
    (-34, 180),
    (0, 222),
    (0, 138),
]
MASKING_REF_IDX = 0

# Without masking: your angles (ref at index 0)
NO_MASKING_PITCH_YAW_PAIRS = [
    (0, 90),   # Reference Pose (ref_idx = 0)
    (32, 0),
    (-42, 0),
    (0, 42),
    (0, -25),
    (42, 180),
    (-32, 180),
    (0, 205),
    (0, 138),
]
NO_MASKING_REF_IDX = 0

VALID_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}

def write_rotation_override(root_dir: Path, pairs, ref_idx: int) -> Path:
    payload = {"pitch_yaw_pairs": pairs, "ref_idx": ref_idx}
    out_path = root_dir / "rotation_override.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

def open_in_explorer(path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass

def list_videos(video_dir: Path):
    return sorted([p for p in video_dir.glob("*") if p.suffix.lower() in VALID_VIDEO_EXT])

# =========================
# UI helpers (set from thread)
# =========================

_root = None
status_var = None
progress_main = None
progress_sub = None
log_text = None

use_masking = None
frame_interval = None
drop_zone = None
run_btn = None
browse_btn = None

def ui_status(msg: str):
    status_var.set(msg)
    _root.update_idletasks()

def ui_log(msg: str):
    log_text.configure(state=NORMAL)
    log_text.insert(END, msg.rstrip() + "\n")
    log_text.see(END)
    log_text.configure(state=DISABLED)
    _root.update_idletasks()

def ui_main_progress(value: float | None = None, indeterminate: bool = False):
    try:
        progress_main.stop()
    except Exception:
        pass
    if indeterminate:
        progress_main["mode"] = "indeterminate"
        progress_main.start(12)
    else:
        progress_main["mode"] = "determinate"
        progress_main["value"] = 0 if value is None else value
    _root.update_idletasks()

def ui_sub_progress(value: float | None = None, indeterminate: bool = False):
    try:
        progress_sub.stop()
    except Exception:
        pass
    if indeterminate:
        progress_sub["mode"] = "indeterminate"
        progress_sub.start(12)
    else:
        progress_sub["mode"] = "determinate"
        progress_sub["value"] = 0 if value is None else value
    _root.update_idletasks()

def ui_disable_inputs(disabled=True):
    state = DISABLED if disabled else NORMAL
    drop_zone.configure(state="disabled" if disabled else "normal")
    run_btn.configure(state=state)
    browse_btn.configure(state=state)
    # removed: loop over _yolo_widgets


# =========================
# Frame extraction with progress
# =========================

def extract_frames_with_progress(video_dir: Path, interval_seconds: float) -> int:
    """
    Extract frames from all videos directly inside video_dir into:
        video_dir/frames/
    (All frames in ONE folder; filenames are prefixed by video name.)

    Returns: number of videos processed.
    """
    output_base_dir = video_dir / "frames"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    vids = list_videos(video_dir)
    if not vids:
        ui_log(f"[WARN] No videos found in: {video_dir}")
        return 0

    total_vids = len(vids)
    for vid_idx, video_file in enumerate(vids, start=1):
        video_name = video_file.stem

        ui_status(f"Extracting frames: {video_file.name}")
        ui_log(f"[EXTRACT] {video_file.name}")

        cap = cv2.VideoCapture(str(video_file))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not fps or fps <= 0:
            ui_log(f"[ERROR] Could not read FPS from {video_file.name}; skipping.")
            cap.release()
            continue

        step_frames = max(1, int(round(fps * float(interval_seconds))))
        frame_idx = 0
        saved_idx = 0

        ui_sub_progress(0, indeterminate=False)
        last_update = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step_frames == 0:
                frame_path = output_base_dir / f"{video_name}_frame_{saved_idx:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_idx += 1

            frame_idx += 1

            if frame_count > 0 and (time.time() - last_update) > 0.05:
                ui_sub_progress(min(100.0, 100.0 * frame_idx / max(1, frame_count)), indeterminate=False)
                last_update = time.time()

        cap.release()
        ui_sub_progress(100.0, indeterminate=False)
        ui_log(f"[OK] Saved {saved_idx} frames into {output_base_dir}")

        ui_main_progress(min(100.0, 100.0 * vid_idx / total_vids), indeterminate=False)

    return total_vids

# =========================
# NEW: YOLO segmentation-based masking
# =========================

def run_yolo_masking(frames_root: Path) -> int:
    """
    EXACT YOLO settings as mask_yolo.py, but overwrites the original JPGs in frames/:
      - model: yolov8x-seg.pt
      - conf: 0.35, iou: 0.45, imgsz: 1024
      - classes: [0] (person)
      - remove_person=True, grow_person_px=30
    Output: in-place .jpg overwrite in frames/
    """
    try:
        from ultralytics import YOLO
        import numpy as np
        import cv2
        import torch
        import time
    except Exception as e:
        ui_log("[ERROR] YOLO/torch/OpenCV not available.")
        ui_log("       pip install ultralytics opencv-python torch torchvision torchaudio")
        return 0

    # --- Fixed parameters (mirroring mask_yolo.py) ---
    model_name = "yolov8x-seg.pt"
    conf = 0.35
    iou = 0.45
    imgsz = 1024
    classes = [0]           # person
    remove_person = True
    grow_person_px = 30     # expand the person region before removal

    def load_bgr(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    def morph_expand(mask_bool: np.ndarray, grow_px: int) -> np.ndarray:
        """Grow (dilate) or shrink (erode) a boolean mask in pixel units."""
        if grow_px == 0:
            return mask_bool
        mask = (mask_bool.astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # ~3px per iteration; mirror your script’s iterative growth behavior
        step = 3
        iters = max(1, int(abs(grow_px) / step))
        out = cv2.dilate(mask, k, iterations=iters) if grow_px > 0 else cv2.erode(mask, k, iterations=iters)
        return (out > 0)

    img_paths = sorted(frames_root.glob("*.jpg"))
    if not img_paths:
        ui_log(f"[WARN] No .jpg frames found in {frames_root}")
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ui_log(f"[YOLO] Loading {model_name} on {device} …")
    model = YOLO(model_name)  # auto-download if missing

    total = len(img_paths)
    processed = 0
    ui_status("Masking User")
    ui_sub_progress(0, indeterminate=False)

    last_update = time.time()

    for i, img_path in enumerate(img_paths, 1):
        try:
            bgr = load_bgr(img_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # Predict with fixed settings (avoids imgsz=None error)
            results = model.predict(
                source=rgb,       # ndarray -> avoids loader path handling
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
                classes=classes,
                device=None       # let ultralytics choose (cuda if available)
            )

            # Default: if no person found, keep image as-is
            keep_mask = np.ones((h, w), dtype=bool)

            if results and results[0].masks is not None and len(results[0].masks) > 0:
                masks_tensor = results[0].masks.data.cpu().numpy()   # (N, Hm, Wm) float [0,1]
                person_small = (masks_tensor.max(axis=0) > 0.5).astype(np.uint8)
                person = cv2.resize(person_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

                # Grow person region to be safe
                person = morph_expand(person, grow_person_px)

                # We remove the person → keep everything else
                if remove_person:
                    keep_mask = ~person
                else:
                    keep_mask = person

            # Apply in-place: black-out pixels where keep_mask is False
            # (JPEG can’t store alpha; this mimics transparency by zeroing)
            out_rgb = rgb.copy()
            out_rgb[~keep_mask] = 0  # black-out removed area
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            processed += 1

        except Exception as e:
            ui_log(f"[ERROR] {img_path.name}: {e}")

        if (time.time() - last_update) > 0.05:
            ui_sub_progress(min(100.0, 100.0 * i / total), indeterminate=False)
            last_update = time.time()

    ui_sub_progress(100.0, indeterminate=False)
    ui_log(f"[OK] In-place masking complete: {processed}/{total} (overwrote JPGs in {frames_root})")
    return processed


# =========================
# External steps (threaded helpers)
# =========================

def run_panorama_sfm(project_root: Path) -> bool:
    wrapper = Path(__file__).parent / "run_panorama_sfm.py"
    if not wrapper.exists():
        ui_log(f"[ERROR] Missing run_panorama_sfm.py next to the GUI: {wrapper}")
        return False

    cmd = [sys.executable, str(wrapper), str(project_root)]
    ui_log(f"[RUN] {' '.join(cmd)}")
    ui_status("Running COLMAP pipeline…")

    # Spin sub-progress as an activity indicator
    ui_sub_progress(indeterminate=True)

    try:
        import subprocess  # <-- keep only subprocess here

        # Stream logs live into the GUI
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        ) as proc:
            if proc.stdout is not None:
                for line in proc.stdout:
                    ui_log(line.rstrip())

            ret = proc.wait()

        ui_sub_progress(100, indeterminate=False)

        if ret == 0:
            ui_log("[OK] panorama_sfm finished.")
            return True
        else:
            ui_log(f"[ERROR] panorama_sfm exited with code {ret}.")
            return False

    except Exception as e:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[ERROR] panorama_sfm failed: {e}")
        return False



def delete_pano_camera0(project_root: Path) -> None:
    deleter = Path(__file__).parent / "delete_pano0.py"
    if not deleter.exists():
        ui_log(f"[WARN] Missing delete_pano0.py (skipping): {deleter}")
        return

    cmd = [sys.executable, str(deleter), str(project_root)]
    ui_log(f"[RUN] {' '.join(cmd)}")
    ui_status("Deleting pano_camera0 folders…")
    ui_sub_progress(indeterminate=True)
    try:
        subprocess.run(cmd, check=True)
        ui_sub_progress(100, indeterminate=False)
        ui_log("[OK] Deleted all pano_camera0 folders.")
    except subprocess.CalledProcessError as e:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[WARN] delete_pano0.py reported an error: {e}")

def run_segment_images(project_root: Path) -> bool:
    """Run segment_images.py at the end, only needed when multiple input videos were used."""
    seg = Path(__file__).parent / "segment_images.py"
    if not seg.exists():
        ui_log(f"[WARN] Missing segment_images.py (skipping): {seg}")
        return False

    cmd = [sys.executable, str(seg)]
    ui_log(f"[RUN] {' '.join(cmd)} (cwd={project_root})")
    ui_status("Segmenting COLMAP images by clip prefix…")
    ui_sub_progress(indeterminate=True)
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        ui_sub_progress(100, indeterminate=False)
        ui_log("[OK] Segmented images (see Segment_images/).")
        return True
    except subprocess.CalledProcessError as e:
        ui_sub_progress(0, indeterminate=False)
        ui_log(f"[WARN] segment_images.py reported an error: {e}")
        return False

# =========================
# End-to-end pipeline (threaded)
# =========================

def pipeline_thread(project_root: Path, seconds_per_frame: float, masking_enabled: bool):
    try:
        ui_disable_inputs(True)

        # Phase A: Extraction
        ui_main_progress(0, indeterminate=False)
        ui_status("Preparing extraction…")
        video_count = extract_frames_with_progress(project_root, seconds_per_frame)
        frames_root = project_root / "frames"

        # Phase B: Angles & optional masking (no UI variables)
        if masking_enabled:
            ui_status("Writing rotation override (masking)…")
            write_rotation_override(project_root, MASKING_PITCH_YAW_PAIRS, MASKING_REF_IDX)
            ui_log(f"[OK] Wrote rotation_override.json (masking) in {project_root}")

            ui_status("Masking frames (fixed YOLO settings)…")
            processed = run_yolo_masking(frames_root=frames_root)
            if processed == 0:
                ui_log("[WARN] No frames were masked (or masking skipped due to error). Continuing…")
        else:
            ui_status("Writing rotation override (no masking)…")
            write_rotation_override(project_root, NO_MASKING_PITCH_YAW_PAIRS, NO_MASKING_REF_IDX)
            ui_log(f"[OK] Wrote rotation_override.json (no masking) in {project_root}")

        ui_main_progress(33, indeterminate=False)

        # Phase C: COLMAP (stream logs into GUI)
        if not run_panorama_sfm(project_root):
            ui_status("COLMAP failed. See log.")
            return

        ui_main_progress(66, indeterminate=False)

        # Phase D: Delete pano_camera0
        delete_pano_camera0(project_root)
        ui_main_progress(90, indeterminate=False)

        # Phase E: Segment images if more than one video
        if video_count > 1:
            ui_log(f"[INFO] Multiple videos detected ({video_count}). Running segment_images…")
            run_segment_images(project_root)

        ui_main_progress(100, indeterminate=False)
        ui_status("All done.")
        ui_log("[DONE] Pipeline complete.")

    finally:
        try:
            progress_main.stop()
            progress_sub.stop()
        except Exception:
            pass
        ui_disable_inputs(False)


# =========================
# GUI
# =========================

_last_folder = None  # remember last path for re-run

def start_pipeline_with_path(folder_path: Path):
    global _last_folder
    _last_folder = folder_path

    try:
        seconds = float(frame_interval.get())
        if seconds <= 0:
            raise ValueError
    except Exception:
        seconds = 1.0
        frame_interval.set("1")

    threading.Thread(
        target=pipeline_thread,
        args=(folder_path, seconds, use_masking.get()),
        daemon=True
    ).start()

def on_drop(event):
    folder_path = Path(event.data.strip("{}"))
    if folder_path.is_dir():
        run_btn.configure(state=NORMAL)  # NEW: enable re-run
        start_pipeline_with_path(folder_path)
    else:
        ui_log("[ERROR] Please drop a valid folder.")

def browse_folder():
    chosen = filedialog.askdirectory(title="Select folder containing 360 video(s)")
    if chosen:
        run_btn.configure(state=NORMAL)  # NEW: enable re-run
        start_pipeline_with_path(Path(chosen))

def run_last():
    if _last_folder and Path(_last_folder).exists():
        start_pipeline_with_path(Path(_last_folder))
    else:
        ui_log("[ERROR] No valid last folder. Please Browse or Drop a folder.")

def on_masking_toggle():
    pass


def main():
    global _root, status_var, progress_main, progress_sub, log_text
    global use_masking, frame_interval, drop_zone, run_btn, browse_btn
    global yolo_model_path, yolo_conf, yolo_dilate_px, yolo_invert_mask, yolo_apply_to_rgb, _yolo_widgets

    _root = TkinterDnD.Tk()
    _root.title("360 Video Dataset Preparation")
    _root.geometry("780x720")
    _root.configure(bg="black")
    _root.resizable(False, False)

    # ---------- Header ----------
    header = Frame(_root, bg="black")
    header.pack(pady=(14, 8))

    icon_path = Path(__file__).parent / "folder_icon.png"
    if icon_path.exists():
        try:
            img = Image.open(icon_path).resize((84, 84))
            icon = ImageTk.PhotoImage(img)
            icon_label = Label(header, image=icon, bg="black")
            icon_label.image = icon
            icon_label.pack(side="left", padx=(0, 12))
        except Exception:
            Label(header, text="📁", font=("Arial", 44), bg="black", fg="white").pack(side="left", padx=(0, 12))
    else:
        Label(header, text="📁", font=("Arial", 44), bg="black", fg="white").pack(side="left", padx=(0, 12))

    Label(header,
          text="Insta360 Video(s) To Training Format Pipeline",
          bg="black", fg="white", font=("Helvetica", 16, "bold")).pack(side="left")

    # ---------- Controls ----------
    ctrl = Frame(_root, bg="black")
    ctrl.pack(pady=(8, 2))

    Label(ctrl, text="Extract 1 frame per", bg="black", fg="white").pack(side="left")

    frame_interval = StringVar(value="1")
    Entry(ctrl, textvariable=frame_interval, width=6).pack(side="left", padx=(6, 6))

    Label(ctrl, text="seconds", bg="black", fg="white").pack(side="left", padx=(0, 16))

    use_masking = BooleanVar(value=False)
    mcb = Checkbutton(
        ctrl,
        text="Enable Masking (removes user from frames)",
        variable=use_masking,
        onvalue=True, offvalue=False,
        bg="black", fg="white", activebackground="black",
        selectcolor="black",
    )
    mcb.pack(side="left")

    # ---------- Drop zone ----------
    drop_zone = Label(
        _root, text="Drop Folder With Insta360 Videos Here",
        bg="#222", fg="white", width=70, height=6,
        relief="ridge", bd=2
    )
    drop_zone.pack(pady=10)
    drop_zone.drop_target_register(DND_FILES)
    drop_zone.dnd_bind('<<Drop>>', on_drop)

    # ---------- Buttons ----------
    btns = Frame(_root, bg="black")
    btns.pack()
    browse_btn = Button(btns, text="Browse…", command=browse_folder)
    browse_btn.pack(side="left", padx=6)
    run_btn = Button(btns, text="Run Last Chosen Folder", state=DISABLED, command=run_last)
    run_btn.pack(side="left", padx=6)

    # ---------- Progress ----------
    prog = Frame(_root, bg="black")
    prog.pack(fill=BOTH, padx=16, pady=(12, 4))

    Label(prog, text="Overall Progress", bg="black", fg="#ccc").pack(anchor="w")
    progress_main = ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=680)
    progress_main.pack(pady=(2, 8))

    Label(prog, text="Current Task", bg="black", fg="#ccc").pack(anchor="w")
    progress_sub = ttk.Progressbar(prog, orient="horizontal", mode="determinate", length=680)
    progress_sub.pack(pady=(2, 8))

    status_var = StringVar(value="Idle.")
    Label(_root, textvariable=status_var, bg="black", fg="white").pack(pady=(0, 6))

    # ---------- Log ----------
    log_frame = Frame(_root, bg="black")
    log_frame.pack(fill=BOTH, expand=True, padx=16, pady=(0, 12))
    log_text = Text(log_frame, height=12, bg="#111", fg="#ddd", insertbackground="white")
    log_text.pack(fill=BOTH, expand=True)
    log_text.configure(state=DISABLED)

    # initialize YOLO controls state
    on_masking_toggle()

    _root.mainloop()

if __name__ == "__main__":
    main()
