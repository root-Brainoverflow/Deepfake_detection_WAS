# ────────────────────────────
# 3) 프레임 추출 함수 (video_id 레벨 디렉토리 제거)
# ────────────────────────────
import subprocess
from pathlib import Path

def extract_and_split_frames(zip_path: Path,
                             label_map: dict,
                             gender_map: dict,
                             output_base: Path,
                             fps: float  = 0.5,
                             size: str   = "224:224",
                             qscale: int = 5):
    """
    1) Unzip zip_path into extracted/<zip_stem>/
    2) For each .mp4 inside, determine original key (orig_key)
       then look up label_map[orig_key] and gender_map[orig_key].
    3) Extract frames at fps, resize to size, save under
       output_base/{real|fake}/{gender}/<orig_filename>_frame_####.jpg
    """
    stem     = zip_path.stem
    work_dir = Path("extracted") / stem
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[UNZIP] {zip_path.name} → {work_dir}")
    subprocess.run(
        ["unzip", "-oq", str(zip_path), "-d", str(work_dir)],
        check=True
    )

    for mp4 in work_dir.rglob("*.mp4"):
        vid_full = mp4.stem                # e.g. "179032_175277_2_0270"
        parts    = vid_full.split("_")

        # determine key to lookup metadata
        if len(parts) == 4:
            orig_key = f"{parts[0]}_{parts[3][:3]}"  # e.g. "179032_027"
        else:
            orig_key = vid_full                     # original videos

        label  = label_map.get(orig_key)
        gender = gender_map.get(orig_key)
        if label is None or gender not in ("남성", "여성"):
            print(f"[SKIP] {mp4.name}: missing metadata for key '{orig_key}'")
            continue

        category = "real" if label == 0 else "fake"
        out_dir  = output_base / category / gender
        out_dir.mkdir(parents=True, exist_ok=True)

        pattern = f"{vid_full}_frame_%04d.jpg"
        print(f"[FRAME] {mp4.name} → {out_dir}/{pattern}")
        subprocess.run([
            "ffmpeg", "-i", str(mp4),
            "-vf", f"fps={fps},scale={size}",
            "-qscale:v", str(qscale),
            str(out_dir / pattern)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        mp4.unlink()

    print(f"[DONE] {stem} → frames saved under {output_base}/{category}/{gender}\n")

    
# ────────────────────────────
# 4) ZIP 경로 수집 & 실행
# ────────────────────────────
train_dirs = [
    Path("딥페이크 변조 영상/1.Training"),
    Path("003.딥페이크/1.Training/원천데이터/train_원본")
]
val_dirs = [
    Path("003.딥페이크/1.Training/원천데이터/validate_변조"),
    Path("003.딥페이크/1.Training/원천데이터/validate_원본")
]

train_zips = [p for d in train_dirs for p in d.rglob("*.zip")]
val_zips   = [p for d in val_dirs   for p in d.rglob("*.zip")]

print("=== TRAIN SET ===")
for zp in train_zips:
    extract_and_split_frames(
        zip_path    = zp,
        label_map   = label_map_train,
        gender_map  = gender_map_train,
        output_base = Path("frames/train")
    )

print("=== VALIDATE SET ===")
for zp in val_zips:
    extract_and_split_frames(
        zip_path    = zp,
        label_map   = label_map_val,
        gender_map  = gender_map_val,
        output_base = Path("frames/validate")
    )