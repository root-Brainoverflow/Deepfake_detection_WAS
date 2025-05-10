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
    stem     = zip_path.stem
    work_dir = Path("extracted") / stem
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[UNZIP] {zip_path.name} → {work_dir}")
    subprocess.run(
        ["unzip", "-oq", str(zip_path), "-d", str(work_dir)],
        check=True
    )

    for mp4 in work_dir.rglob("*.mp4"):
        vid_full = mp4.stem  # e.g. "171347_176210_5_1030"
        label    = label_map .get(vid_full)
        gender   = gender_map.get(vid_full)

        if label is None or gender not in ("남성", "여성"):
            print(f"[SKIP] {mp4.name}: missing metadata for key '{vid_full}'")
            continue

        category = "real" if label == 0 else "fake"
        out_dir  = output_base / category / gender
        out_dir.mkdir(parents=True, exist_ok=True)

        # 이미 처리된 프레임이 있으면 스킵
        first_frame = out_dir / f"{vid_full}_frame_0001.jpg"
        if first_frame.exists():
            print(f"[SKIP] frames already exist for {vid_full}")
            mp4.unlink()
            continue

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