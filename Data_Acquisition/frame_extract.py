# ────────────────────────────
# 3) 프레임 추출 함수 (video_id 레벨 디렉토리 제거)
# ────────────────────────────
def extract_and_split_frames(zip_path: Path,
                             label_map: dict,
                             gender_map: dict,
                             output_base: Path,
                             fps: float  = 0.5,
                             size: str   = "224:224",
                             qscale: int = 5):
    stem = zip_path.stem
    work_dir = Path("extracted")/stem
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[UNZIP] {zip_path.name} → {work_dir}")
    subprocess.run(
        ["unzip", "-oq", str(zip_path), "-d", str(work_dir)],
        check=True
    )

    for mp4 in work_dir.rglob("*.mp4"):
        vid    = mp4.stem
        label  = label_map .get(vid)
        gender = gender_map.get(vid)
        if label is None or gender not in ("남자","여자"):
            print(f"[SKIP] {mp4.name}: meta missing")
            continue

        cat     = "real" if label == 0 else "fake"
        # video_id 디렉토리 없이 바로 gender 폴더
        out_dir = output_base/cat/gender
        out_dir.mkdir(parents=True, exist_ok=True)

        # 프레임 파일명에 video_id 접두어 추가
        pattern = f"{vid}_frame_%04d.jpg"
        print(f"[FRAME] {mp4.name} → {out_dir}/{pattern}")
        subprocess.run([
            "ffmpeg", "-i", str(mp4),
            "-vf", f"fps={fps},scale={size}",
            "-qscale:v", str(qscale),
            str(out_dir/pattern)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        mp4.unlink()

    print(f"[DONE] {stem} → frames in {output_base}\n")


# ────────────────────────────
# 4) ZIP 경로 수집 & 실행
# ────────────────────────────
train_dirs = [ Path("딥페이크 변조 영상/1.Training"),
               Path("003.딥페이크/1.Training/원천데이터/train_원본") ]
val_dirs   = [ Path("003.딥페이크/1.Training/원천데이터/validate_변조"),
               Path("003.딥페이크/1.Training/원천데이터/validate_원본") ]

train_zips = sum((list(d.rglob("*.zip")) for d in train_dirs), [])
val_zips   = sum((list(d.rglob("*.zip")) for d in val_dirs),   [])

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