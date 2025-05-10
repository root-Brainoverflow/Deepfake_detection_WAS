import subprocess
import pandas as pd
from pathlib import Path
from zipfile import ZipFile

def extract_and_split_frames(
    zip_path: Path,
    label_map: dict,
    output_base: Path,
    fps: float = 0.5,
    width: int = 224,
    height: int = 224,
    qscale: int = 5
):
    """
    1) unzip zip_path → extract_dir/
    2) extract_dir/**/*.mp4 마다 video_id = mp4.stem 으로 라벨 조회
       → frames/<train|validate>/{real, fake}/{zip_stem}/{video_id}/frame_XXXX.jpg
    """
    zip_stem = zip_path.stem
    extract_dir = Path("extracted") / zip_stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[UNZIP] {zip_path} → {extract_dir}")
    subprocess.run(
        ["unzip", "-oq", str(zip_path), "-d", str(extract_dir)],
        check=True
    )

    for mp4 in extract_dir.rglob("*.mp4"):
        video_id = mp4.stem
        label = label_map.get(video_id)
        if label is None:
            print(f"[SKIP] no label for {mp4.name}")
            continue

        split = output_base.name  # output_base = Path("frames/train") or ("frames/validate")
        category = "real" if label == 0 else "fake"
        out_dir = output_base / category / zip_stem / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-i", str(mp4),
            "-vf", f"fps={fps},scale={width}:{height}",
            "-qscale:v", str(qscale),
            str(out_dir / "frame_%04d.jpg")
        ]
        print(f"[FRAME] {mp4.name} → {out_dir}/")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mp4.unlink()

    print(f"[DONE] frames saved under {output_base}\n")


if __name__ == "__main__":
    # ────────────────────────────────
    # 1) 메타 CSV 로드 (stem 기준 키 생성)
    # ────────────────────────────────
    meta_dir = Path("meta")

    df_train = pd.read_csv(meta_dir / "train_meta_full.csv", encoding="utf-8-sig")
    label_map_train = {
        Path(r["영상ID"]).stem: r["라벨"]
        for _, r in df_train.iterrows()
    }

    df_val = pd.read_csv(meta_dir / "validate_meta_full.csv", encoding="utf-8-sig")
    label_map_val = {
        Path(r["영상ID"]).stem: r["라벨"]
        for _, r in df_val.iterrows()
    }

    # ────────────────────────────────
    # 2) ZIP 파일 경로 수집
    # ────────────────────────────────
    train_dirs = [
        Path("딥페이크 변조 영상/1.Training"),
        Path("003.딥페이크/1.Training/원천데이터/train_원본")
    ]
    val_dirs = [
        Path("003.딥페이크/1.Training/원천데이터/validate_변조"),
        Path("003.딥페이크/1.Training/원천데이터/validate_원본")
    ]

    train_zips = []
    for d in train_dirs:
        train_zips += list(d.rglob("*.zip"))

    val_zips = []
    for d in val_dirs:
        val_zips += list(d.rglob("*.zip"))

    # ────────────────────────────────
    # 3) 순차 처리
    # ────────────────────────────────
    print("=== TRAINING SET PROCESSING ===")
    for zp in train_zips:
        extract_and_split_frames(
            zip_path=zp,
            label_map=label_map_train,
            output_base=Path("frames/train")
        )

    print("=== VALIDATION SET PROCESSING ===")
    for zp in val_zips:
        extract_and_split_frames(
            zip_path=zp,
            label_map=label_map_val,
            output_base=Path("frames/validate")
        )