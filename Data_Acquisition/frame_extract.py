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
    zip_path 을 압축 해제한 뒤,
    각 .mp4 파일 이름(영상ID)으로 label_map에서 0(real)/1(fake) 조회
    → frames/train/{real|fake}/{zip_stem}/{video_stem}/frame_XXXX.jpg 에 저장
    """
    zip_stem = zip_path.stem
    # 1) 압축 해제
    extract_dir = Path("extracted_videos") / zip_stem
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[UNZIP] {zip_path} → {extract_dir}")
    subprocess.run(
        ["unzip", "-oq", str(zip_path), "-d", str(extract_dir)],
        check=True
    )
    print("[UNZIP] complete")

    # 2) mp4 순회하면서 프레임 추출
    for mp4 in extract_dir.rglob("*.mp4"):
        vid = mp4.name           # e.g. "179032_027.mp4"
        video_id = mp4.stem      # "179032_027"
        label = label_map.get(video_id)
        if label is None:
            print(f"[SKIP] no label for {vid}")
            continue

        subdir = "real" if label == 0 else "fake"
        out_dir = output_base / subdir / zip_stem / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-i", str(mp4),
            "-vf", f"fps={fps},scale={width}:{height}",
            "-qscale:v", str(qscale),
            str(out_dir / "frame_%04d.jpg")
        ]
        print(f"[FRAME] {vid} → {out_dir}/")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mp4.unlink()  # 원본 mp4 삭제

    print(f"[DONE] frames saved under {output_base}\n")


if __name__ == "__main__":
    # 1) 메타 CSV 로드
    meta_dir = Path("meta")
    df_train = pd.read_csv(meta_dir / "train_meta_full.csv", encoding="utf-8-sig")
    label_map_train = {row["영상ID"]: row["라벨"] for _, row in df_train.iterrows()}

    df_val = pd.read_csv(meta_dir / "validate_meta_full.csv", encoding="utf-8-sig")
    label_map_val = {row["영상ID"]: row["라벨"] for _, row in df_val.iterrows()}

    # 2) ZIP 파일 리스트(폴더) 정의
    train_zips = list(Path("003.딥페이크/1.Training/원천데이터/train_변조").glob("*.zip")) \
               + list(Path("003.딥페이크/1.Training/원천데이터/train_원본").glob("*.zip"))
    val_zips   = list(Path("003.딥페이크/1.Training/원천데이터/validate_변조").glob("*.zip")) \
               + list(Path("003.딥페이크/1.Training/원천데이터/validate_원본").glob("*.zip"))

    # 3) 순차 처리
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