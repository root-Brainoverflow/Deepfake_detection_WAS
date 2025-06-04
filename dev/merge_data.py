import shutil
from pathlib import Path

# 원본 폴더들
source_dirs = ["real"]

# 대상 폴더
target_base = Path("frames/train/real")
target_male = target_base / "남성"
target_female = target_base / "여성"

target_male.mkdir(parents=True, exist_ok=True)
target_female.mkdir(parents=True, exist_ok=True)

for src_name in source_dirs:
    src_path = Path(src_name)
    for gender in ["남성", "여성"]:
        src_gender_dir = src_path / gender
        if not src_gender_dir.exists():
            print(f"[SKIP] {src_gender_dir} 존재하지 않음")
            continue

        for jpg_file in src_gender_dir.glob("*.jpg"):
            dst = (target_male if gender == "남성" else target_female) / jpg_file.name

            # 중복되면 복사하지 않음
            if dst.exists():
                print(f"[SKIP] 중복 파일: {dst.name}")
                continue

            shutil.copy2(jpg_file, dst)
            print(f"Copied: {jpg_file} → {dst}")

print("\n🎉 모든 복사 완료!")