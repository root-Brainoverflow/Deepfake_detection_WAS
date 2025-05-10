import pandas as pd
import subprocess
from pathlib import Path
from zipfile import ZipFile

# ────────────────────────────
# 1) 메타파일 경로 준비 및 full 메타 생성
# ────────────────────────────
meta_dir    = Path("meta")
real_csv    = meta_dir / "원본영상_training_메타데이터.csv"
fake_csv    = meta_dir / "변조영상_training_메타데이터.csv"
real_val    = meta_dir / "원본영상_validation_메타데이터.csv"
fake_val    = meta_dir / "변조영상_validation_메타데이터.csv"

# 학습용 메타 full 생성
# 컬럼: 영상ID, 인물성별, 라벨
cols = ["영상ID", "인물성별"]
df_real    = pd.read_csv(real_csv, encoding="utf-8")[cols].copy()
df_real["라벨"] = 0
df_fake    = pd.read_csv(fake_csv, encoding="utf-8")[cols].copy()
df_fake["라벨"] = 1
df_train_full = pd.concat([df_real, df_fake], ignore_index=True)
out_train = meta_dir / "train_meta_full.csv"
df_train_full.to_csv(out_train, index=False, encoding="utf-8-sig")
print(f"[✓] 생성 완료: {out_train} (총 {len(df_train_full)}개)")

# 검증용 메타 full 생성
# 컬럼: 영상ID, 인물성별, 라벨
df_real_v  = pd.read_csv(real_val, encoding="utf-8")[cols].copy()
df_real_v["라벨"] = 0
df_fake_v  = pd.read_csv(fake_val, encoding="utf-8")[cols].copy()
df_fake_v["라벨"] = 1
df_val_full = pd.concat([df_real_v, df_fake_v], ignore_index=True)
out_val = meta_dir / "validate_meta_full.csv"
df_val_full.to_csv(out_val, index=False, encoding="utf-8-sig")
print(f"[✓] 생성 완료: {out_val} (총 {len(df_val_full)}개)")

# ────────────────────────────
# 2) 메타데이터 로드 및 라벨 맵 생성
# ────────────────────────────
train_meta_path    = out_train
validate_meta_path = out_val
df_train_meta = pd.read_csv(train_meta_path, encoding="utf-8-sig")
label_map_train = {row['영상ID']: row['라벨'] for _, row in df_train_meta.iterrows()}
df_val_meta   = pd.read_csv(validate_meta_path, encoding="utf-8-sig")
label_map_val = {row['영상ID']: row['라벨'] for _, row in df_val_meta.iterrows()}
