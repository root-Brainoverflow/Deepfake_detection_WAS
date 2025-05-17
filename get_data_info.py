from pathlib import Path
from collections import Counter
import pandas as pd

# 1) 프레임 루트 경로
root = Path("frames")          # frames/train/…, frames/validate/… 가 모두 들어 있는 상위 폴더

# 2) 모든 jpg 파일을 순회하며 video_id 추출
cnt = Counter()
for jpg in root.rglob("*.jpg"):
    # 파일명 예시: 171347_176210_5_1030_frame_0001.jpg
    video_id = jpg.stem.rsplit("_frame_", 1)[0]
    cnt[video_id] += 1

# 3) Counter → DataFrame 변환
df = pd.DataFrame(cnt.items(), columns=["video_id", "num_frames"])

# 4) 전체 통계
total_videos   = len(df)
total_frames   = df["num_frames"].sum()
min_frames     = df["num_frames"].min()
max_frames     = df["num_frames"].max()
mean_frames    = df["num_frames"].mean()

print(f"전체 영상 수          : {total_videos:,d}")
print(f"전체 프레임 수        : {total_frames:,d}")
print(f"영상 1개당 프레임 수(평균): {mean_frames:.2f}")
print(f" └ 최소 {min_frames} ~ 최대 {max_frames} 장")