from pathlib import Path

def get_folder_info(path_str):
    path = Path(path_str)
    total_size = 0
    total_files = 0

    for f in path.rglob("*"):
        if f.is_file():
            total_files += 1
            total_size += f.stat().st_size  # 바이트 단위

    size_mb = total_size / (1024 ** 2)
    size_gb = total_size / (1024 ** 3)

    print(f"\n 경로: {path.resolve()}")
    print(f"총 파일 개수: {total_files}개")
    print(f"총 용량: {size_mb:.2f} MB ({size_gb:.2f} GB)")

get_folder_info("frames")