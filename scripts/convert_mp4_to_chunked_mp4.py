import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def ensure_videos_ori(videos_dir: Path):
    """
    确保 videos 目录被重命名为 videos_ori，新建 videos 用于转码输出。
    """
    videos_ori = videos_dir.parent / "videos_ori"
    if not videos_ori.exists():
        videos_dir.rename(videos_ori)
    else:
        # 已经有 videos_ori 说明 videos 应该是新目录，但也可能用户意外创建了
        if videos_dir.exists() and any(videos_dir.iterdir()):
            raise RuntimeError(f"Conflict: both {videos_dir} and {videos_ori} exist and are not empty.")

    videos_dir.mkdir(exist_ok=True)  # 创建 videos 目录用于新转码输出
    return videos_ori

def is_valid_mp4(file_path: Path) -> bool:
    """
    用 ffprobe 检查 mp4 文件是否损坏。
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # 只要能解析出 duration 就算有效
        return result.returncode == 0 and result.stdout and float(result.stdout) > 0
    except Exception:
        return False

def transcode_to_chunked_mp4(input_video: Path, input_root: Path, output_root: Path, crf: int = 23, preset: str = "fast"):
    """
    转码 input_video 成 Chunked MP4，保存到 output_root 对应路径，只输出失败信息。
    """
    relative_path = input_video.relative_to(input_root)
    output_video = output_root / relative_path
    output_video.parent.mkdir(parents=True, exist_ok=True)

    # 如果输出文件存在且可用，跳过
    if output_video.exists():
        if is_valid_mp4(output_video):
            return
        else:
            output_video.unlink()  # 删除损坏的

    temp_output = output_video.with_suffix(".temp.mp4")

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_video),
            "-c:v", "libx264",
            "-g", "1", "-keyint_min", "1", "-sc_threshold", "0",
            "-crf", str(crf), "-preset", preset, "-an",
            str(temp_output)
        ]
        with open(os.devnull, 'wb') as devnull:
            subprocess.run(cmd, stdout=devnull, stderr=devnull, check=True)
        temp_output.rename(output_video)
    except Exception:
        print(f"❌ 转码失败: {input_video}", flush=True)
        if temp_output.exists():
            temp_output.unlink()
        if output_video.exists():
            output_video.unlink()

def batch_transcode_videos(videos_ori: Path, videos_new: Path, workers: int = 8, crf: int = 23, preset: str = "fast"):
    """
    批量转码 videos_ori 下所有 mp4 视频，输出到 videos_new 目录。
    """
    # 自动搜寻所有 mp4
    video_files = list(videos_ori.glob("chunk-*/images_dict.*.*/*.mp4"))

    if not video_files:
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(transcode_to_chunked_mp4, video_file, videos_ori, videos_new, crf, preset)
            for video_file in video_files
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Transcoding {videos_ori.parent}"):
            pass

def traverse_and_transcode(base_path: str, workers: int = 8, crf: int = 23, preset: str = "fast"):
    """
    遍历 base_path，批量处理所有 'videos' 文件夹。
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Path not found: {base_path}")

    # 递归查找所有 videos 文件夹（不限层级）
    videos_dirs = list(base_path.rglob("videos"))
    if not videos_dirs:
        print(f"[Error] No 'videos' folders found under {base_path}")
        return

    for videos_dir in videos_dirs:
        if not videos_dir.is_dir():
            continue
        try:
            videos_ori = ensure_videos_ori(videos_dir)
        except Exception as e:
            print(f"[Error] Skipping {videos_dir}: {e}")
            continue
        batch_transcode_videos(videos_ori, videos_dir, workers=workers, crf=crf, preset=preset)

from pathlib import Path

def remove_iframe_videos(root_dir):
    """
    递归删除 root_dir 下所有以 _iframe.mp4 结尾的文件。
    """
    root_dir = Path(root_dir)
    count = 0
    for file in root_dir.rglob("*_iframe.mp4"):
        if file.is_file():
            try:
                file.unlink()
                print(f"已删除: {file}")
                count += 1
            except Exception as e:
                print(f"无法删除: {file}  原因: {e}")
    print(f"共删除 {count} 个 _iframe.mp4 文件。")





if __name__ == "__main__":
    # import sys

    # if len(sys.argv) < 2:
    #     print("Usage: python convert_mp4_to_chunked_mp4.py <dataset_or_base_path> [num_workers] [crf] [preset]")
    #     exit(1)

    # input_path = sys.argv[1]
    # num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    # crf = int(sys.argv[3]) if len(sys.argv) > 3 else 23
    # preset = sys.argv[4] if len(sys.argv) > 4 else "fast"

    # traverse_and_transcode(input_path, workers=num_workers, crf=crf, preset=preset)


    import sys
    if len(sys.argv) < 2:
        print("Usage: python del_iframe.py <root_path>")
        exit(1)
    remove_iframe_videos(sys.argv[1])