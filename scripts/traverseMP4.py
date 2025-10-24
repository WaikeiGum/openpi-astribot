# import os
# from pathlib import Path
# import numpy as np
# import cv2
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed

# def video_to_array(video_path: Path) -> np.ndarray:
#     cap = cv2.VideoCapture(str(video_path))
#     frames = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = frame.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
#         frame = frame.astype(np.float32) / 255.0  # Normalize
#         frames.append(frame)

#     cap.release()
#     frames = np.array(frames)  # (num_frames, C, H, W)
#     return frames

# def save_array(frames: np.ndarray, np_file: Path, save_as_npz: bool):
#     np_file.parent.mkdir(parents=True, exist_ok=True)
#     if save_as_npz:
#         np.savez_compressed(np_file, frames=frames)
#     else:
#         np.save(np_file, frames)

# def process_video(video_file: Path, videos_dir: Path, np_dir: Path, save_as_npz: bool):
#     relative_path = video_file.relative_to(videos_dir)
#     np_file = np_dir / relative_path.with_suffix(".npz" if save_as_npz else ".npy")

#     if np_file.exists():
#         return  # Already exists

#     try:
#         frames = video_to_array(video_file)
#         save_array(frames, np_file, save_as_npz)
#     except Exception as e:
#         print(f"[Error] Failed to process {video_file}: {e}")

# def batch_convert_videos_in_folder(videos_dir: Path, workers: int = 8, save_as_npz: bool = True):
#     np_dir = videos_dir.parent / ("npz" if save_as_npz else "npy")
#     video_files = list(videos_dir.glob("chunk-*/images_dict.*.*/*.mp4"))

#     if not video_files:
#         print(f"[Skip] No mp4 files found in {videos_dir}")
#         return

#     print(f"[Info] Found {len(video_files)} videos in {videos_dir}")

#     with ProcessPoolExecutor(max_workers=workers) as executor:
#         futures = [
#             executor.submit(process_video, video_file, videos_dir, np_dir, save_as_npz)
#             for video_file in video_files
#         ]

#         for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Converting {videos_dir.parent.name}"):
#             pass

# def traverse_and_convert(base_path: str, workers: int = 8, save_as_npz: bool = True):
#     base_path = Path(base_path)

#     if not base_path.exists():
#         raise FileNotFoundError(f"Path does not exist: {base_path}")

#     if (base_path / "videos").exists():
#         print(f"[Info] Single dataset detected: {base_path}")
#         batch_convert_videos_in_folder(base_path / "videos", workers=workers, save_as_npz=save_as_npz)
#     else:
#         videos_dirs = list(base_path.glob("*/videos"))

#         if not videos_dirs:
#             print(f"[Error] No videos/ folders found under {base_path}")
#             return

#         for videos_dir in videos_dirs:
#             batch_convert_videos_in_folder(videos_dir, workers=workers, save_as_npz=save_as_npz)

# def delete_np_files(directory: Path, save_as_npz: bool = True):
#     suffix = ".npz" if save_as_npz else ".npy"
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(suffix):
#                 file_path = Path(root) / file
#                 print(f"Deleting {file_path}")
#                 file_path.unlink()

# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) < 3:
#         print("Usage: python convert_mp4_to_np_multi.py <dataset_or_base_path> <npz|npy> [num_workers]")
#         exit(1)

#     input_path = sys.argv[1]
#     np_format = sys.argv[2].lower()
#     num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 140

#     if np_format not in ("npz", "npy"):
#         raise ValueError("Format must be 'npz' or 'npy'.")

#     save_as_npz = (np_format == "npz")

#     # 选择是生成还是删除
#     mode = input("Choose mode: [convert/delete]: ").strip().lower()

#     if mode == "convert":
#         traverse_and_convert(input_path, workers=num_workers, save_as_npz=save_as_npz)
#     elif mode == "delete":
#         delete_np_files(Path(input_path), save_as_npz=save_as_npz)
#     else:
#         print("Unknown mode. Please choose 'convert' or 'delete'.")

import shutil
from pathlib import Path
import os

def delete_np_dirs(base_path: str):
    """
    遍历 base_path 下所有子目录，删除名字为 'npy' 或 'npz' 的文件夹。

    Args:
        base_path (str): 要遍历的根目录路径
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Path not found: {base_path}")

    deleted_dirs = []

    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name in ("npy", "npz"):
                dir_path = Path(root) / dir_name
                try:
                    shutil.rmtree(dir_path)
                    deleted_dirs.append(str(dir_path))
                except Exception as e:
                    print(f"[Error] Failed to delete {dir_path}: {e}")

    print(f"[Info] Deleted {len(deleted_dirs)} folders.")
    return deleted_dirs

deleted = delete_np_dirs("/cognition/lerobot_Oatmeal/lerobot_split")
