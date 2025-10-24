import os
import subprocess
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def find_all_mp4_files(paths: List[str]) -> List[Path]:
    mp4_files = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: Path {path} does not exist, skipping.")
            continue
        for file in path.rglob("*.mp4"):
            mp4_files.append(file)
    return mp4_files

def repair_mp4_file(input_path: Path) -> bool:
    temp_output = input_path.with_suffix(".temp_fixed.mp4")
    try:
        cmd_copy = [
            "ffmpeg",
            "-y",
            "-hwaccel", "cuda",
            "-err_detect", "ignore_err",
            "-i", str(input_path),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-an",
            "-c", "copy",
            str(temp_output)
        ]
        result = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0:
            os.replace(temp_output, input_path)
            return True
        else:
            if temp_output.exists():
                temp_output.unlink()
            return reencode_mp4_file(input_path)
    except Exception as e:
        print(f"Error repairing {input_path}: {e}")
        if temp_output.exists():
            temp_output.unlink()
        return False

def reencode_mp4_file(input_path: Path) -> bool:
    temp_output = input_path.with_suffix(".temp_fixed.mp4")
    try:
        cmd_reencode = [
            "ffmpeg",
            "-y",
            "-hwaccel", "cuda",
            "-i", str(input_path),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "hevc_nvenc",
            "-preset", "p4",
            "-b:v", "5M",
            "-c:a", "aac",
            str(temp_output)
        ]
        result = subprocess.run(cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0:
            os.replace(temp_output, input_path)
            return True
        else:
            if temp_output.exists():
                temp_output.unlink()
            return False
    except Exception as e:
        print(f"Error re-encoding {input_path}: {e}")
        if temp_output.exists():
            temp_output.unlink()
        return False

def batch_repair_mp4(paths: List[str]):
    mp4_files = find_all_mp4_files(paths)
    print(f"Found {len(mp4_files)} mp4 files.")
    failed_files = []
    with ProcessPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(repair_mp4_file, mp4_file): mp4_file for mp4_file in mp4_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Repairing videos"):
            mp4_file = futures[future]
            try:
                result = future.result()
                if not result:
                    print(f"Failed to repair {mp4_file}")
                    failed_files.append(str(mp4_file))
            except Exception as exc:
                print(f"Exception repairing {mp4_file}: {exc}")
                failed_files.append(str(mp4_file))
    if failed_files:
        print("\nSummary: The following files failed to repair:")
        with open("repair_failed.txt", "w") as f:
            for file in failed_files:
                print(file)
                f.write(file + "\n")
    else:
        print("\nAll files repaired successfully!")

def generate_input_paths(repo_id: List[str], dataset_root: str) -> List[str]:
    return [str(Path(dataset_root) / repo) for repo in repo_id]

# if __name__ == "__main__":
#     repo_id = [
        # '0319','0319_2','0320','0320_2','0321','0324','0324_1','0324_2','0325_1','0325_2','0325_3','0325_4',
        # "0326","0401","0402","0403_1","0403_2","0407","0408",
        # "0417_3","0418","0418_1","0418_2", "0418_3","0418_4","0419","0421",
        # "0421_1","0421_2","0421_3",
        # "0422_1","0422_2","0422_3",
    #     # "0422_4","0423_1","0423_2","0423_3","0423_4",
    #     # "0424_3_pick_scoop",
    #     "0424_1_scoop_after_pour", "0424_2_scoop_after_pour", "0425_1_turn", "0425_3_turn", "0425_4_single_scoop", "0425_5_single_scoop"
    # ]
    # dataset_root = "/cognition/lerobot_Oatmeal/lerobot_split"
    # input_paths = generate_input_paths(repo_id, dataset_root)
    # batch_repair_mp4(input_paths)