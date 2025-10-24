import os
import json
import random
import argparse
from pathlib import Path
import pandas as pd

# =====================
# 修改配置
# =====================
original_task = "turn the shovel with the right arm"
new_task = "hold the shovel tip with the right arm and rotate it clockwise until the tip points forward"
random_check_num = 5  # 随机抽多少个 parquet 检查


def modify_task_jsonl(jsonl_path: Path):
    modified = False
    new_lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == '':
                continue  # 跳过空行
            try:
                item = json.loads(line)
                if isinstance(item, dict) and item.get("task") == original_task:
                    item["task"] = new_task
                    modified = True
                new_lines.append(item)
            except json.JSONDecodeError:
                print(f"[ERROR] Cannot parse line in {jsonl_path}")
                continue

    if modified:
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in new_lines:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[✓] Modified task.jsonl: {jsonl_path}")



import numpy as np

def decode_prompt(p):
    if isinstance(p, np.ndarray) and np.issubdtype(p.dtype, np.integer):
        try:
            return ''.join([chr(int(c)) for c in p.tolist()]).strip()
        except:
            return str(p)
    elif isinstance(p, list) and all(isinstance(c, int) for c in p):
        try:
            return ''.join([chr(c) for c in p]).strip()
        except:
            return str(p)
    elif isinstance(p, str):
        return p.strip()
    else:
        return str(p)


def modify_parquet_prompt(parquet_path: Path):
    try:
        df = pd.read_parquet(parquet_path)
        if 'prompt' in df.columns:
            # 先decode所有prompt列
            df['prompt'] = df['prompt'].apply(decode_prompt)
            mask = df['prompt'] == original_task
            if mask.any():
                df.loc[mask, 'prompt'] = new_task
                df['prompt'] = df['prompt'].astype(str)  # 最后强制为str
                df.to_parquet(parquet_path, index=False)
                print(f"[✓] Modified parquet prompts: {parquet_path}")
    except Exception as e:
        print(f"[ERROR] Failed processing {parquet_path}: {e}")




def random_check_parquet(parquet_files: list, sample_num: int):
    print(f"\n[Random Check] Sample {sample_num} parquet files:")
    sampled_files = random.sample(parquet_files, min(sample_num, len(parquet_files)))
    for pfile in sampled_files:
        try:
            df = pd.read_parquet(pfile)
            print(f"\nFile: {str(pfile)}")

            if 'prompt' in df.columns:
                # 🛠️ 这里加上decode
                decoded_prompts = df['prompt'].apply(decode_prompt)
                print(decoded_prompts.head(10).to_string(index=False))
            else:
                print("[!] No 'prompt' column found.")
        except Exception as e:
            print(f"[ERROR] Cannot read {pfile}: {e}")



def main(root_path: str):
    root = Path(root_path)

    # 修改所有 task.jsonl
    jsonl_files = list(root.rglob('tasks.jsonl'))
    for jsonl_file in jsonl_files:
        modify_task_jsonl(jsonl_file)

    # 修改所有 parquet
    parquet_files = list(root.rglob('*.parquet'))
    for parquet_file in parquet_files:
        modify_parquet_prompt(parquet_file)

    # 随机抽取 N 个 parquet 检查
    if parquet_files:
        random_check_parquet(parquet_files, random_check_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Root folder to search')
    args = parser.parse_args()

    main(args.input)
