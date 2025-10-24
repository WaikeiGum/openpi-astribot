import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import Dict, List




def update_parquet_prompts(base_dir: str):
    print(f"\n🔍 [PARQUET] Replacing prompts under: {base_dir}")
    for root, _, files in os.walk(base_dir):
        for fname in tqdm(files, desc=f"Scanning {root}"):
            if not fname.endswith(".parquet"):
                continue
            fpath = os.path.join(root, fname)

            try:
                table = pq.read_table(fpath)
                if "prompt" not in table.column_names:
                    continue

                first_prompt = table["prompt"][0].as_py()
                if isinstance(first_prompt, list):
                    decoded_prompt = ''.join(chr(x) for x in first_prompt).split('\0')[0].strip()
                elif isinstance(first_prompt, str):
                    decoded_prompt = first_prompt.strip()
                else:
                    continue

                if decoded_prompt not in REPLACEMENTS:
                    continue

                new_prompt = REPLACEMENTS[decoded_prompt]
                print(f"[PARQUET REPLACE] {decoded_prompt} → {new_prompt}")

                # ✅ 转成 UTF-8 编码的 List[int]，并加 null 终止符（0）
                encoded_prompt = [ord(c) for c in new_prompt] + [0]
                new_prompt_array = pa.array([encoded_prompt] * len(table), type=pa.list_(pa.int32()))

                new_columns, new_names = [], []
                for name in table.column_names:
                    if name == "prompt":
                        new_columns.append(new_prompt_array)
                    else:
                        new_columns.append(table[name])
                    new_names.append(name)

                new_table = pa.table(new_columns, names=new_names)
                pq.write_table(new_table, fpath)

            except Exception as e:
                print(f"[ERROR] Failed to process {fpath}: {e}")

    print("✅ 所有 prompt 替换并编码为 List[int] 完成。")


def update_json_prompts(base_dir: str):
    print(f"\n🔍 [JSON] Replacing prompts under: {base_dir}")
    for root, _, files in os.walk(base_dir):
        for fname in tqdm(files, desc=f"Scanning {root}"):
            if not fname.endswith(".json") and not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(root, fname)

            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                updated_lines = []
                changed = False
                for line in lines:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        updated_lines.append(line)
                        continue

                    for key in ['task', 'prompt']:
                        if key in data and data[key] in REPLACEMENTS:
                            print(f"[JSON REPLACE] {data[key]} → {REPLACEMENTS[data[key]]}")
                            data[key] = REPLACEMENTS[data[key]]
                            changed = True
                    updated_lines.append(json.dumps(data, ensure_ascii=False) + "\n")

                if changed:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.writelines(updated_lines)

            except Exception as e:
                print(f"[ERROR] Failed to update {fpath}: {e}")


import os
import json
from tqdm import tqdm
from typing import Dict

def update_meta_tasks_jsonl(base_dir: str):
    print(f"\n🔍 [JSON] Replacing prompts in tasks.jsonl under: {base_dir}")
    for root, _, files in os.walk(base_dir):
        # 只处理meta目录
        if not root.endswith("meta"):
            continue
        for fname in files:
            if fname != "tasks.jsonl":
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                updated_lines = []
                changed = False
                for line in lines:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        updated_lines.append(line)
                        continue

                    for key in ['task', 'prompt']:
                        if key in data and data[key] in REPLACEMENTS:
                            print(f"[JSON REPLACE] {data[key]} → {REPLACEMENTS[data[key]]}")
                            data[key] = REPLACEMENTS[data[key]]
                            changed = True
                    updated_lines.append(json.dumps(data, ensure_ascii=False) + "\n")

                if changed:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.writelines(updated_lines)
                    print(f"✅ 替换完成：{fpath}")
                else:
                    print(f"✅ 无需修改：{fpath}")

            except Exception as e:
                print(f"[ERROR] Failed to update {fpath}: {e}")





def update_meta_episodes(base_dir: str):
    """
    只处理各数据集 meta 文件夹内的 episodes.jsonl
    只替换 tasks 字段（支持列表），其他文件不动
    """
    for root, dirs, files in os.walk(base_dir):
        if not root.endswith("meta"):
            continue
        for fname in files:
            if fname != "episodes.jsonl":
                continue
            fpath = os.path.join(root, fname)
            print(f"🔍 [EPISODES] Replacing tasks in: {fpath}")
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                changed = False
                new_lines = []
                for line in lines:
                    try:
                        data = json.loads(line)
                    except Exception:
                        new_lines.append(line)
                        continue
                    if "tasks" in data and isinstance(data["tasks"], list):
                        new_tasks: List[str] = []
                        task_changed = False
                        for t in data["tasks"]:
                            if t in REPLACEMENTS:
                                print(f"[EPISODE REPLACE] {t} → {REPLACEMENTS[t]}")
                                new_tasks.append(REPLACEMENTS[t])
                                task_changed = True
                            else:
                                new_tasks.append(t)
                        if task_changed:
                            data["tasks"] = new_tasks
                            changed = True
                    new_lines.append(json.dumps(data, ensure_ascii=False) + "\n")

                if changed:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    print(f"✅ 替换完成：{fpath}")
                else:
                    print(f"✅ 无需修改：{fpath}")
            except Exception as e:
                print(f"[ERROR] Failed to update {fpath}: {e}")


                    

# ✅ 替换表：仅替换 prompt，不动 task_index
# REPLACEMENTS: Dict[str, str] = {
#     "hold the shovel handle with the right arm and rotate it clockwise until the tip points forward":"grip the shovel handle with the right arm and rotate the entire shovel forward",
# }

# REPLACEMENTS: Dict[str, str] = {
#         "grip the shovel handle with the right arm and rotate the entire shovel forward":"turn the handle of the shovel with your right arm",
#         "hold the shovel tip with the right arm and rotate it clockwise until the tip points forward": "turn the front of the shovel with your right arm",
#         "turn the shovel with the right arm":"turn the front of the shovel with your right arm"
#     }


REPLACEMENTS: Dict[str, str] = {
        "turn the handle of the shovel with your right arm":"grip the shovel handle with the right arm and rotate the entire shovel forward",
        "turn the front of the shovel with your right arm":"hold the shovel tip with the right arm and rotate it clockwise until the tip points forward"
    }

def t1():


    # ✅ 要处理的根目录
    BASE_DIR = "/cognition/lerobot_Oatmeal/lerobot_split/test_turn"

    update_parquet_prompts(BASE_DIR)
    update_json_prompts(BASE_DIR)
    update_meta_tasks_jsonl(BASE_DIR)
    update_meta_episodes(BASE_DIR)
    print("\n✅ 所有 prompt 替换完成。")


# t1()

import pyarrow.parquet as pq

def get_prompt_from_parquet(parquet_path: str) -> str:
    """
    从指定 parquet 文件读取首个 prompt 字段，并解码成字符串。
    支持 prompt 为 List[int] 或 str。
    """
    table = pq.read_table(parquet_path)
    if "prompt" not in table.column_names:
        raise ValueError(f"文件 {parquet_path} 没有 prompt 字段！")

    first_prompt = table["prompt"][0].as_py()
    if isinstance(first_prompt, list):
        # 解码成字符串，去除 null 终止符
        s = ''.join(chr(x) for x in first_prompt).split('\0')[0].strip()
        return s
    elif isinstance(first_prompt, str):
        return first_prompt.strip()
    else:
        raise ValueError(f"prompt 字段类型未知: {type(first_prompt)}")



    

    
# # 示例用法
prompt = get_prompt_from_parquet("/cognition/lerobot_Oatmeal/lerobot_split/test_turn/0425_1_turn/data/chunk-000/episode_000000.parquet")
print(prompt)
    
    
    
import os
import pyarrow.parquet as pq
from tqdm import tqdm

def is_ascii_list(lst):
    """检查是否为仅包含ASCII码(0~127)的List[int]，允许末尾有0(null结束符)"""
    return isinstance(lst, list) and all(isinstance(x, int) and 0 <= x <= 127 for x in lst)

def check_all_parquet_prompt_types(base_dir: str):
    print(f"🔍 正在递归检查 {base_dir} 下所有parquet文件的prompt类型...\n")
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".parquet"):
                continue
            fpath = os.path.join(root, fname)
            try:
                table = pq.read_table(fpath)
                if "prompt" not in table.column_names:
                    print(f"❌ {fpath} 没有 prompt 字段，跳过")
                    continue

                prompt0 = table["prompt"][0].as_py()
                if isinstance(prompt0, list):
                    # 检查是否全部为ASCII
                    if is_ascii_list(prompt0):
                        print(f"✅ {fpath}: prompt 为 ASCII List[int]")
                    else:
                        print(f"⚠️  {fpath}: prompt 为 List[int] 但包含非 ASCII 范围，内容示例：{prompt0[:10]}")
                elif isinstance(prompt0, str):
                    print(f"⚠️  {fpath}: prompt 为 str 类型，内容示例：{prompt0[:50]!r}")
                else:
                    print(f"❌ {fpath}: prompt 类型未知: {type(prompt0)}, 内容示例：{repr(prompt0)}")
            except Exception as e:
                print(f"[ERROR] 读取 {fpath} 失败: {e}")

# # 示例用法
# BASE_DIR = "/cognition/lerobot_Oatmeal/lerobot_split"
# check_all_parquet_prompt_types(BASE_DIR)

