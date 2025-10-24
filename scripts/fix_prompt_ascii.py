import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def encode_text_to_ascii_array(text: str) -> list:
    return [ord(c) for c in text]

def fix_prompt_column(parquet_path: Path):
    try:
        df = pd.read_parquet(parquet_path)
        if 'prompt' not in df.columns:
            print(f"[!] Skipped (no prompt): {parquet_path}")
            return

        modified = False

        def fix_prompt(p):
            nonlocal modified
            if isinstance(p, str):
                modified = True
                return encode_text_to_ascii_array(p)
            elif isinstance(p, np.ndarray) and np.issubdtype(p.dtype, np.integer):
                return p.tolist()
            elif isinstance(p, list) and all(isinstance(c, int) for c in p):
                return p
            else:
                print(f"[WARNING] Unexpected type in {parquet_path}: {type(p)}, treating as str.")
                modified = True
                return encode_text_to_ascii_array(str(p))

        df['prompt'] = df['prompt'].apply(fix_prompt)

        if modified:
            df.to_parquet(parquet_path, index=False)
            print(f"[✓] Fixed: {parquet_path}")
        else:
            print(f"[✓] Already OK: {parquet_path}")

    except Exception as e:
        print(f"[ERROR] Failed fixing {parquet_path}: {e}")

def fix_all_prompts(root_dir: str):
    root = Path(root_dir)
    parquet_files = list(root.rglob('*.parquet'))

    print(f"Found {len(parquet_files)} parquet files.")
    for pfile in tqdm(parquet_files, desc="Fixing prompts"):
        fix_prompt_column(pfile)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Root folder of parquet files')
    args = parser.parse_args()

    fix_all_prompts(args.input)
