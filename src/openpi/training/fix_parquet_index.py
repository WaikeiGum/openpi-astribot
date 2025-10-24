import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from tqdm import tqdm
import argparse
import os

def fix_parquet_indices(lerobot_dir, output_dir=None):
    lerobot_dir = Path(lerobot_dir)
    data_dir = lerobot_dir / "data"

    # 默认输出到 data_fix，否则覆盖原 data
    if output_dir is None or output_dir == "":
        output_dir = data_dir
        print(f"⚠️  将会直接覆盖原 data 目录: {data_dir}")
    else:
        output_dir = Path(output_dir)
        print(f"✅ 修正后的 parquet 文件会保存到: {output_dir}")

    parquet_files = sorted(
        data_dir.rglob("*.parquet"),
        key=lambda x: int(x.stem.split('_')[1])
    )
    global_index = 0

    for i, parquet_path in enumerate(tqdm(parquet_files, desc="Fixing parquet files")):
        table = pq.read_table(parquet_path)
        num_rows = table.num_rows
        # 更新 episode_index
        episode_index_array = pa.array([i] * num_rows, type=pa.int32())
        # 更新全局 index
        index_array = pa.array(list(range(global_index, global_index + num_rows)), type=pa.int64())
        global_index += num_rows

        # 替换字段
        new_table = table.set_column(
            table.schema.get_field_index('episode_index'), 'episode_index', episode_index_array
        )
        if 'index' in table.schema.names:
            new_table = new_table.set_column(
                new_table.schema.get_field_index('index'), 'index', index_array
            )
        else:
            new_table = new_table.append_column('index', index_array)

        # 输出到新路径，保持原有 chunk 目录结构
        relative_path = parquet_path.relative_to(data_dir)
        out_file = Path(output_dir) / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(new_table, out_file)

    print(f"\n所有 parquet 文件修正完毕，输出到 {output_dir}，最大全局 frame_index: {global_index - 1}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='lerobot 数据集根目录（含 data 子目录）')
    parser.add_argument('--output-dir', type=str, default=None, help='修正后输出目录，默认为覆盖原 data')
    args = parser.parse_args()
    fix_parquet_indices(args.data_dir, args.output_dir)
