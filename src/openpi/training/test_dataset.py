import pyarrow.parquet as pq
from pathlib import Path

parquet_dir = Path("/cognition/lerobot_Oatmeal/lerobot_split/0319/data/chunk-000")

for parquet_file in sorted(parquet_dir.glob("*.parquet")):
    print(f"尝试打开 {parquet_file.name}")
    try:
        table = pq.read_table(parquet_file)
        print(f"成功打开: {parquet_file.name}, 行数: {table.num_rows}")
    except Exception as e:
        print(f"打开失败: {parquet_file.name}, 错误: {e}")
