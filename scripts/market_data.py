from pathlib import Path
import pandas as pd

def load_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == '.csv':
        return pd.read_csv(source)
    if source.suffix.lower() in {'.parquet', '.pq'}:
        return pd.read_parquet(source)
    raise ValueError(f'unsupported data format: {source.suffix}')
