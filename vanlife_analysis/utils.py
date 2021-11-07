from typing import Callable
import tempfile
import shutil
import os
import pandas as pd
import numpy as np


def get_figsize(fig_height: float = 10.0) -> tuple([float, float]):
    golden = (1 + np.sqrt(5)) / 2.0
    return (fig_height * golden, fig_height)


def uncompress_and_load(ann_dir_or_zip: str, data_loader: Callable) -> list:
    if ann_dir_or_zip.endswith('.zip'):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # unzip annotations to temporary directory
            shutil.unpack_archive(ann_dir_or_zip, tmp_dir, 'zip')
            files = os.listdir(tmp_dir)
            if len(files) == 1 and os.path.isdir(dir_path := os.path.join(tmp_dir, files[0])):
                return data_loader(dir_path)
            else:
                return data_loader(tmp_dir)
    else:
        return data_loader(tmp_dir)


def parse_fuel_manager(csv_path: str) -> dict:
    parsed_csv = {}
    current_name = None
    current_start_line = 0
    with open(csv_path, 'r') as f:
        for i_line, line in enumerate(f.readlines()):
            if line.startswith('###'):
                if i_line - current_start_line > 1:
                    parsed_csv[current_name] = pd.read_csv(
                        csv_path,
                        sep='\t',
                        skiprows=current_start_line,
                        nrows=i_line - current_start_line - 1)
                current_name = line[4:-1]
                current_start_line = i_line + 1
    return parsed_csv


def load_fuel_records(csv_path: str) -> pd.DataFrame:
    all_csv = parse_fuel_manager(csv_path)
    fuel_records_df = all_csv['records info (Vanderfool)']
    fuel_records_df.columns = fuel_records_df.columns.str.strip()
    fuel_records_df['type'] = fuel_records_df['type'].str.strip()
    fuel_records_df.loc[fuel_records_df['type'].eq('E95'), 'type'] = 'E10'
    fuel_records_df['type'] = fuel_records_df['type'].astype(
        pd.CategoricalDtype(['E10', 'GNC', 'BioGNC'], ordered=True))
    fuel_records_df['date'] = pd.to_datetime(fuel_records_df['date'], format='%Y%m%d')
    return fuel_records_df
