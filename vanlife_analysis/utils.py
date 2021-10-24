import pandas as pd


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
    fuel_records_df['date'] = pd.to_datetime(fuel_records_df['date'], format='%Y%m%d')
    return fuel_records_df
