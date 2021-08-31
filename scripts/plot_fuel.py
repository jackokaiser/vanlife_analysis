import argparse
import os
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


def plot_fuel(path_to_fuel: str, save_dir: Optional[str]):
    # unit: kg of CO2
    personal_co2 = {}

    # fuel
    # source:
    # - https://www.rncan.gc.ca/sites/www.nrcan.gc.ca/files/oee/pdf/transportation/fuel-efficient-technologies/autosmart_factsheet_6_f.pdf
    # - https://www.afgnv.org/bilan-co2-du-gnv-ou-biognv/
    co2_per_fuel = {
        'E10': 2.21,
        'GNC': 2.96,
        'BioGNC': 0.61
    }

    fuel_records_df = load_fuel_records(path_to_fuel)
    begin_date = fuel_records_df.iloc[0]['date']
    end_date = fuel_records_df.iloc[-1]['date']
    n_days = (end_date - begin_date).days

    fuel_consumption = {}
    for fuel, kg_per_unit in co2_per_fuel.items():
        fuel_records = fuel_records_df[fuel_records_df['type'] == fuel]
        fuel_consumption[fuel] = fuel_records['volume'].sum()

    for fuel, consumption in fuel_consumption.items():
        personal_co2[fuel] = int(consumption * co2_per_fuel[fuel])

    # food
    # source: https://librairie.ademe.fr/consommer-autrement/779-empreinte-energetique-et-carbone-de-l-alimentation-en-france.html

    france_co2_for_food = 163 * 10**9
    n_french = 67.06 * 10**6
    personal_co2['food'] = int(france_co2_for_food / n_french) / 365 * n_days

    print(f'kg of CO2 emitted for {n_days} days (from {begin_date.date()} to {end_date.date()})')
    print(personal_co2)
    print(fuel_consumption)
    print(f'total: {sum(personal_co2.values())} kg of CO2 equivalent')

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to plot fuel usage and related co2 emissions')
    parser.add_argument('--fuel', help='Path to the exported fuel manager csv', type=str, required=True,
                        dest='path_to_fuel')
    parser.add_argument('--save_dir', help='Directory where the computed data is saved', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_fuel(**vars(args))
