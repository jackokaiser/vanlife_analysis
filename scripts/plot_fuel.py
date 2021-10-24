import argparse
import os
from typing import Optional
import pandas as pd

from vanlife_analysis.utils import load_fuel_records


def compute_personal_co2(fuel_records_df: pd.DataFrame) -> pd.DataFrame:
    # fuel
    # source:
    # - https://www.rncan.gc.ca/sites/www.nrcan.gc.ca/files/oee/pdf/transportation/fuel-efficient-technologies/autosmart_factsheet_6_f.pdf
    # - https://www.afgnv.org/bilan-co2-du-gnv-ou-biognv/
    personal_co2 = pd.DataFrame
    {
        'name': ['E10', 'GNC', 'BioGNC'],
        'kg_per_unit': [2.21, 2.96, 0.61]
    }

    co2_per_fuel = {
        'E10': 2.21,
        'GNC': 2.96,
        'BioGNC': 0.61
    }

    personal_co2 = pd.DataFrame()
    for fuel, kg_per_unit in co2_per_fuel.items():
        fuel_records = fuel_records_df[fuel_records_df['type'] == fuel]
        consumption = fuel_records['volume'].sum()
        fuel_data = {
            'name': fuel,
            'co2': int(consumption * kg_per_unit),
            'kg_per_unit': kg_per_unit,
            'consumption': consumption
        }
        personal_co2 = personal_co2.append(fuel_data, ignore_index=True)

    # food
    # source: https://librairie.ademe.fr/consommer-autrement/779-empreinte-energetique-et-carbone-de-l-alimentation-en-france.html
    france_co2_for_food = 163 * 10**9
    n_french = 67.06 * 10**6

    n_days = (fuel_records_df.iloc[-1]['date'] - fuel_records_df.iloc[0]['date']).days

    personal_co2 = personal_co2.append({
        'name': 'food',
        'co2': int(france_co2_for_food / n_french) / 365 * n_days
    }, ignore_index=True)
    return personal_co2


def plot_fuel(path_to_fuel: str, save_dir: Optional[str]) -> None:
    fuel_records_df = load_fuel_records(path_to_fuel)
    personal_co2 = compute_personal_co2(fuel_records_df)

    begin_date = fuel_records_df.iloc[0]['date']
    end_date = fuel_records_df.iloc[-1]['date']
    n_days = (end_date - begin_date).days

    print(f'kg of CO2 emitted for {n_days} days (from {begin_date.date()} to {end_date.date()})')
    print(personal_co2)
    print(f'total: {personal_co2["co2"].sum()} kg of CO2 equivalent')

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
