import argparse
import os
from typing import Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import seaborn as sns

from vanlife_analysis.utils import load_fuel_records, get_figsize, parse_date_interval


def is_full_refill(record: pd.Series) -> bool:
    # full/checkpoint checkboxes aren't in exported data, so let's use "missed" to track checkpoints
    return not record.missed and not is_tank_switch(record)


def is_tank_switch(record: pd.Series) -> bool:
    return np.isclose(record.volume, 0, atol=0.001)


def is_gnc_refill(record: pd.Series) -> bool:
    return record.type in ['GNC', 'BioGNC'] and not is_tank_switch(record)


def is_e10_refill(record: pd.Series) -> bool:
    return record.type == 'E10' and not is_tank_switch(record)


def compute_personal_co2(fuel_records_df: pd.DataFrame, n_days: int) -> pd.DataFrame:
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

    personal_co2 = personal_co2.append({
        'name': 'food',
        'co2': int(france_co2_for_food / n_french) / 365 * n_days
    }, ignore_index=True)
    return personal_co2


def annotate_rectangles(ax: Axes) -> None:
    for rectangle in ax.patches:
        height = rectangle.get_height()
        too_few_kms = 50
        if not np.isclose(height, 0, atol=too_few_kms):
            x = rectangle.get_x() + rectangle.get_width() / 2.0
            y = rectangle.get_y() + height - too_few_kms
            ax.annotate(f'{int(height)} km', (x, y), ha='center', va='top')


def annotate_kms(ax: Axes, xx: np.ndarray, yy: np.ndarray) -> None:
    assert len(xx) == len(yy)
    for x, height in zip(xx, yy):
        ax.annotate(f'{int(height)} km', (x, height), ha='center', va='bottom')


def fuel_unit(fuel: str) -> str:
    return 'L' if fuel == 'E10' else 'kg'


def annotate_volumes(ax: Axes, xx: np.ndarray, bar_heights: np.ndarray,
                     bar_values: np.ndarray, fuels: np.ndarray) -> None:
    assert bar_heights.shape == bar_values.shape
    assert len(xx) == len(bar_heights) == len(bar_values)

    too_little_volume = 5
    for x, heights, values in zip(xx, bar_heights, bar_values):
        y = 0
        for fuel, height, value in zip(fuels, heights, values):
            y += height / 2
            if not np.isclose(value, 0, atol=too_little_volume):
                ax.annotate(f'{int(value)} {fuel_unit(fuel)}', (x, y), ha='center', va='center')
            y += height / 2


def draw_driven_km(ax: Axes, driven_freq_df: pd.DataFrame) -> None:
    def line_format(date):
        """
        Convert time label to the format of pandas line plot
        """
        month = date.month_name()[:3]
        if month == 'Jan':
            month += f'\n{date.year}'
        return month

    driven_kms = driven_freq_df['driven'].sum(axis=1).values
    driven_volume_df = driven_freq_df['volume']
    driven_volumes = driven_volume_df.sum(axis=1).values
    scaling_factors = driven_kms / driven_volumes

    renormalized_driven_volume_df = driven_volume_df.copy()
    for col in driven_volume_df.columns:
        renormalized_driven_volume_df[col] *= scaling_factors

    renormalized_driven_volume_df.plot(ax=ax, kind='bar', ylabel='km', stacked=True)
    ax.set_xticklabels([line_format(index) for index in driven_freq_df.index])

    annotate_kms(ax, ax.get_xticks(), driven_kms)
    annotate_volumes(ax, ax.get_xticks(), renormalized_driven_volume_df.values,
                     driven_volume_df.values, driven_volume_df.columns.categories.values)


def get_driven_freq(fuel_records_df: pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
    driven_freq_df = fuel_records_df[['date', 'type', 'volume']].copy()
    driven_freq_df['driven'] = fuel_records_df['mileage'].diff()
    driven_freq_df = driven_freq_df.groupby([pd.Grouper(key='date', freq=freq), 'type']).sum()
    return driven_freq_df.unstack()


@dataclass
class FuelEfficiency:
    driven: int
    volume: float
    cost: float
    type: str

    def km_per_unit(self) -> float:
        return self.driven / self.volume

    def unit_per_km(self) -> float:
        return 1.0 / self.km_per_unit


def get_gnc_efficiencies(fuel_records_df: pd.DataFrame) -> list:
    first_tank_switch_idx = fuel_records_df.index[fuel_records_df.apply(is_tank_switch, axis=1)][0]
    # remove indexes before first tank switch (with start with an empty gnc tank)
    drop_mask = (fuel_records_df.index < first_tank_switch_idx + 1)
    # remove indexes that aren't gnc refills or tank switches
    drop_mask |= ((~fuel_records_df.apply(is_gnc_refill, axis=1)) & (~fuel_records_df.apply(is_tank_switch, axis=1)))
    gnc_refuels_and_switches = fuel_records_df.drop(fuel_records_df.index[drop_mask])
    gnc_refuels_and_switches['driven'] = gnc_refuels_and_switches['mileage'].diff()

    fuel_efficiencies = []

    current_driven = 0
    current_volume = 0
    current_cost = 0
    current_type = None

    for _, record in gnc_refuels_and_switches.iterrows():
        if current_type is not None:
            current_driven += record.driven

        if is_gnc_refill(record):
            if current_type is None:
                current_type = record.type
            assert current_type == record.type, f'Unsupported GNC mix {current_type} and {record.type}'
            current_volume += record.volume
            current_cost += record.cost
        elif is_tank_switch(record):
            assert current_type is not None, 'Subsequent tank switch'
            fuel_efficiency = FuelEfficiency(current_driven, current_volume, current_cost, current_type)
            fuel_efficiencies.append(fuel_efficiency)
            current_driven = 0
            current_volume = 0
            current_cost = 0
            current_type = None
        else:
            raise RuntimeError(f'Unexpected record: {record}')

    return fuel_efficiencies


def get_e10_efficiencies(fuel_records_df: pd.DataFrame) -> list:
    first_tank_switch_idx = fuel_records_df.index[fuel_records_df.apply(is_tank_switch, axis=1)][0]
    drop_mask = (fuel_records_df.index < first_tank_switch_idx + 1)
    fuel_records_df_cropped = fuel_records_df.drop(fuel_records_df.index[drop_mask])
    fuel_records_df_cropped['driven'] = fuel_records_df_cropped['mileage'].diff()

    # fast-forward to first full E10 refill, keeping track of current_type
    current_type = 'E10'
    fuel_records_it = fuel_records_df_cropped.iterrows()
    for _, record in fuel_records_it:
        if is_e10_refill(record) and is_full_refill(record):
            break
        elif is_gnc_refill(record):
            current_type = record.type
        elif is_tank_switch(record):
            current_type = 'E10'

    # continue the iteration, starting with full tank of E10
    fuel_efficiencies = []
    current_driven = 0
    current_volume = 0
    current_cost = 0

    for _, record in fuel_records_it:
        if current_type == 'E10':
            current_driven += record.driven

        if is_e10_refill(record):
            current_volume += record.volume
            current_cost += record.cost
            if is_full_refill(record):
                fuel_efficiencies.append(FuelEfficiency(current_driven, current_volume, current_cost, 'E10'))
                current_volume = 0
                current_cost = 0
                current_driven = 0
        elif is_gnc_refill(record):
            current_type = record.type
        elif is_tank_switch(record):
            current_type = 'E10'

    return fuel_efficiencies


def filter_fuel_efficiencies_outliers(fuel_efficiencies: list, gnc_km_per_unit_upper_thresh: float = 16,
                                      e10_km_per_unit_lower_thresh: float = 4.5) -> list:
    filtered_fuel_efficiencies = []
    for fuel_efficiency in fuel_efficiencies:
        if (fuel_efficiency.type == 'GNC' or fuel_efficiency.type == 'BioGNC') and \
           (fuel_efficiency.km_per_unit() > gnc_km_per_unit_upper_thresh):
            print(f'Dropping {fuel_efficiency}: {fuel_efficiency.km_per_unit()}km/kg > {gnc_km_per_unit_upper_thresh}km/kg'
                  ' (did you forget to record a tank switch?)')
            continue
        elif (fuel_efficiency.type == 'E10') and fuel_efficiency.km_per_unit() < e10_km_per_unit_lower_thresh:
            print(f'Dropping {fuel_efficiency}: {fuel_efficiency.km_per_unit()}km/L < {e10_km_per_unit_lower_thresh}km/L'
                  ' (did you forget to record a tank switch?)')
            continue
        filtered_fuel_efficiencies.append(fuel_efficiency)
    return filtered_fuel_efficiencies


def get_fuel_efficiencies(fuel_records_df: pd.DataFrame) -> pd.DataFrame:
    # ignore all records before the first tracked tank switch
    first_tank_switch_idx = fuel_records_df.index[fuel_records_df.volume.eq(0)][0]
    gnc_refills_idxs = fuel_records_df.index[fuel_records_df.type.isin(['GNC', 'BioGNC']) & (~fuel_records_df.missed)]
    last_gnc_refill_idx_before_first_switch = gnc_refills_idxs[gnc_refills_idxs < first_tank_switch_idx][-1]
    fuel_records_df = fuel_records_df[last_gnc_refill_idx_before_first_switch:]

    fuel_efficiencies = get_gnc_efficiencies(fuel_records_df) + get_e10_efficiencies(fuel_records_df)
    fuel_efficiencies = filter_fuel_efficiencies_outliers(fuel_efficiencies)

    fuel_efficiencies_df = pd.DataFrame(fuel_efficiencies)
    fuel_efficiencies_df['unit_per_km'] = fuel_efficiencies_df['volume'] / fuel_efficiencies_df['driven']
    fuel_efficiencies_df['unit_per_100km'] = fuel_efficiencies_df['unit_per_km'] * 100
    fuel_efficiencies_df['km_per_unit'] = 1 / fuel_efficiencies_df['unit_per_km']
    return fuel_efficiencies_df


def print_hline():
    print('\n')
    print('='*15)
    print('\n')


def plot_fuel(path_to_fuel: str, save_dir: Optional[str], date_interval: Optional[list]) -> None:
    sns.set_theme(style="ticks", context="talk", rc={"axes.spines.right": False, "axes.spines.top": False})
    plt.style.use("dark_background")
    sns.set_palette("muted")

    fuel_records_df = load_fuel_records(path_to_fuel)

    print_hline()

    if date_interval is not None:
        start_date, end_date = parse_date_interval(date_interval)
        fuel_records_df = fuel_records_df[(start_date <= fuel_records_df['date'])
                                          & (fuel_records_df['date'] <= end_date)]
    else:
        start_date = fuel_records_df.iloc[0]['date']
        end_date = fuel_records_df.iloc[-1]['date']

    n_days = (end_date - start_date).days

    personal_co2 = compute_personal_co2(fuel_records_df, n_days)
    print(f'kg of CO2 emitted for {n_days} days (from {start_date.date()} to {end_date.date()}):')
    print(personal_co2)
    print(f'total: {personal_co2["co2"].sum()} kg of CO2 equivalent')

    print_hline()

    fuel_efficiencies = get_fuel_efficiencies(fuel_records_df)
    print('Fuel efficiency:')
    print(fuel_efficiencies)

    print_hline()

    driven_freq_df = get_driven_freq(fuel_records_df)
    fig, ax = plt.subplots(1, 1, figsize=get_figsize())
    draw_driven_km(ax, driven_freq_df)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'driven_km.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to plot fuel usage and related co2 emissions')
    parser.add_argument('--fuel', help='Path to the exported fuel manager csv', type=str, required=True,
                        dest='path_to_fuel')
    parser.add_argument('--date_interval', help='Date interval for plotting locations', nargs='+', type=str,
                        required=False)
    parser.add_argument('--save_dir', help='Directory where the computed data is saved', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_fuel(**vars(args))
