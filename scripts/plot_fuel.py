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

from vanlife_analysis.utils import load_fuel_records, get_figsize


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
    driven_freq_df = fuel_records_df[['date']].copy()
    driven_freq_df['driven'] = fuel_records_df['mileage'].diff()
    driven_freq_df['type'] = fuel_records_df['type']
    driven_freq_df['volume'] = fuel_records_df['volume']
    driven_freq_df = driven_freq_df.groupby([pd.Grouper(key='date', freq=freq), 'type']).sum()
    return driven_freq_df.unstack()


@dataclass
class FuelEfficiency:
    driven: int
    volume: float
    type: str


def get_gnc_efficiencies(fuel_records_df: pd.DataFrame) -> list:
    fuel_efficiencies = []

    gnc_mask = fuel_records_df.type.isin(['GNC', 'BioGNC']) | fuel_records_df.volume.eq(0)
    gnc_refuels_and_switches = fuel_records_df[gnc_mask].copy()
    gnc_refuels_and_switches['driven'] = gnc_refuels_and_switches['mileage'].diff()
    fuel_efficiency = None
    for _, record in gnc_refuels_and_switches.iterrows():
        if fuel_efficiency is None:
            # first refuel after a tank switch
            fuel_efficiency = FuelEfficiency(0, record.volume, record.type)
        else:
            fuel_efficiency.driven += record.driven
            fuel_efficiency.volume += record.volume
            if record.volume == 0:
                # tank switch
                fuel_efficiencies.append(fuel_efficiency)
                fuel_efficiency = None
            else:
                assert fuel_efficiency.type == record.type, \
                    f'Unsupported GNC mix {fuel_efficiency.type} and {record.type}'
    return fuel_efficiencies


def is_full_tank(record: pd.Series) -> bool:
    # full/checkpoint checkboxes aren't in exported data, so let's use "missed" to track checkpoints
    return not record.missed


def get_e10_efficiencies(fuel_records_df: pd.DataFrame) -> list:
    tank_switch_mask = fuel_records_df.volume.eq(0)
    gnc_refuel_mask = fuel_records_df.type.isin(['GNC', 'BioGNC']) & fuel_records_df.volume.gt(0)

    # find the intervals of records where we drove on GNC
    stop_idxs = fuel_records_df.index[tank_switch_mask]
    start_candidates_idxs = fuel_records_df.index[gnc_refuel_mask]
    gnc_driving_intervals = []
    for stop_idx in stop_idxs:
        gnc_driving_intervals.append([start_candidates_idxs[0], stop_idx])
        # remove re-fueling idx that happened before the swtich
        start_candidates_idxs = start_candidates_idxs[start_candidates_idxs > stop_idx]

    fuel_records_df = fuel_records_df.copy()
    fuel_records_df['driven'] = fuel_records_df['mileage'].diff()

    first_e10_refuel_idx = fuel_records_df.index[fuel_records_df.type.eq('E10') &
                                                 fuel_records_df.volume.gt(0) &
                                                 fuel_records_df.missed.eq(0)][0]

    # start with a full tank
    fuel_records_it = fuel_records_df.loc[first_e10_refuel_idx:].iterrows()
    _, record = next(fuel_records_it)
    fuel_efficiency = FuelEfficiency(0, 0, record.type)

    fuel_efficiencies = []
    for idx, record in fuel_records_it:
        if not any(start_gnc < idx <= stop_gnc for start_gnc, stop_gnc in gnc_driving_intervals):
            # add up driven kms if we were not driving on GNC
            fuel_efficiency.driven += record.driven
        if record.volume > 0 and record.type == 'E10' and is_full_tank(record):
            fuel_efficiencies.append(fuel_efficiency)
            fuel_efficiency = FuelEfficiency(0, record.volume, record.type)

    return fuel_efficiencies


def get_fuel_efficiencies(fuel_records_df: pd.DataFrame) -> pd.DataFrame:
    # ignore all records before the first tracked tank switch
    first_tank_switch_idx = fuel_records_df.index[fuel_records_df.volume.eq(0)][0]
    fuel_records_df = fuel_records_df[first_tank_switch_idx+1:]

    fuel_efficiencies = get_gnc_efficiencies(fuel_records_df) + get_e10_efficiencies(fuel_records_df)

    fuel_efficiencies_df = pd.DataFrame(fuel_efficiencies)
    fuel_efficiencies_df['unit_per_km'] = fuel_efficiencies_df['volume'] / fuel_efficiencies_df['driven']
    fuel_efficiencies_df['unit_per_100km'] = fuel_efficiencies_df['unit_per_km'] * 100
    fuel_efficiencies_df['km_per_unit'] = 1 / fuel_efficiencies_df['unit_per_km']
    return fuel_efficiencies_df


def print_hline():
    print('\n')
    print('='*15)
    print('\n')


def plot_fuel(path_to_fuel: str, save_dir: Optional[str]) -> None:
    # Apply the default theme
    sns.set_theme()

    fuel_records_df = load_fuel_records(path_to_fuel)

    print_hline()

    begin_date = fuel_records_df.iloc[0]['date']
    end_date = fuel_records_df.iloc[-1]['date']
    n_days = (end_date - begin_date).days
    personal_co2 = compute_personal_co2(fuel_records_df, n_days)
    print(f'kg of CO2 emitted for {n_days} days (from {begin_date.date()} to {end_date.date()}):')
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
    parser.add_argument('--save_dir', help='Directory where the computed data is saved', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_fuel(**vars(args))
