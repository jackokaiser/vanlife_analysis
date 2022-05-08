import argparse
import os
from typing import Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
import seaborn as sns
import datetime
import logging

from vanlife_analysis.utils import load_fuel_records, get_figsize, parse_date_interval, configure_logger

logger = logging.getLogger(__name__)


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
        fuel_records = fuel_records_df[fuel_records_df['type'].eq(fuel)]
        consumption = fuel_records['volume'].sum()
        fuel_data = {
            'name': fuel,
            'co2': int(consumption * kg_per_unit),
            'kg_per_unit': kg_per_unit,
            'consumption': consumption
        }
        personal_co2 = personal_co2.append(fuel_data, ignore_index=True)

    return personal_co2


def annotate_euros(ax: Axes) -> None:
    for rectangle in ax.patches:
        height = rectangle.get_height()
        x = rectangle.get_x() + rectangle.get_width() / 2.0
        y = rectangle.get_y() + height
        ax.annotate(f'{height:.2f} €', (x, height), ha='center', va='bottom')


def annotate_kms(ax: Axes, xx: np.ndarray, yy: np.ndarray) -> None:
    assert len(xx) == len(yy)
    for x, height in zip(xx, yy):
        ax.annotate(f'{int(height)} km', (x, height), ha='center', va='bottom')


def annotate_kms_volumes(ax: Axes, fuel_efficiencies_per_month: pd.DataFrame) -> None:
    for xx, (_, driven_row), (_, volume_row) in zip(ax.get_xticks(),
                                                    fuel_efficiencies_per_month['driven'].iterrows(),
                                                    fuel_efficiencies_per_month['volume'].iterrows()):
        total_kms = driven_row.values.sum()
        ax.annotate(f'{int(total_kms)} km', (xx, total_kms), ha='center', va='bottom')

        vertical_centers = driven_row.values.cumsum() - driven_row.values / 2.
        for fuel, vertical_center, volume in zip(volume_row.index, vertical_centers, volume_row.values):
            if volume > 6:
                ax.annotate(f'{int(volume)}{fuel_unit(fuel)}', (xx, vertical_center), ha='center', va='center')


def annotate_volumes(ax: Axes, xx: np.ndarray, bar_heights: np.ndarray,
                     bar_values: np.ndarray, fuels: np.ndarray) -> None:
    assert bar_heights.shape == bar_values.shape
    assert len(xx) == len(bar_heights) == len(bar_values)
    y_heights = np.hstack((np.zeros((len(bar_heights), 1)), np.cumsum(bar_heights, axis=1)[:, :-1])) + bar_heights / 2
    too_little_volume = 5
    for x, yy, values in zip(xx, y_heights, bar_values):
        for y, fuel, value in zip(yy, fuels, values):
            if not value < too_little_volume:
                ax.annotate(f'{int(value)}{fuel_unit(fuel)}', (x, y), ha='center', va='center')


def fuel_unit(fuel: str) -> str:
    return 'L' if fuel == 'E10' else 'kg'


@dataclass
class FuelEfficiency:
    start_date: datetime.datetime
    end_date: datetime.datetime
    driven: int
    volume: float
    cost: float
    type: str

    def km_per_unit(self) -> float:
        return self.driven / self.volume

    def unit_per_km(self) -> float:
        return 1.0 / self.km_per_unit


def get_gnc_efficiencies(fuel_records_df: pd.DataFrame) -> pd.DataFrame:
    first_tank_switch_idx = fuel_records_df.index[fuel_records_df.apply(is_tank_switch, axis=1)][0]
    # remove indexes before first tank switch (with start with an empty gnc tank)
    drop_mask = (fuel_records_df.index < first_tank_switch_idx + 1)
    # remove indexes that aren't gnc refills or tank switches
    drop_mask |= ((~fuel_records_df.apply(is_gnc_refill, axis=1)) & (~fuel_records_df.apply(is_tank_switch, axis=1)))
    gnc_refuels_and_switches = fuel_records_df.drop(fuel_records_df.index[drop_mask])
    gnc_refuels_and_switches['driven'] = gnc_refuels_and_switches['mileage'].diff()

    fuel_efficiencies = []

    current_start_date = None
    current_end_date = None
    current_driven = 0
    current_volume = 0
    current_cost = 0
    current_type = None

    for record in gnc_refuels_and_switches.itertuples():
        if current_type is not None:
            current_driven += record.driven

        if is_gnc_refill(record):
            if current_type is None:
                assert current_start_date is None and (current_driven, current_volume, current_cost) == (0, 0, 0)
                current_start_date = record.date
                current_type = record.type
            assert current_type == record.type, f'Unsupported GNC mix {current_type} and {record.type}'
            current_volume += record.volume
            current_cost += record.cost
        elif is_tank_switch(record):
            assert current_type is not None, 'Subsequent tank switch'
            current_end_date = record.date
            fuel_efficiency = FuelEfficiency(current_start_date, current_end_date,
                                             current_driven, current_volume, current_cost, current_type)
            fuel_efficiencies.append(fuel_efficiency)
            current_start_date = None
            current_end_date = None
            current_driven = 0
            current_volume = 0
            current_cost = 0
            current_type = None
        else:
            raise RuntimeError(f'Unexpected record: {record}')

    return pd.DataFrame.from_records([asdict(fuel_efficiency) for fuel_efficiency in fuel_efficiencies])


def get_e10_efficiencies(fuel_records_df: pd.DataFrame) -> pd.DataFrame:
    first_tank_switch_idx = fuel_records_df.index[fuel_records_df.apply(is_tank_switch, axis=1)][0]
    drop_mask = (fuel_records_df.index < first_tank_switch_idx + 1)
    fuel_records_df_cropped = fuel_records_df.drop(fuel_records_df.index[drop_mask])
    fuel_records_df_cropped['driven'] = fuel_records_df_cropped['mileage'].diff()

    # fast-forward to first full E10 refill, keeping track of current_type
    current_type = 'E10'
    fuel_records_it = fuel_records_df_cropped.iterrows()
    for _, record in fuel_records_it:
        if is_e10_refill(record) and is_full_refill(record):
            # tank is full of E10, and we are currently driving on fuel of type current_type
            break
        elif is_gnc_refill(record):
            current_type = record.type
        elif is_tank_switch(record):
            current_type = 'E10'

    # continue the iteration, starting with full tank of E10
    fuel_efficiencies = []
    current_start_date = None
    current_end_date = None
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
                assert current_start_date is not None
                current_end_date = record.date if current_end_date is None else current_end_date
                fuel_efficiency = FuelEfficiency(current_start_date, current_end_date,
                                                 current_driven, current_volume, current_cost, 'E10')
                fuel_efficiencies.append(fuel_efficiency)
                current_start_date = record.date if current_type == 'E10' else None
                current_end_date = None
                current_volume = 0
                current_cost = 0
                current_driven = 0
        elif is_gnc_refill(record):
            current_type = record.type
            current_end_date = record.date
        elif is_tank_switch(record):
            current_type = 'E10'
            current_start_date = record.date if current_start_date is None else current_start_date

    return pd.DataFrame.from_records([asdict(fuel_efficiency) for fuel_efficiency in fuel_efficiencies])


def get_gnc_outliers(fuel_efficiencies_df: pd.DataFrame, gnc_km_per_unit_upper_thresh: float = 16) -> pd.DataFrame:
    outliers_mask = fuel_efficiencies_df.type.isin(["GNC", "BioGNC"]) & \
        (fuel_efficiencies_df['km_per_unit'] > gnc_km_per_unit_upper_thresh)

    outliers_records = fuel_efficiencies_df[outliers_mask]
    if len(outliers_records) > 0:
        logger.info(f'Found {len(outliers_records)} outliers in GNC records:\n{outliers_records}')
    return outliers_records


def get_e10_outliers(fuel_efficiencies_df: pd.DataFrame, e10_km_per_unit_lower_thresh: float = 4.5) -> pd.DataFrame:
    outliers_mask = fuel_efficiencies_df.type.eq("E10") & \
        (fuel_efficiencies_df['km_per_unit'] < e10_km_per_unit_lower_thresh)
    outliers_records = fuel_efficiencies_df[outliers_mask]
    if len(outliers_records) > 0:
        logger.info(f'Found {len(outliers_records)} outliers in E10 records:\n{outliers_records}')
    return outliers_records


def filter_fuel_efficiencies_outliers(fuel_efficiencies_df: pd.DataFrame, do_fail=False) -> pd.DataFrame:
    gnc_outliers = get_gnc_outliers(fuel_efficiencies_df)
    e10_outliers = get_e10_outliers(fuel_efficiencies_df)
    outlier_idxs = pd.Index.union(gnc_outliers.index, e10_outliers.index)
    if do_fail and len(outlier_idxs) > 0:
        raise RuntimeError('Outliers not tolerated')
    return fuel_efficiencies_df.drop(outlier_idxs)


def get_fuel_efficiencies(fuel_records_df: pd.DataFrame) -> pd.DataFrame:
    gnc_fuel_efficiencies = get_gnc_efficiencies(fuel_records_df)
    e10_fuel_efficiencies = get_e10_efficiencies(fuel_records_df)
    fuel_efficiencies = pd.concat([gnc_fuel_efficiencies, e10_fuel_efficiencies], ignore_index=True)
    fuel_efficiencies['km_per_unit'] = fuel_efficiencies['driven'] / fuel_efficiencies['volume']
    fuel_efficiencies['unit_per_km'] = fuel_efficiencies['volume'] / fuel_efficiencies['driven']
    fuel_efficiencies['euro_per_km'] = fuel_efficiencies['cost'] / fuel_efficiencies['driven']
    fuel_efficiencies['n_days'] = fuel_efficiencies['end_date'] - fuel_efficiencies['start_date']
    fuel_efficiencies = filter_fuel_efficiencies_outliers(fuel_efficiencies, do_fail=True)
    return fuel_efficiencies


def draw_fuel_efficiencies(fuel_efficiencies: pd.DataFrame) -> plt.Figure:
    avg_fuel_efficiencies = fuel_efficiencies.groupby(['type'], as_index=True).agg({
        'km_per_unit': ['mean', 'std'],
        'unit_per_km': ['mean', 'std'],
        'euro_per_km': ['mean', 'std'],
        'driven': 'sum',
        'volume': 'sum',
        'cost': 'sum'
    })

    sns.set_context('poster')
    fig, ax = plt.subplots(1, 1, figsize=get_figsize())
    avg_fuel_efficiencies['euro_per_km'].plot(kind='bar', ax=ax)
    ax.set_ylabel('Price [€/km]')
    ax.set_xlabel('Fuel type')
    annotate_euros(ax)
    E10_unit_per_km = avg_fuel_efficiencies[avg_fuel_efficiencies.index.str.match("E10")]['unit_per_km']
    logger.info(f'E10 fuel consumption: {E10_unit_per_km["mean"].item() * 100:.1f} L/100km '
                f'(+/-{E10_unit_per_km["std"].item() * 100:.1f})')
    return fig


def get_fuel_efficiencies_per_month(fuel_efficiencies: pd.DataFrame) -> pd.DataFrame:
    columns = ['date', 'type', 'volume', 'driven']
    records = []
    for row in fuel_efficiencies.itertuples():
        driven_per_day = row.driven / row.n_days.days
        volume_per_day = row.volume / row.n_days.days
        for date in pd.date_range(row.start_date, row.end_date, freq='1D', closed='left'):
            records.append([date, row.type, volume_per_day, driven_per_day])

    fuel_efficiencies_per_day = pd.DataFrame.from_records(records, columns=columns)
    fuel_efficiencies_per_month = fuel_efficiencies_per_day.groupby([pd.Grouper(key='date', freq='M'), 'type']).sum()
    return fuel_efficiencies_per_month.unstack().fillna(0)


def draw_driven_km(fuel_efficiencies: pd.DataFrame) -> plt.Figure:
    sns.set_context('talk')
    fig_width, fig_height = get_figsize()
    fig, ax = plt.subplots(1, 1, figsize=(fig_width * 1.5, fig_height))

    fuel_efficiencies_per_month = get_fuel_efficiencies_per_month(fuel_efficiencies)

    def line_format(date):
        """
        Convert time label to the format of pandas line plot
        """
        month = date.month_name()[:3]
        if month == 'Jan':
            month += f'\n{date.year}'
        return month

    fuel_efficiencies_per_month['driven'].plot(ax=ax, kind='bar', ylabel='km', stacked=True)
    ax.set_xticklabels([line_format(index) for index in fuel_efficiencies_per_month.index])
    annotate_kms_volumes(ax, fuel_efficiencies_per_month)
    return fig


def plot_fuel(path_to_fuel: str, save_dir: Optional[str], date_interval: Optional[list]) -> None:
    sns.set_theme(style="ticks", context="poster", rc={"axes.spines.right": False, "axes.spines.top": False})
    sns.set_palette("muted")

    log_path = os.path.join(save_dir, 'plut_fuel.log') if save_dir is not None else None
    configure_logger(logger)

    fuel_records_df = load_fuel_records(path_to_fuel)

    if date_interval is not None:
        start_date, end_date = parse_date_interval(date_interval)
        fuel_records_df = fuel_records_df[(start_date <= fuel_records_df['date'])
                                          & (fuel_records_df['date'] <= end_date)]
    else:
        start_date = fuel_records_df.iloc[0]['date']
        end_date = fuel_records_df.iloc[-1]['date']

    n_days = (end_date - start_date).days
    total_kms = fuel_records_df.iloc[-1]['mileage'] - fuel_records_df.iloc[0]['mileage']
    logger.info(f'plotting fuel consumption from {start_date.date()} to {end_date.date()}')
    logger.info(f'drove {total_kms}km in {n_days} days (average {total_kms/n_days:.0f}km/day)')

    personal_co2 = compute_personal_co2(fuel_records_df, n_days)
    logger.info(f'kg of CO2 emitted:')
    logger.info(personal_co2)
    logger.info(f'total: {personal_co2["co2"].sum()} kg of CO2 equivalent')

    logger.info('Median price per volume for E10: '
                f'{fuel_records_df[fuel_records_df["type"].eq("E10")]["volumecost"].median()}€/L')
    logger.info('Median price per volume for GNC: '
                f'{fuel_records_df[fuel_records_df["type"].isin(["GNC", "BioGNC"])]["volumecost"].median()}€/kg')

    fuel_efficiencies = get_fuel_efficiencies(fuel_records_df)
    logger.info('Fuel efficiency:')
    logger.info(fuel_efficiencies)

    fig_fuel_efficiencies = draw_fuel_efficiencies(fuel_efficiencies)
    fig_driven_km = draw_driven_km(fuel_efficiencies)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig_driven_km.savefig(os.path.join(save_dir, 'driven_km.png'), bbox_inches='tight')
        fig_fuel_efficiencies.savefig(os.path.join(save_dir, 'fuel_efficiencies.png'), bbox_inches='tight', dpi=150)
    else:
        plt.show()
    plt.close(fig_driven_km)
    plt.close(fig_fuel_efficiencies)


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
