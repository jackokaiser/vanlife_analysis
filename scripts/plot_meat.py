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

from vanlife_analysis.utils import load_fuel_records, get_figsize, parse_date_interval, configure_logger, format_date

co2_per_kg_fish = 0.46

co2_per_kg_by_meat = {
    'beef': 34.5,
    'chicken': 3.5,
    'duck': 4.09,
    'pork': 4.9,
    'salmon': 3.5,
    'tuna': 6.1,
    'lamb': 17.9,
    'sheep': 0.3,
    'shrimps': 2000,
    'cod': 11,
    'crabs': 0.5,
    'veal': 37,
    'herrings': co2_per_kg_fish,
    'sea bass': co2_per_kg_fish,
    'pike perch': co2_per_kg_fish,
    'trout': co2_per_kg_fish,
    'dab': co2_per_kg_fish,
    'sander': co2_per_kg_fish
}


def draw_meat_quantity(meat_per_month: pd.DataFrame) -> plt.Figure:
    fig_width, fig_height = get_figsize()
    fig, ax = plt.subplots(1, 1, figsize=(fig_width * 1.5, fig_height))
    quantity_df = meat_per_month['quantity [g]'].copy()
    top_10_meat = quantity_df.sum().nlargest(10).index
    quantity_df.columns = [col if col in top_10_meat else f'_{col}' for col in quantity_df.columns]
    quantity_df.plot(ax=ax, kind='bar', ylabel='Meat consumption quantity [g]', stacked=True)
    ax.set_xticklabels([format_date(index) for index in meat_per_month.index])
    return fig


def draw_co2(meat_per_month: pd.DataFrame) -> plt.Figure:
    fig_width, fig_height = get_figsize()
    fig, ax = plt.subplots(1, 1, figsize=(fig_width * 1.5, fig_height))
    co2_df = meat_per_month['co2'].copy()
    top_10_meat = co2_df.sum().nlargest(10).index
    co2_df.columns = [col if col in top_10_meat else f'_{col}' for col in co2_df.columns]
    co2_df.plot(ax=ax, kind='bar', ylabel='CO2 equivalent [kg]', stacked=True)
    ax.set_xticklabels([format_date(index) for index in meat_per_month.index])
    return fig


def plot_meat(path_to_meat: str, date_interval: list, save_dir: str) -> None:
    sns.set_theme(style="ticks", context="poster", rc={"axes.spines.right": False, "axes.spines.top": False})
    # plt.style.use("dark_background")
    sns.set_palette("muted")

    df = pd.read_csv(path_to_meat)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['meat'] = df['meat'].str.strip()
    df['meat'] = df['meat'].str.lower()

    if date_interval is not None:
        start_date, end_date = parse_date_interval(date_interval)
        df = df[(start_date <= df['date']) & (df['date'] <= end_date)]
    else:
        start_date = df['date'].iloc[0] + pd.offsets.MonthBegin()
        end_date = df['date'].iloc[-1] + pd.dateOffset(months=1)

    df['co2'] = df.apply(lambda row: row['quantity [g]'] * 1e-3 * co2_per_kg_by_meat[row['meat']], axis=1)
    meat_per_month = df.groupby([pd.Grouper(key='date', freq='M'), 'meat']).sum().unstack()

    fig_quantity = draw_meat_quantity(meat_per_month)
    fig_co2 = draw_co2(meat_per_month)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig_quantity.savefig(os.path.join(save_dir, 'meat_consumption_quantity.png'), bbox_inches='tight', dpi=150)
        fig_co2.savefig(os.path.join(save_dir, 'meat_consumption_co2.png'), bbox_inches='tight', dpi=150)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to plot carbon footprint of meat consumption')
    parser.add_argument('--meat', help='Path to the meat consumption csv', type=str, required=True, dest='path_to_meat')
    parser.add_argument('--date_interval', help='Date interval for plotting locations', nargs='+', type=str,
                        required=False)
    parser.add_argument('--save_dir', help='Directory where the computed data is saved', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_meat(**vars(args))
