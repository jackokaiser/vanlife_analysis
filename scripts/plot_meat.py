import argparse
import os
from typing import Optional, Tuple
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

# from AGRIBALYSE3.1
co2_per_kg_by_meat = {
    'beef': 34.1,
    'chicken': 6.79,
    'duck': 9.33,
    'pork': 5.09,
    'salmon': 8.77,
    'tuna': 5.43,
    'lamb': 27.1,
    'sheep': 39.5,
    'shrimps': 7.64,
    'cod': 13.2,
    'crabs': 31.1,
    'veal': 30.7,
    'herrings': 2.19,
    'sea bass': 15.5,
    'pike perch': 6.64,
    'trout': 8.08,
    'dab': 10.8
}


def silence_other_columns(df: pd.DataFrame, other: str = 'other meat') -> None:
    columns = [f'_{col}' for col in df.columns]
    columns[0] = other
    df.columns = columns


def split_top_other(df: pd.DataFrame, topk: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    top_idxs = df.sum().nlargest(topk).index
    other_idxs = df.columns[~df.columns.isin(top_idxs)]
    top_df = df[top_idxs]
    other_df = df[other_idxs]
    silence_other_columns(other_df)
    return top_df, other_df


def draw_meat_quantity(meat_per_month: pd.DataFrame, topk: int = 10) -> plt.Figure:
    fig_width, fig_height = get_figsize()
    fig, ax = plt.subplots(1, 1, figsize=(fig_width * 1.5, fig_height))
    quantity_df = meat_per_month['quantity [g]'].copy()
    quantity_top_df, quantity_other_df = split_top_other(quantity_df, topk)
    quantity_top_df.plot(ax=ax, kind='bar', ylabel='Meat consumption quantity [g]', stacked=True)
    quantity_other_df.plot(ax=ax, kind='bar', stacked=True, color='black')

    ax.set_xticklabels([format_date(index) for index in meat_per_month.index])
    return fig


def draw_co2(meat_per_month: pd.DataFrame, topk: int = 5) -> plt.Figure:
    fig_width, fig_height = get_figsize()
    fig, ax = plt.subplots(1, 1, figsize=(fig_width * 1.5, fig_height))
    co2_df = meat_per_month['co2'].copy()
    co2_top_df, co2_other_df = split_top_other(co2_df, topk)

    co2_top_df.plot(ax=ax, kind='bar', ylabel='CO2 equivalent [kg]', stacked=True)
    co2_other_df.plot(ax=ax, kind='bar', stacked=True, color='black')

    ax.set_xticklabels([format_date(index) for index in meat_per_month.index])
    return fig


def plot_meat(path_to_meat: str, date_interval: list, save_dir: str) -> None:
    sns.set_theme(style="ticks", context="poster", rc={"axes.spines.right": False, "axes.spines.top": False})
    # plt.style.use("dark_background")
    # sns.set_palette("muted")

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
