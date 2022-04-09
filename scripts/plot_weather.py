import glob
import os
from tqdm import tqdm
import argparse
from typing import Optional, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from vanlife_analysis.utils import uncompress_and_load, get_figsize

TIME_TO_SLEEP = 15


def get_end_time(csv_path: str):
    sync_t = os.path.splitext(os.path.basename(csv_path))[0]
    end_sec = int(sync_t[len('sync_'):])
    return pd.to_datetime(end_sec, unit='s')


def average(df: pd.DataFrame, n_secs: int) -> pd.DataFrame:
    freq = f'{n_secs}S'
    return df.groupby(pd.Grouper(key='time', freq=freq)).mean()


def data_loader(weather_dir: str):
    filepaths = glob.glob(os.path.join(weather_dir, 'sync_*.csv'))
    dfs = []
    for csv_path in tqdm(filepaths, desc='Loading csv data'):
        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
        except pd.errors.EmptyDataError:
            continue
        time = pd.date_range(end=get_end_time(csv_path), freq=f'{TIME_TO_SLEEP}S', periods=len(df))
        df = df.assign(time=time)
        df = df.set_index(time)
        dfs.append(df)

    return pd.concat(dfs).sort_index()


def draw_temperature(ax: Axes, day_df: pd.DataFrame):
    day_df[['temp_ext', 'temp_room', 'temp_wall', 'temp_ceiling']].plot(ax=ax).legend(loc='upper left')
    ax.set_ylabel('Temperature [°C]')
    ax.set_xlabel('Time [hh:mm]')


def draw_humidity(ax: Axes, day_df: pd.DataFrame):
    day_df[['hum_ext', 'hum_room', 'hum_wall', 'hum_ceiling']].plot(ax=ax).legend(loc='upper left')
    ax.set_ylabel('Humidity [%]')
    ax.set_xlabel('Time [hh:mm]')


def draw_co2(ax: Axes, day_df: pd.DataFrame):
    day_df[['co2', 'tvoc']].plot(ax=ax).legend(loc='upper left')
    ax.set_ylabel('Concentration [ppm]')
    ax.set_xlabel('Time [hh:mm]')


def plot_and_save_fig(drawer: Callable, name: str, day_df: pd.DataFrame, day: pd.Timestamp,
                      save_dir: Optional[str] = None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    drawer(ax, day_df)
    fig.tight_layout()
    date = day.strftime("%Y-%m-%d")
    ax.set_title(f'{name} on day {date}', pad=-40)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, f'{date}_{name}.png'), dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_weather(weather_dir: str, save_dir: Optional[str]):
    sns.set_theme(style="ticks", context="paper", rc={"axes.spines.right": False, "axes.spines.top": False})
    sns.set_palette("muted")

    df = uncompress_and_load(weather_dir, data_loader)
    df = df.dropna()
    df = average(df, n_secs=120)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for day, day_df in tqdm(df.groupby(pd.Grouper(level='time', freq='D')), desc='Plotting days'):
        plot_and_save_fig(draw_temperature, 'temperature', day_df, day, save_dir)
        plot_and_save_fig(draw_humidity, 'humidity', day_df, day, save_dir)
        plot_and_save_fig(draw_co2, 'co2', day_df, day, save_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to plot weather data')
    parser.add_argument('--weather_dir', help='Directory containing the weather data', type=str, required=True)
    parser.add_argument('--save_dir', help='Directory where the computed data is saved', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_weather(**vars(args))
