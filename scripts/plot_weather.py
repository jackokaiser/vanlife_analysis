import glob
import os
from tqdm import tqdm
import argparse
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy import stats
import seaborn as sns

from vanlife_analysis.utils import uncompress_and_load, get_figsize

TIME_TO_SLEEP = 15


def get_end_time(csv_path: str):
    sync_t = os.path.splitext(os.path.basename(csv_path))[0]
    end_sec = int(sync_t[len('sync_'):])
    return pd.to_datetime(end_sec, unit='s')


def average(df: pd.DataFrame, n_secs: int = 60) -> pd.DataFrame:
    freq = f'{n_secs}S'
    return df.groupby(pd.Grouper(key='time', freq=freq)).mean().reset_index()


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


def plot_weather(weather_dir: str, save_dir: Optional[str]):
    sns.set_theme()
    df = uncompress_and_load(weather_dir, data_loader)
    df = df.dropna()
    df = average(df)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    start_date = df['time'].iloc[0].date()
    end_date = df['time'].iloc[-1].date()
    n_days = (end_date - start_date).days
    one_day = pd.DateOffset(days=1)

    with tqdm(range(n_days)) as progress_bar:
        for delta_day in progress_bar:
            ii_date = start_date + one_day * delta_day
            progress_bar.set_description(f'Plotting {ii_date.date()}')

            fig, axs = plt.subplots(3, 1, sharex=True, figsize=get_figsize())
            fig.suptitle(f'Day {ii_date.date()}', fontsize=16)
            plot_df = df[((ii_date <= df['time']) & (df['time'] < ii_date + one_day))]

            legend_loc = {'loc': 'center left', 'bbox_to_anchor': (1.0, 0.5)}
            ax_co2, ax_temp, ax_hum = axs
            plot_df[['co2', 'tvoc']].plot(ax=ax_co2).legend(**legend_loc)
            plot_df[['temp_ext', 'temp_room', 'temp_wall', 'temp_ceiling']].plot(ax=ax_temp).legend(**legend_loc)
            plot_df[['hum_ext', 'hum_room', 'hum_wall', 'hum_ceiling']].plot(ax=ax_hum).legend(**legend_loc)

            ax_co2.set_ylabel('Concentration [ppm]')
            ax_temp.set_ylabel('Temperature [°C]')
            ax_hum.set_ylabel('Humidity [%]')
            ax_hum.set_xlabel('Time [hh:mm]')
            # Define the datetime format
            date_form = DateFormatter("%H:%M")
            for ax in axs:
                ax.xaxis.set_major_formatter(date_form)

            plt.tight_layout()
            if save_dir is not None:
                filename = f'{ii_date.strftime("%Y-%m-%d")}.png'
                fig.savefig(os.path.join(save_dir, filename))
            else:
                plt.show()
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to plot weather data')
    parser.add_argument('--weather_dir', help='Directory containing the weather data', type=str, required=True)
    parser.add_argument('--save_dir', help='Directory where the computed data is saved', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_weather(**vars(args))
