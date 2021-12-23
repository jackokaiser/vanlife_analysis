import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import seaborn as sns

from scipy import stats

from vanlife_analysis.utils import parse_date_interval, get_figsize


def plot_monthly_expenses(df: pd.DataFrame, others_thresh: float = 0.03) -> plt.Figure:
    categories = df.groupby(['Category']).sum()['Cost']
    total_cost = categories.sum()

    print(categories)
    print(f'total cost: {total_cost:.0f}€')

    height = 7
    fig, axs = plt.subplots(1, 2, figsize=get_figsize(height))

    mask_others = (categories / total_cost) >= others_thresh
    series_others = pd.Series([categories[~mask_others].sum()], index=['Others'])

    masked_categories = categories[mask_others]
    masked_categories = masked_categories.append(series_others)
    masked_categories.sort_values(ascending=False, inplace=True)

    masked_categories.plot.pie(ylabel='', ax=axs[0], legend=False, autopct='%.0f%%', pctdistance=0.7)
    masked_categories.plot.bar(ylabel='Cost [€]', ax=axs[1], rot=60)
    plt.tight_layout()
    return fig, total_cost


def cleanup_df(df: pd.DataFrame) -> None:
    df.dropna(subset=["Cost"], inplace=True)
    df.drop(df[df['Currency'] != 'EUR'].index, inplace=True)
    df.replace(' ', float('NaN'), inplace=True)
    df['Category'].replace({'Entertainment - Other': 'Entertainment'}, inplace=True)
    df['Category'].replace({'Dining out': 'Restaurant'}, inplace=True)
    df['Category'].replace({'Hotel': 'Camping'}, inplace=True)
    df['Cost'] = pd.to_numeric(df['Cost'])
    df['Date'] = pd.to_datetime(df['Date'])


def plot_expenses(path_to_expenses: str, save_dir: str, date_interval: Optional[list]):
    sns.set_theme(style="ticks", context="talk", rc={"axes.spines.right": False, "axes.spines.top": False})
    plt.style.use("dark_background")
    sns.set_palette("muted")
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(path_to_expenses)
    cleanup_df(df)

    if date_interval is not None:
        start_date, end_date = parse_date_interval(date_interval)
        df = df[(start_date <= df['Date']) & (df['Date'] <= end_date)]
    else:
        start_date = df['Date'].iloc[0] + pd.offsets.MonthBegin()
        end_date = df['Date'].iloc[-1] + pd.DateOffset(months=1)

    fig, total_cost = plot_monthly_expenses(df)
    fig.suptitle(f'Expenses from {start_date.strftime("%d of %B")} to {end_date.strftime("%d of %B")}\n'
                 f'Total: {total_cost:.0f}€ over {(end_date - start_date).days} days', fontsize=16, x=0.25)
    fig.savefig(os.path.join(save_dir, 'total_expenses.png'), dpi=400)
    plt.close(fig)

    df_by_month = df.groupby([pd.Grouper(key='Date', freq='MS')])
    for timestamp, df_month in df_by_month:
        fig, total_cost = plot_monthly_expenses(df_month)
        month_year = timestamp.strftime('%B %Y')
        print('=' * 50)
        print(month_year)
        fig.suptitle(f'Expenses of {month_year} (total {total_cost:.0f}€)', fontsize=16, x=0.25)

        fig.savefig(os.path.join(save_dir, f'{timestamp.year}-{timestamp.month}_expenses.png'))
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to plot expenses')
    parser.add_argument('--expenses', help='Path to the exported splitwise csv', type=str, required=True,
                        dest='path_to_expenses')
    parser.add_argument('--date_interval', help='Date interval for plotting locations', nargs='+', type=str,
                        required=False)
    parser.add_argument('--save_dir', help='Directory where the computed data is saved', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_expenses(**vars(args))
