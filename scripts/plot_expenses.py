import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from scipy import stats


def plot_monthly_expenses(df: pd.DataFrame, month_year: str) -> plt.Figure:
    categories = df.groupby(['Category']).sum()
    categories.sort_values('Cost', ascending=False, inplace=True)
    total_cost = categories["Cost"].sum()

    print(month_year)
    print(categories)
    print(f'total cost: {total_cost}€')

    legend_loc = {'loc': 'center left', 'bbox_to_anchor': (1.0, 0.5)}
    height = 7
    fig, axs = plt.subplots(1, 2, figsize=[1.618 * height, height])
    fig.suptitle(f'Expenses of {month_year} (total {total_cost}€)', fontsize=16)

    cost_series = categories[(categories['Cost'] / total_cost) >= 0.01]['Cost']
    cost_series.plot.pie(y='Cost', ax=axs[0], legend=False)
    cost_series.plot.bar(y='Cost', ax=axs[1], rot=60)
    plt.tight_layout()
    return fig


def plot_expenses(path_to_expenses: str, save_dir: Optional[str]):
    df = pd.read_csv(path_to_expenses)
    df.replace(' ', float('NaN'), inplace=True)
    df.dropna(subset=["Cost"], inplace=True)
    df.drop(df[df['Currency'] != 'EUR'].index, inplace=True)

    df['Cost'] = pd.to_numeric(df['Cost'])
    df['Date'] = pd.to_datetime(df['Date'])

    start_date = df['Date'].iloc[0] + pd.offsets.MonthBegin()
    end_date = df['Date'].iloc[-1] + pd.DateOffset(months=1)
    months = pd.date_range(start=start_date, end=end_date, freq="MS")

    for ii, prev_month in enumerate(months[:-1]):
        next_month = months[ii+1]
        plot_df = df[((prev_month <= df['Date']) & (df['Date'] < next_month))]

        month_year = prev_month.strftime('%B %Y')
        print('=' * 50)
        fig = plot_monthly_expenses(plot_df, month_year)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f'{prev_month.year}-{prev_month.month}_expenses.png'))
        else:
            plt.show()
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to plot expenses')
    parser.add_argument('--expenses', help='Path to the exported splitwise csv', type=str, required=True,
                        dest='path_to_expenses')
    parser.add_argument('--save_dir', help='Directory where the computed data is saved', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_expenses(**vars(args))
