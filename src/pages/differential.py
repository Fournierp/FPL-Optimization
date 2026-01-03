import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import patches

from src.pages.vanilla import PROJECTIONS_PATH, get_basic_parameters
from src.team_optimization import TeamOptimization
from src.utils import bezier_path, get_next_gameweek


def get_differential_parameters() -> dict:
    params = {}

    with st.expander('Differential', expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            params['nb_diff'] = st.slider('# Differentials', min_value=0, max_value=11, value=3, step=1)
        with col2:
            params['threshold'] = st.slider('Threshold', min_value=1, max_value=100, value=10, step=1)
        with col3:
            params['rank'] = st.selectbox(
                'Rank', ['Top_100', 'Top_1K', 'Top_10K', 'Top_50K', 'Top_100K', 'Top_250K', 'Top_500K'], 4
            )

    return params


def display_metrics(total_ev: float, total_obj: float) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Expected Value', np.round(total_ev, 2))
    with col2:
        st.metric('Objective Function Value', np.round(total_obj, 2))


def draw_team_column(  # noqa: PLR0913
    ax,  # noqa: ANN001
    team_df: pd.DataFrame,
    col_x: float,
    header_pos: float,
    color_position: dict,
    gw: str = 'Base',
    threshold: int | None = None,
) -> None:
    for j, row in team_df.iterrows():
        rectangle = patches.Rectangle((col_x, 14 - j), 12, 0.75, facecolor=color_position[row['Position']])
        ax.add_patch(rectangle)
        rx, ry = rectangle.get_xy()
        cx = rx + rectangle.get_width() / 2.0
        cy = ry + rectangle.get_height() / 2.0

        player_name = 'TAA' if row['Name'] == 'Alexander-Arnold' else row['Name']
        if gw != 'Base':
            player_name += ' (C)' if row['Cap'] else ''
            player_name += ' (V)' if row['Vice'] else ''
            if threshold is not None and 'Ownership' in row and row['Ownership'] < threshold:
                player_name += ' (D)'

        ax.annotate(
            player_name,
            (cx, cy),
            color='black',
            weight='bold',
            fontsize=14,
            ha='center',
            va='center',
        )

    ax.text(col_x + 6, header_pos, str(gw), fontsize=14, weight='bold', ha='center')

    ax.plot([col_x, col_x + 12], [3.875, 3.875], ls=':', lw='2.5', c='grey')

    ax.plot([col_x, col_x + 12], [15.0, 15.0], ls='-', lw='2.5', c='grey')


def draw_transfers(
    ax,  # noqa: ANN001
    team_from: pd.DataFrame,
    team_to: pd.DataFrame,
    col_from: float,
    col_to: float,
) -> None:
    transfers = team_from.append(team_to, ignore_index=True)[['Name', 'Position']]
    transfers = transfers.drop_duplicates(keep=False).sort_index()

    for pos in ['GK', 'DF', 'MD', 'FW']:
        transfer_ = transfers.loc[transfers.Position == pos]

        for _ in range(int(transfer_.shape[0] / 2)):
            ax.add_patch(
                bezier_path(
                    (col_from + 12, 14 - transfer_.head(1).index[0] + 0.75 / 2),
                    (col_to, 14 - transfer_.tail(1).index[0] + 15 + 0.75 / 2),
                )
            )
            transfer_ = transfer_.drop([transfer_.head(1).index[0], transfer_.tail(1).index[0]])


def display_team_visualization(to: TeamOptimization, df: pd.DataFrame, chip_strat: list, horizon: int) -> None:
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_ylim(0, 15 + 1)
    ax.set_xlim(0, (horizon + 1) * 16 + 2.5)
    ax.axis('off')
    header_pos = 15.25

    color_position = {'GK': '#ebff00', 'DF': '#00ff87', 'MD': '#05f0ff', 'FW': '#e90052'}

    draw_team_column(ax, to.initial_team_df, 0, header_pos, color_position, 'Base')

    for i, gw in enumerate(np.sort(df.GW.unique())):
        df_gw = df.loc[gw == df.GW]
        df_gw = df_gw.sort_values(by=['Position'], key=lambda x: x.map({'GK': 0, 'DF': 1, 'MD': 2, 'FW': 3}))
        df_gw = df_gw.sort_values(by=['Start'], ascending=False)
        df_gw = df_gw.reset_index(drop=True)

        draw_team_column(ax, df_gw, (i + 1) * 16, header_pos, color_position, gw)

        if chip_strat[i] is not None:
            ax.text((i + 1) * 16 + 6, header_pos + 1, chip_strat[i], fontsize=14, weight='bold', ha='center')

        if i == 0:
            draw_transfers(ax, to.initial_team_df, df_gw, 0, 16)
        else:
            df_prev = df.loc[gw - 1 == df.GW]
            df_prev = df_prev.sort_values(by=['Position'], key=lambda x: x.map({'GK': 0, 'DF': 1, 'MD': 2, 'FW': 3}))
            df_prev = df_prev.sort_values(by=['Start'], ascending=False)
            df_prev = df_prev.reset_index(drop=True)
            draw_transfers(ax, df_prev, df_gw, i * 16, (i + 1) * 16)

    st.pyplot(fig, ax)
    plt.close(fig)


def run_optimization(
    basic_params: dict, diff_params: dict, team_id: int, start: int, projection_data: pd.DataFrame
) -> None:
    to = TeamOptimization(
        {
            'filter_ev': None,
            'horizon': basic_params['horizon'],
            'noise': False,
            'ownership': True,
            'predictions': projection_data,
            'start': start,
            'team_id': team_id,
        }
    )

    to.build_model(
        {
            'model_name': 'vanilla',
            'objective_type': 'decay' if basic_params['decay'] != 0 else 'linear',
            'decay_gameweek': basic_params['decay'],
            'vicecap_decay': basic_params['vicecap_decay'],
            'decay_bench': [
                basic_params['gk_weight'],
                basic_params['first_bench_weight'],
                basic_params['second_bench_weight'],
                basic_params['third_bench_weight'],
            ],
            'ft_val': basic_params['ft_val'],
            'itb_val': basic_params['itb_val'],
            'hit_val': basic_params['hit_val'],
        }
    )

    to.differential_model(
        nb_differentials=diff_params['nb_diff'], threshold=diff_params['threshold'], target=diff_params['rank']
    )

    df, chip_strat, total_ev, total_obj = to.solve(model_name='differential', log=True, time_lim=0)

    display_metrics(total_ev, total_obj)
    display_team_visualization(to, df, chip_strat, basic_params['horizon'], diff_params['threshold'])


def write() -> None:
    st.title('FPL - Differential Model')

    plt.style.use('.streamlit/style.mplstyle')
    start = get_next_gameweek()

    csv_path = PROJECTIONS_PATH / f'enriched_expected_points_GW{start}.csv'
    projection_data = pd.read_csv(csv_path)

    max_horizon = len([col for col in projection_data.columns if 'GW' in col])

    basic_params = get_basic_parameters(max_horizon)
    diff_params = get_differential_parameters()

    if st.button('Run Optimization'):
        with Path('info.json').open() as f:
            info = json.load(f)
            team_id = info['team-id']

        with st.spinner('Running Optimization ...'):
            run_optimization(basic_params, diff_params, team_id, start, projection_data)
