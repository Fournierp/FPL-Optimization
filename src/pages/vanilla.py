import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import patches

from src.team_optimization import Team_Optimization
from src.utils import bezier_path, get_next_gameweek

PROJECTIONS_PATH = Path('data/projections')


def get_parameter_inputs(max_horizon: int) -> dict:
    params = {}

    with st.expander('Parameters', expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            params['horizon'] = st.slider(
                'Horizon', min_value=1, max_value=max_horizon, value=min(max_horizon, 5), step=1
            )
        with col2:
            params['premium'] = st.selectbox('Data type', ['Premium', 'Free'], 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            params['gk_weight'] = st.slider('GK Weight', min_value=0.01, max_value=1.0, value=0.03, step=0.02)
        with col2:
            params['first_bench_weight'] = st.slider('1st Weight', min_value=0.01, max_value=1.0, value=0.21, step=0.02)
        with col3:
            params['second_bench_weight'] = st.slider(
                '2nd Weight', min_value=0.01, max_value=1.0, value=0.06, step=0.02
            )
        with col4:
            params['third_bench_weight'] = st.slider('3rd Weight', min_value=0.01, max_value=1.0, value=0.01, step=0.02)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            params['decay'] = st.slider('Decay rate', min_value=0.0, max_value=1.0, value=0.9, step=0.02)
        with col2:
            params['vicecap_decay'] = st.slider('Vicecap rate', min_value=0.0, max_value=1.0, value=0.1, step=0.02)
        with col3:
            params['ft_val'] = st.slider('FT value', min_value=0.0, max_value=5.0, value=1.5, step=0.2)
        with col4:
            params['hit_val'] = st.slider('Hit value', min_value=2.0, max_value=8.0, value=6.0, step=0.5)
        with col5:
            params['itb_val'] = st.slider('ITB value', min_value=0.0, max_value=1.0, value=0.008, step=0.02)

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
) -> None:
    for j, row in team_df.iterrows():
        rectangle = patches.Rectangle((col_x, 14 - j), 12, 0.75, facecolor=color_position[row['Position']])
        ax.add_patch(rectangle)
        rx, ry = rectangle.get_xy()
        cx = rx + rectangle.get_width() / 2.0
        cy = ry + rectangle.get_height() / 2.0

        player_name = 'TAA' if row['Name'] == 'Alexander-Arnold' else row['Name']
        if gw != 'Base':
            player_name += (' (C)' if row['Cap'] else '') + (' (V)' if row['Vice'] else '')

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


def draw_transfers(ax, team_from: pd.DataFrame, team_to: pd.DataFrame, col_from: float, col_to: float) -> None:  # noqa: ANN001
    transfers = pd.concat([team_from, team_to], ignore_index=True)[['Name', 'Position']]
    transfers = transfers.drop_duplicates(keep=False).sort_index()

    for pos in ['G', 'D', 'M', 'F']:
        transfer_ = transfers.loc[transfers.Position == pos]

        for _ in range(int(transfer_.shape[0] / 2)):
            ax.add_patch(
                bezier_path(
                    (col_from + 12, 14 - transfer_.head(1).index[0] + 0.75 / 2),
                    (col_to, 14 - transfer_.tail(1).index[0] + 15 + 0.75 / 2),
                )
            )
            transfer_ = transfer_.drop([transfer_.head(1).index[0], transfer_.tail(1).index[0]])


def display_team_visualization(to: Team_Optimization, df: pd.DataFrame, chip_strat: list, horizon: int) -> None:
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_ylim(0, 15 + 1)
    ax.set_xlim(0, (horizon + 1) * 16 + 2.5)
    ax.axis('off')
    header_pos = 15.25

    color_position = {'GK': '#ebff00', 'DF': '#00ff87', 'MD': '#05f0ff', 'FW': '#e90052'}

    draw_team_column(ax, to.initial_team_df, 0, header_pos, color_position, 'Base')

    for i, gw in enumerate(np.sort(df.GW.unique())):
        df_gw = df.loc[gw == df.GW].reset_index(drop=True)

        draw_team_column(ax, df_gw, (i + 1) * 16, header_pos, color_position, gw)

        if chip_strat[i] is not None:
            ax.text((i + 1) * 16 + 6, header_pos + 1, chip_strat[i], fontsize=14, weight='bold', ha='center')

        if i == 0:
            draw_transfers(ax, to.initial_team_df, df_gw, 0, 16)
        else:
            df_prev = df.loc[gw - 1 == df.GW]
            draw_transfers(ax, df_prev, df_gw, i * 16, (i + 1) * 16)

    st.pyplot(fig, ax)
    plt.close(fig)


def run_optimization(params: dict, team_id: int, start: int, projection_data: pd.DataFrame) -> None:
    to = Team_Optimization(
        {
            'filter_ev': None,
            'horizon': params['horizon'],
            'noise': False,
            'ownership': False,
            'predictions': projection_data,
            'start': start,
            'team_id': team_id,
        }
    )

    to.build_model(
        model_name='vanilla',
        objective_type='decay' if params['decay'] != 0 else 'linear',
        decay_gameweek=params['decay'],
        vicecap_decay=params['vicecap_decay'],
        decay_bench=[
            params['gk_weight'],
            params['first_bench_weight'],
            params['second_bench_weight'],
            params['third_bench_weight'],
        ],
        ft_val=params['ft_val'],
        itb_val=params['itb_val'],
        hit_val=params['hit_val'],
    )

    df, chip_strat, total_ev, total_obj = to.solve(model_name='vanilla', log=True, time_lim=0)

    display_metrics(total_ev, total_obj)
    display_team_visualization(to, df, chip_strat, params['horizon'])


def write() -> None:
    st.title('FPL - Vanilla Model')
    st.header('Vanilla FPL Optimization.')

    plt.style.use('.streamlit/style.mplstyle')
    start = get_next_gameweek()

    csv_path = PROJECTIONS_PATH / f'enriched_expected_points_GW{start}.csv'
    projection_data = pd.read_csv(csv_path)

    max_horizon = len([col for col in projection_data.columns if 'GW' in col])

    params = get_parameter_inputs(max_horizon)

    if st.button('Run Optimization'):
        with Path('info.json').open() as f:
            info = json.load(f)
            team_id = info['team-id']

        with st.spinner('Running Optimization ...'):
            run_optimization(params, team_id, start, projection_data)
