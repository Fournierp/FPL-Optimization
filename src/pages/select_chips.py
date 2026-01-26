import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import patches

from src.pages.vanilla import PROJECTIONS_PATH, display_metrics, get_basic_parameters
from src.team_optimization import TeamOptimization
from src.utils import bezier_path, get_next_gameweek


def get_chip_parameters(start: int, horizon: int) -> dict:
    with st.expander('Chips', expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            wc_gw = st.selectbox('Wildcard', [None] + [start + gw for gw in np.arange(horizon)], 0)
        with col2:
            fh_gw = st.selectbox('Freehit', [None] + [start + gw for gw in np.arange(horizon)], 0)
        with col3:
            tc_gw = st.selectbox('Triple Captain', [None] + [start + gw for gw in np.arange(horizon)], 0)
        with col4:
            bb_gw = st.selectbox('Bench Boost', [None] + [start + gw for gw in np.arange(horizon)], 0)

    return {'wc_gw': wc_gw, 'fh_gw': fh_gw, 'tc_gw': tc_gw, 'bb_gw': bb_gw}


def validate_chip_selection(wc_gw: int, fh_gw: int, tc_gw: int, bb_gw: int) -> bool:
    return not (
        wc_gw in (fh_gw, tc_gw, bb_gw)  # noqa: RUF021
        and wc_gw is not None
        or fh_gw in (wc_gw, tc_gw, bb_gw)  # noqa: RUF021
        and fh_gw is not None
        or tc_gw in (wc_gw, fh_gw, bb_gw)  # noqa: RUF021
        and tc_gw is not None
        or bb_gw in (fh_gw, tc_gw, wc_gw)  # noqa: RUF021
        and bb_gw is not None
    )


def draw_initial_team(ax, team_df: pd.DataFrame, color_position: dict, header_pos: float) -> None:  # noqa: ANN001
    for j, row in team_df.iterrows():
        rectangle = patches.Rectangle((0, 14 - j), 12, 0.75, facecolor=color_position[row['Position']])
        ax.add_patch(rectangle)
        rx, ry = rectangle.get_xy()
        cx = rx + rectangle.get_width() / 2.0
        cy = ry + rectangle.get_height() / 2.0
        ax.annotate(
            'TAA' if row['Name'] == 'Alexander-Arnold' else row['Name'],
            (cx, cy),
            color='black',
            weight='bold',
            fontsize=14,
            ha='center',
            va='center',
        )

    ax.text(cx, header_pos, 'Base', fontsize=14, weight='bold', ha='center')

    ax.plot([0, 12], [3.875, 3.875], ls=':', lw='2.5', c='grey')

    ax.plot([0, 12], [15.0, 15.0], ls='-', lw='2.5', c='grey')


def draw_gameweek_column(  # noqa: PLR0913
    ax,  # noqa: ANN001
    df_gw: pd.DataFrame,
    gw: int,
    i: int,
    color_position: dict,
    header_pos: float,
    chip_strat: list,
) -> None:
    for j, row in df_gw.iterrows():
        rectangle = patches.Rectangle(((i + 1) * 16, 14 - j), 12, 0.75, facecolor=color_position[row['Position']])
        ax.add_patch(rectangle)
        rx, ry = rectangle.get_xy()
        cx = rx + rectangle.get_width() / 2.0
        cy = ry + rectangle.get_height() / 2.0
        ax.annotate(
            (
                ('TAA' if row['Name'] == 'Alexander-Arnold' else row['Name'])
                + (' (C)' if row['Cap'] else '')
                + (' (V)' if row['Vice'] else '')
            ),
            (cx, cy),
            color='black',
            weight='bold',
            fontsize=14,
            ha='center',
            va='center',
        )

    ax.text(cx, header_pos, str(gw), fontsize=14, weight='bold', ha='center')
    if chip_strat[i] is not None:
        ax.text(cx, header_pos + 1, chip_strat[i], fontsize=14, weight='bold', ha='center')

    ax.plot([(i + 1) * 16, (i + 1) * 16 + 12], [3.875, 3.875], ls=':', lw='2.5', c='grey')
    ax.plot([(i + 1) * 16, (i + 1) * 16 + 12], [15.0, 15.0], ls='-', lw='2.5', c='grey')


def draw_transfers(ax, df, df_gw, to, i: int, gw: int) -> None:  # noqa: ANN001, PLR0913
    if i == 0:
        transfers = pd.concat([to.initial_team_df, df_gw], ignore_index=True)[['Name', 'Position']]
        transfers = transfers.drop_duplicates(keep=False).sort_index()

        for pos in ['GK', 'DF', 'MD', 'FW']:
            transfer_ = transfers.loc[transfers.Position == pos]

            for _ in range(int(transfer_.shape[0] / 2)):
                ax.add_patch(
                    bezier_path(
                        (12, 14 - transfer_.head(1).index[0] + 0.75 / 2),
                        (16, 14 - transfer_.tail(1).index[0] + 15 + 0.75 / 2),
                    )
                )
                transfer_ = transfer_.drop([transfer_.head(1).index[0], transfer_.tail(1).index[0]])
    else:
        df_prev = df.loc[gw - 1 == df.GW]
        transfers = pd.concat([df_prev, df_gw], ignore_index=True)[['Name', 'Position']]
        transfers = transfers.drop_duplicates(keep=False).sort_index()

        for pos in ['GK', 'DF', 'MD', 'FW']:
            transfer_ = transfers.loc[transfers.Position == pos]

            for _ in range(int(transfer_.shape[0] / 2)):
                ax.add_patch(
                    bezier_path(
                        (i * 16 + 12, 14 - transfer_.head(1).index[0] + 0.75 / 2),
                        ((i + 1) * 16, 14 - transfer_.tail(1).index[0] + 15 + 0.75 / 2),
                    )
                )
                transfer_ = transfer_.drop([transfer_.head(1).index[0], transfer_.tail(1).index[0]])


def display_team_visualization(df: pd.DataFrame, chip_strat: list, to, horizon: int) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_ylim(0, 15 + 1)
    ax.set_xlim(0, (horizon + 1) * 16 + 2.5)
    ax.axis('off')
    header_pos = 15.25

    color_position = {'GK': '#ebff00', 'DF': '#00ff87', 'MD': '#05f0ff', 'FW': '#e90052'}

    draw_initial_team(ax, to.initial_team_df, color_position, header_pos)

    for i, gw in enumerate(np.sort(df.GW.unique())):
        df_gw = df.loc[gw == df.GW]
        df_gw = df_gw.sort_values(by=['Position'], key=lambda x: x.map({'GK': 0, 'DF': 1, 'MD': 2, 'FW': 3}))
        df_gw = df_gw.sort_values(by=['Start'], ascending=False)
        df_gw = df_gw.reset_index(drop=True)
        draw_gameweek_column(ax, df_gw, gw, i, color_position, header_pos, chip_strat)

        df_prev = df.loc[gw - 1 == df.GW]
        df_prev = df_prev.sort_values(by=['Position'], key=lambda x: x.map({'GK': 0, 'DF': 1, 'MD': 2, 'FW': 3}))
        df_prev = df_prev.sort_values(by=['Start'], ascending=False)
        df_prev = df_prev.reset_index(drop=True)
        draw_transfers(ax, df_prev, df_gw, to, i, gw)

    st.pyplot(fig, ax)
    plt.close(fig)


def run_optimization(params: dict, chips: dict, team_id: int, start: int, projection_data: pd.DataFrame) -> None:
    to = TeamOptimization(
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
        {
            'model_name': 'select_chips',
            'freehit_gw': chips['fh_gw'] - start if chips['fh_gw'] is not None else -1,
            'wildcard_gw': chips['wc_gw'] - start if chips['wc_gw'] is not None else -1,
            'bboost_gw': chips['bb_gw'] - start if chips['bb_gw'] is not None else -1,
            'threexc_gw': chips['tc_gw'] - start if chips['tc_gw'] is not None else -1,
            'objective_type': 'decay' if params['decay'] != 0 else 'linear',
            'decay_gameweek': params['decay'],
            'vicecap_decay': params['vicecap_decay'],
            'decay_bench': [
                params['gk_weight'],
                params['first_bench_weight'],
                params['second_bench_weight'],
                params['third_bench_weight'],
            ],
            'ft_val': params['ft_val'],
            'itb_val': params['itb_val'],
            'hit_val': params['hit_val'],
        }
    )

    df, chip_strat, total_ev, total_obj = to.solve(model_name='select_chips', log=True, time_lim=0)

    display_metrics(total_ev, total_obj)
    display_team_visualization(df, chip_strat, to, params['horizon'])


def write() -> None:
    st.title('FPL - Select Chips Model')

    with st.expander('📖 Instructions', expanded=False):
        st.markdown(
            """
            **Select Chips Optimization** - Manually specify which chips to use and when.

            This model gives you full control over chip timing. You decide when to use Wildcard, Free Hit,
            Bench Boost, and Triple Captain, and the optimizer will build the best strategy around your choices.
            """
        )

    plt.style.use('.streamlit/style.mplstyle')

    start = get_next_gameweek()

    csv_path = PROJECTIONS_PATH / f'enriched_expected_points_GW{start}.csv'
    projection_data = pd.read_csv(csv_path)

    max_horizon = len([col for col in projection_data.columns if 'GW' in col])

    params = get_basic_parameters(max_horizon)
    chips = get_chip_parameters(start, params['horizon'])

    if st.button('Run Optimization'):
        with st.spinner('Running Optimization ...'):
            if not validate_chip_selection(chips['wc_gw'], chips['fh_gw'], chips['tc_gw'], chips['bb_gw']):
                st.warning('Two chips cannot be used in the same GW')
            else:
                run_optimization(params, chips, st.session_state.fpl_team_id, start, projection_data)
