import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.pages.vanilla import PROJECTIONS_PATH, display_metrics, display_team_visualization, get_basic_parameters
from src.team_optimization import TeamOptimization
from src.utils import get_next_gameweek


def get_bias_parameters(start: int, max_gws: int, player_names: pd.Series) -> dict:  # noqa: PLR0915
    # TODO: debug, it seems like the selected player is not properly passed to the optimization
    params = {}

    with st.expander('Bias', expanded=True):
        if max_gws == 3:  # noqa: PLR2004
            col1, col2, col3 = st.columns(3)
            with col1:
                params['team_in_1'] = st.multiselect(f'Forced in Team GW {start}', player_names.values)
            with col2:
                params['team_in_2'] = st.multiselect(f'In GW {start + 1}', player_names.values)
            with col3:
                params['team_in_3'] = st.multiselect(f'In GW {start + 2}', player_names.values)

            col1, col2, col3 = st.columns(3)
            with col1:
                params['team_out_1'] = st.multiselect(f'Forced out Team GW {start}', player_names.values)
            with col2:
                params['team_out_2'] = st.multiselect(f'Out GW {start + 1}', player_names.values)
            with col3:
                params['team_out_3'] = st.multiselect(f'Out GW {start + 2}', player_names.values)

            col1, col2, col3 = st.columns(3)
            with col1:
                params['hit_1'] = st.slider(f'Maximim hits in GW {start}', min_value=0, max_value=5, value=5)
            with col2:
                params['hit_2'] = st.slider(f'GW {start + 1}', min_value=0, max_value=5, value=5)
            with col3:
                params['hit_3'] = st.slider(f'GW {start + 2}', min_value=0, max_value=5, value=5)

            params['rolling'] = st.multiselect('Rolling transfers', np.arange(start + 1, start + 3))

        elif max_gws == 2:  # noqa: PLR2004
            col1, col2 = st.columns(2)
            with col1:
                params['team_in_1'] = st.multiselect(f'Forced in Team GW {start}', player_names.values)
            with col2:
                params['team_in_2'] = st.multiselect(f'In GW {start + 1}', player_names.values)
            params['team_in_3'] = []

            col1, col2 = st.columns(2)
            with col1:
                params['team_out_1'] = st.multiselect(f'Forced out Team GW {start}', player_names.values)
            with col2:
                params['team_out_2'] = st.multiselect(f'Out GW {start + 1}', player_names.values)
            params['team_out_3'] = []

            col1, col2 = st.columns(2)
            with col1:
                params['hit_1'] = st.slider(f'Maximim hits in GW {start}', min_value=0, max_value=5, value=5)
            with col2:
                params['hit_2'] = st.slider(f'GW {start + 1}', min_value=0, max_value=5, value=5)
            params['hit_3'] = []

            params['rolling'] = st.multiselect('Rolling transfers', np.arange(start + 1, start + 2))

        else:  # max_gws == 1
            params['team_in_1'] = st.multiselect(f'Forced in Team GW {start}', player_names.values)
            params['team_in_2'], params['team_in_3'] = [], []

            params['team_out_1'] = st.multiselect(f'Forced out Team GW {start}', player_names.values)
            params['team_out_2'], params['team_out_3'] = [], []

            params['hit_1'] = st.slider(f'Maximim hits in GW {start}', min_value=0, max_value=5, value=5)
            params['hit_2'], params['hit_3'] = [], []
            params['rolling'] = []

    return params


def run_optimization(  # noqa: PLR0913
    basic_params: dict,
    bias_params: dict,
    team_id: int,
    start: int,
    projection_data: pd.DataFrame,
    player_names: pd.Series,
    max_gws: int,
) -> None:
    to = TeamOptimization(
        {
            'filter_ev': None,
            'horizon': basic_params['horizon'],
            'noise': False,
            'ownership': False,
            'predictions': projection_data,
            'start': start,
            'team_id': team_id,
        }
    )

    to.build_model(
        {
            'model_name': 'biased',
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

    # Prepare hit limit based on max_gws
    if max_gws == 3:  # noqa: PLR2004
        hit_limit = [
            (start, bias_params['hit_1']),
            (start + 1, bias_params['hit_2']),
            (start + 2, bias_params['hit_3']),
        ]
    elif max_gws == 2:  # noqa: PLR2004
        hit_limit = [(start, bias_params['hit_1']), (start + 1, bias_params['hit_2'])]
    else:
        hit_limit = [(start, bias_params['hit_1'])]

    to.biased_model(
        love={
            'buy': {},
            'start': {},
            'team': (
                [(player[0], start) for player in player_names[player_names.isin(bias_params['team_in_1'])].items()]
                + [
                    (player[0], start + 1)
                    for player in player_names[player_names.isin(bias_params['team_in_2'])].items()
                ]
                + [
                    (player[0], start + 2)
                    for player in player_names[player_names.isin(bias_params['team_in_3'])].items()
                ]
            ),
            'cap': {},
        },
        hate={
            'sell': {},
            'team': (
                [
                    (player[0], start)
                    for player in player_names[player_names.isin(bias_params['team_out_1'])].items()
                ]
                + [
                    (player[0], start + 1)
                    for player in player_names[player_names.isin(bias_params['team_out_2'])].items()
                ]
                + [
                    (player[0], start + 2)
                    for player in player_names[player_names.isin(bias_params['team_out_3'])].items()
                ]
            ),
            'bench': {},
        },
        hit_limit={'max': hit_limit, 'eq': {}, 'min': {}},
        two_ft_gw=bias_params['rolling'],
    )

    df, chip_strat, total_ev, total_obj = to.solve(model_name='biased', log=True, time_lim=0)

    display_metrics(total_ev, total_obj)
    display_team_visualization(to.initial_team_df, df, chip_strat, basic_params['horizon'])


def write() -> None:
    st.title('FPL - Biased Model')

    plt.style.use('.streamlit/style.mplstyle')
    start = get_next_gameweek()

    csv_path = PROJECTIONS_PATH / f'enriched_expected_points_GW{start}.csv'
    projection_data = pd.read_csv(csv_path)

    max_horizon = len([col for col in projection_data.columns if 'GW' in col])

    basic_params = get_basic_parameters(max_horizon)

    bias_params = get_bias_parameters(start, basic_params['horizon'], projection_data.Name)

    if st.button('Run Optimization'):
        with Path('info.json').open() as f:
            info = json.load(f)
            team_id = info['team-id']

        with st.spinner('Running Optimization ...'):
            run_optimization(
                basic_params, bias_params, team_id, start, projection_data, projection_data.Name, max_horizon
            )
