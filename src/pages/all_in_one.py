import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.pages.biased import get_bias_parameters
from src.pages.select_chips import display_team_visualization, get_chip_parameters
from src.pages.vanilla import PROJECTIONS_PATH, display_metrics, get_basic_parameters
from src.team_optimization import TeamOptimization
from src.utils import get_next_gameweek


def get_differential_parameters() -> dict:
    with st.expander('Differential'):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            do_diff = st.checkbox('Enable Differentials', value=True)
        with col2:
            nb_diff = st.slider('# Differentials', min_value=0, max_value=11, value=3, step=1)
        with col3:
            threshold = st.slider('Threshold', min_value=1, max_value=100, value=10, step=1)
        with col4:
            rank = st.selectbox(
                'Rank', ['Top_100', 'Top_1K', 'Top_10K', 'Top_50K', 'Top_100K', 'Top_250K', 'Top_500K'], 4
            )

    return {'do_diff': do_diff, 'nb_diff': nb_diff, 'threshold': threshold, 'rank': rank}


def run_optimization(  # noqa: PLR0913
    params: dict,
    differential: dict,
    chips: dict,
    bias: dict,
    team_id: int,
    start: int,
    projection_data: pd.DataFrame,
    player_names: pd.Series,
) -> None:
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

    # Build base model with chips
    to.build_model(
        {
            'model_name': 'all_in_one',
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

    # Add differential constraints
    if differential['do_diff']:
        to.differential_model(
            nb_differentials=differential['nb_diff'], threshold=differential['threshold'], target=differential['rank']
        )

    # Build hit limit list
    if params['horizon'] == 3:  # noqa: PLR2004
        hit_limit = [
            (start, bias['hit_limits'][0]),
            (start + 1, bias['hit_limits'][1]),
            (start + 2, bias['hit_limits'][2]),
        ]
    elif params['horizon'] == 2:  # noqa: PLR2004
        hit_limit = [(start, bias['hit_limits'][0]), (start + 1, bias['hit_limits'][1])]
    else:
        hit_limit = [(start, bias['hit_limits'][0])]

    # Add bias constraints
    to.biased_model(
        love={
            'buy': {},
            'start': {},
            'team': (
                [(player[0], start) for player in player_names[player_names.isin(bias['team_in'][0])].items()]
                + [(player[0], start + 1) for player in player_names[player_names.isin(bias['team_in'][1])].items()]
                + [(player[0], start + 2) for player in player_names[player_names.isin(bias['team_in'][2])].items()]
            ),
            'cap': {},
        },
        hate={
            'sell': {},
            'team': (
                [(player[0], start) for player in player_names[player_names.isin(bias['team_out'][0])].items()]
                + [
                    (player[0], start + 1)
                    for player in player_names[player_names.isin(bias['team_out'][1])].items()
                ]
                + [
                    (player[0], start + 2)
                    for player in player_names[player_names.isin(bias['team_out'][2])].items()
                ]
            ),
            'bench': {},
        },
        hit_limit={'max': hit_limit, 'eq': {}, 'min': {}},
        two_ft_gw=bias['rolling'],
    )

    # Solve and display
    df, chip_strat, total_ev, total_obj = to.solve(model_name='all_in_one', log=True, time_lim=0)

    display_metrics(total_ev, total_obj)
    display_team_visualization(df, chip_strat, to, params['horizon'])


def write() -> None:
    st.title('FPL - All In One Model')

    plt.style.use('.streamlit/style.mplstyle')
    start = get_next_gameweek()

    csv_path = PROJECTIONS_PATH / f'enriched_expected_points_GW{start}.csv'
    projection_data = pd.read_csv(csv_path)

    max_horizon = len([col for col in projection_data.columns if 'GW' in col])

    params = get_basic_parameters(max_horizon)

    differential = get_differential_parameters()
    chips = get_chip_parameters(start, params['horizon'])
    bias = get_bias_parameters(start, params['horizon'], projection_data.Name)

    if st.button('Run Optimization'):
        with Path('info.json').open() as f:
            info = json.load(f)
            team_id = info['team-id']

        with st.spinner('Running Optimization ...'):
            run_optimization(params, differential, chips, bias, team_id, start, projection_data, projection_data.Name)
