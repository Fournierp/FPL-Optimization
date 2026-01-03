import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.pages.vanilla import PROJECTIONS_PATH, display_metrics, display_team_visualization, get_basic_parameters
from src.team_optimization import TeamOptimization
from src.utils import get_next_gameweek


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
        with st.spinner('Running Optimization ...'):
            run_optimization(basic_params, diff_params, st.session_state.fpl_team_id, start, projection_data)
