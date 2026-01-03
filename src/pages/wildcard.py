import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.pages.vanilla import PROJECTIONS_PATH, display_metrics, display_team_visualization, get_basic_parameters
from src.team_optimization import TeamOptimization
from src.utils import get_next_gameweek


def get_wildcard_parameters() -> dict:
    params = {}

    with st.expander('Wildcard', expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            params['decay_long'] = st.slider(
                'Longterm WC Decay rate', min_value=0.0, max_value=1.0, value=0.9, step=0.02
            )
        with col2:
            params['decay_med'] = st.slider(
                'Medium-term WC Decay rate', min_value=0.0, max_value=1.0, value=0.8, step=0.02
            )
        with col3:
            params['decay_short'] = st.slider(
                'Short-term WC Decay rate', min_value=0.0, max_value=1.0, value=0.7, step=0.02
            )

    return params


def run_optimization(
    basic_params: dict, wildcard_params: dict, team_id: int, start: int, projection_data: pd.DataFrame
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

    df, chip_strat, total_ev, total_obj = to.advanced_wildcard(
        {
            'objective_type': 'decay',
            'decay_gameweek': [
                wildcard_params['decay_short'],
                wildcard_params['decay_med'],
                wildcard_params['decay_long'],
            ],
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

    display_metrics(total_ev, total_obj)
    display_team_visualization(to.initial_team_df, df, chip_strat, basic_params['horizon'])


def write() -> None:
    st.title('FPL - Wildcard Model')

    plt.style.use('.streamlit/style.mplstyle')
    start = get_next_gameweek()

    csv_path = PROJECTIONS_PATH / f'enriched_expected_points_GW{start}.csv'
    projection_data = pd.read_csv(csv_path)

    max_horizon = len([col for col in projection_data.columns if 'GW' in col])

    basic_params = get_basic_parameters(max_horizon)
    wildcard_params = get_wildcard_parameters()

    if st.button('Run Optimization'):
        with st.spinner('Running Optimization ...'):
            run_optimization(basic_params, wildcard_params, st.session_state.fpl_team_id, start, projection_data)
