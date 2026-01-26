import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.pages.select_chips import display_team_visualization
from src.pages.vanilla import PROJECTIONS_PATH, display_metrics, get_basic_parameters
from src.team_optimization import TeamOptimization
from src.utils import get_next_gameweek


def get_chip_value_parameters() -> dict:
    with st.expander('Chips', expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            wc_val = st.slider('Wildcard Added Value', min_value=0, max_value=25, value=18)
        with col2:
            fh_val = st.slider('Freehit AV', min_value=0, max_value=25, value=18)
        with col3:
            tc_val = st.slider('Triple Captain AV', min_value=0, max_value=25, value=12)
        with col4:
            bb_val = st.slider('Bench Boost AV', min_value=0, max_value=25, value=14)

    return {'wc_val': wc_val, 'fh_val': fh_val, 'tc_val': tc_val, 'bb_val': bb_val}


def run_optimization(params: dict, chip_vals: dict, team_id: int, start: int, projection_data: pd.DataFrame) -> None:
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

    to.automated_chips_model(
        {
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
            'triple_val': chip_vals['tc_val'],
            'bboost_val': chip_vals['bb_val'],
            'freehit_val': chip_vals['fh_val'],
            'wildcard_val': chip_vals['wc_val'],
        }
    )

    df, chip_strat, total_ev, total_obj = to.solve(model_name='automated_chips', log=True, time_lim=0)

    display_metrics(total_ev, total_obj)
    display_team_visualization(df, chip_strat, to, params['horizon'])


def write() -> None:
    st.title('FPL - Automated Chips Model')

    with st.expander('📖 Instructions', expanded=False):
        st.markdown(
            """
            **Automated Chips Optimization** - Let the optimizer decide when to use chips based on added value.

            Instead of manually choosing chip timing, this model evaluates the value each chip adds in each
            gameweek and automatically schedules them for maximum impact.

            **Chip Value Thresholds:**
            - Set minimum expected point gains required for each chip to be activated
            - Higher thresholds = more conservative chip usage
            """
        )

    plt.style.use('.streamlit/style.mplstyle')
    start = get_next_gameweek()

    csv_path = PROJECTIONS_PATH / f'enriched_expected_points_GW{start}.csv'
    projection_data = pd.read_csv(csv_path)

    max_horizon = len([col for col in projection_data.columns if 'GW' in col])

    params = get_basic_parameters(max_horizon)

    chip_vals = get_chip_value_parameters()

    if st.button('Run Optimization'):
        with st.spinner('Running Optimization ...'):
            run_optimization(params, chip_vals, st.session_state.fpl_team_id, start, projection_data)
