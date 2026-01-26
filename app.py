import streamlit as st

from src.pages import (
    all_in_one,
    automated_chips,
    biased,
    differential,
    home,
    projection_data,
    select_chips,
    sensitivity_analysis,
    vanilla,
    wildcard,
)
from src.utils import load_team_id

st.set_page_config(page_title='FPL Optimization', page_icon='chart_with_upwards_trend')

PAGES = {
    'Home': home,
    'Projection Data': projection_data,
    'Vanilla': vanilla,
    'Differential': differential,
    'Biased': biased,
    'Wildcard': wildcard,
    'Select Chips': select_chips,
    'Automated Chips': automated_chips,
    'Sensitivity Analysis': sensitivity_analysis,
    'All In One': all_in_one,
}


def _display_team_id_in_sidebar() -> None:
    st.sidebar.markdown('---')
    st.sidebar.subheader('⚙️ Settings')

    if 'fpl_team_id' not in st.session_state:
        loaded_team_id = load_team_id()
        st.session_state.fpl_team_id = str(loaded_team_id)

    team_id_input = st.sidebar.text_input(
        'FPL Team ID',
        value=st.session_state.fpl_team_id,
        help="""
        Your Fantasy Premier League team ID
            **Finding your Team ID:**
            - Go to the FPL website and click on "Points" or "Transfers"
            - Look at the URL: `https://fantasy.premierleague.com/entry/XXXXXX/event/X`
            - The number after `/entry/` is your Team ID
        """,
        placeholder='e.g., 516095',
    )

    if team_id_input and team_id_input.isdigit() and int(team_id_input) > 0:
        st.session_state.fpl_team_id = team_id_input
        st.sidebar.info(f'ℹ️ Selected team ID : {team_id_input}')  # noqa: RUF001


def main() -> None:
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio('Visit', list(PAGES.keys()))

    _display_team_id_in_sidebar()

    page = PAGES[selection]

    with st.spinner(f'Loading {selection} ...'):
        page.write()


if __name__ == '__main__':
    main()
