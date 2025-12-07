import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.process_projection_data import convert_txt_to_csv
from src.utils import get_next_gw

PROJECTIONS_PATH = Path('data/projections')
POSSIBLE_POSITIONS = ['GK', 'DF', 'MD', 'FW']


@st.cache_data
def _load_projection_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    price_of_unpurchasable_player = 99.9
    return df.loc[df['PRICE'] != price_of_unpurchasable_player]


def _check_existing_data(next_gameweek: int | None) -> tuple[bool, Path | None]:
    if not next_gameweek:
        return False, None

    csv_path = PROJECTIONS_PATH / f'expected_points_GW{next_gameweek}.csv'
    return csv_path.exists(), csv_path


def _display_upload_tab(next_gameweek: int | None, has_existing_data: bool) -> None:  # noqa: FBT001
    with st.expander('📖 Instructions', expanded=False):
        st.markdown("""
        ### How to Use
        1. Copy the Free projection data from FPL Review
        2. Paste it in the text area below
        3. Click "Process Data" to save the data
        """)

    if has_existing_data:
        st.info(
            f'💡Projection data is already available for GW{next_gameweek}. Upload new projection data to update it.'
        )

    projection_data = st.text_area(
        'Paste Projection Data Here',
        height=400,
        placeholder="""
        Example format:\n\txMins\tGW1\tGW2\tGW3\tGW4\tTotal\t/£M\tElite%\n\n
        Haaland\nFW 14.9\n\t88\t5.6\t6.4\t5.1\t7.0\t24.0\t0.40\t97.8%""",
    )

    if st.button('🚀 Process Data', type='primary', width='stretch'):
        _process_uploaded_data(projection_data, next_gameweek)


def _process_uploaded_data(projection_data: str, next_gameweek: int) -> None:
    if not projection_data.strip():
        st.error('❌ Please paste some data before processing!')
        return

    try:
        raw_file = PROJECTIONS_PATH / f'raw_expected_points_GW{next_gameweek}.txt'
        csv_file = PROJECTIONS_PATH / f'expected_points_GW{next_gameweek}.csv'

        with raw_file.open('w') as f:
            f.write(projection_data)
        st.success(f'✅ Raw data saved to: `{raw_file}`')

        convert_txt_to_csv(str(raw_file), str(csv_file))
        st.success(f'✅ Processed data saved to: `{csv_file}`')

        time.sleep(3)

        _load_projection_data.clear()
        st.session_state.data_uploaded = True
        st.rerun()

    except Exception as e:  # noqa: BLE001
        st.error(f'❌ Error processing data: {e}')


def _apply_filters(
    df: pd.DataFrame, position_filter: list, team_filter: list | None, price_range: tuple
) -> pd.DataFrame:
    filtered_df = df[df['POSITION'].isin(position_filter)]

    if team_filter and 'Team' in df.columns:
        filtered_df = filtered_df[filtered_df['Team'].isin(team_filter)]

    return filtered_df[(filtered_df['PRICE'] >= price_range[0]) & (filtered_df['PRICE'] <= price_range[1])]


def _display_full_data_tab(df: pd.DataFrame) -> None:
    st.markdown('### All Player Data')

    col1, col2, col3 = st.columns(3)

    with col1:
        position_filter = st.multiselect('Filter by Position', options=POSSIBLE_POSITIONS, default=POSSIBLE_POSITIONS)

    with col2:
        # TODO: Verify 'Team' column existence
        teams = sorted(df['Team'].unique()) if 'Team' in df.columns else []
        team_filter = st.multiselect('Filter by Team', options=teams, default=teams) if teams else None

    with col3:
        price_range = st.slider(
            'Price Range (£M)',
            min_value=float(df['PRICE'].min()),
            max_value=float(df['PRICE'].max()),
            value=(float(df['PRICE'].min()), float(df['PRICE'].max())),
        )

    filtered_df = _apply_filters(df, position_filter, team_filter, price_range)

    display_df = filtered_df.sort_values('Total', ascending=False) if 'Total' in filtered_df.columns else filtered_df
    st.dataframe(display_df, width='stretch', height=400)


def _display_top_players_tab(df: pd.DataFrame) -> None:
    st.markdown('### Top Players by Position')

    gw_columns = [col for col in df.columns if col.startswith('GW')]
    if gw_columns:
        gw_columns = sorted(gw_columns, key=lambda x: int(x[2:]))
        next_gameweek_column = gw_columns[0]

    sort_options = ['Total']
    if next_gameweek_column:
        sort_options.insert(0, next_gameweek_column)

    sort_by = st.radio(
        'Rank players by:',
        options=sort_options,
        index=0 if next_gameweek_column else sort_options.index('Total'),
        horizontal=True,
        help=f'Select whether to rank by {next_gameweek_column} or total points across all gameweeks',
    )

    for position in POSSIBLE_POSITIONS:
        st.markdown(f'#### {position}')
        pos_df = df[df['POSITION'] == position]

        display_columns = ['NAME', 'PRICE', next_gameweek_column, 'Total', '/£M']
        display_columns = [col for col in display_columns if col in pos_df.columns]
        top_players = pos_df.nlargest(5, sort_by)[display_columns]
        st.dataframe(top_players, hide_index=True, width='stretch')


def _display_existing_data(csv_path: Path, next_gameweek: int | None) -> None:
    just_uploaded = st.session_state.get('data_uploaded', False)

    mod_time = csv_path.stat().st_mtime
    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))

    if just_uploaded:
        st.success(f'✅ Data available for GW{next_gameweek} & Last updated: {formatted_time} (just now)')
        st.session_state.data_uploaded = False
    else:
        st.success(f'✅ Data available for GW{next_gameweek} & Last updated: {formatted_time}')

    try:
        df = _load_projection_data(csv_path)

        tab1, tab2, tab3 = st.tabs(['📤 Upload Data', '📊 All Data', '🏆 Top Players'])

        with tab1:
            _display_upload_tab(next_gameweek, has_existing_data=True)

        with tab2:
            _display_full_data_tab(df)

        with tab3:
            _display_top_players_tab(df)

    except Exception as e:  # noqa: BLE001
        st.error(f'❌ Error loading existing data: {e}')


def write() -> None:
    st.title('FPL - Projection Data')

    PROJECTIONS_PATH.mkdir(parents=True, exist_ok=True)

    try:
        next_gameweek = get_next_gw()
    except Exception as e:  # noqa: BLE001
        st.error(f'Error fetching current gameweek: {e}')
        next_gameweek = None

    has_existing_data, csv_path = _check_existing_data(next_gameweek)

    if has_existing_data:
        _display_existing_data(csv_path, next_gameweek)
    else:
        st.info('No projection data found. Please upload data below.')
        _display_upload_tab(next_gameweek, has_existing_data=False)


if __name__ == '__main__':
    write()
