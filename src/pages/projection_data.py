import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_collection import convert_txt_to_csv, match_player_names
from src.utils import get_next_gameweek

PROJECTIONS_PATH = Path('data/projections')
POSSIBLE_POSITIONS = ['GK', 'DF', 'MD', 'FW']


@st.cache_data
def _load_projection_data(csv_path: Path, overrides: dict | None = None) -> pd.DataFrame:
    projection_data = pd.read_csv(csv_path)
    price_of_unpurchasable_player = 99.9
    projection_data = projection_data.loc[projection_data['PRICE'] != price_of_unpurchasable_player]

    if overrides:
        projection_data = _apply_overrides_to_dataframe(projection_data, overrides)

    return projection_data


def _check_existing_data(next_gameweek: int | None) -> tuple[bool, Path | None]:
    if not next_gameweek:
        return False, None

    csv_path = PROJECTIONS_PATH / f'enriched_expected_points_GW{next_gameweek}.csv'
    return csv_path.exists(), csv_path


def _save_overrides(overrides: dict, next_gameweek: int) -> None:
    overrides_file = PROJECTIONS_PATH / f'overrides_GW{next_gameweek}.json'
    with overrides_file.open('w') as f:
        json.dump(overrides, f, indent=2)


def _load_overrides(next_gameweek: int) -> dict:
    overrides_file = PROJECTIONS_PATH / f'overrides_GW{next_gameweek}.json'
    if overrides_file.exists():
        with overrides_file.open('r') as f:
            return json.load(f)
    return {}


def _display_upload_tab(next_gameweek: int | None, has_existing_data: bool) -> None:  # noqa: FBT001
    with st.expander('📖 Instructions', expanded=False):
        st.markdown("""
        ### How to Use
        (Optional) Change the Team ID in the sidebar settings
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

    fpl_team_id = st.session_state.get('fpl_team_id', '')

    if st.button('🚀 Process Data', type='primary', width='stretch'):
        _process_uploaded_data(projection_data, next_gameweek, fpl_team_id)


def _process_uploaded_data(projection_data: str, next_gameweek: int, team_id: int | None = None) -> None:
    if not projection_data.strip():
        st.error('❌ Please paste some data before processing!')
        return

    raw_file = PROJECTIONS_PATH / f'raw_expected_points_GW{next_gameweek}.txt'
    csv_file = PROJECTIONS_PATH / f'expected_points_GW{next_gameweek}.csv'
    enriched_file = PROJECTIONS_PATH / f'enriched_expected_points_GW{next_gameweek}.csv'

    with raw_file.open('w') as f:
        f.write(projection_data)
    st.success(f'✅ Raw data saved to: `{raw_file}`')

    convert_txt_to_csv(str(raw_file), str(csv_file))
    st.success(f'✅ Projection data saved to: `{csv_file}`')

    with st.spinner('🔗 Matching players with FPL API...'):
        projection_data = _load_projection_data(csv_file)
        enriched_data = match_player_names(projection_data, team_id, next_gameweek)
        enriched_data = enriched_data.rename(columns={'NAME': 'Name', 'POSITION': 'Position'})
        enriched_data.to_csv(str(enriched_file), index=False)

    time.sleep(2)

    _load_projection_data.clear()
    st.session_state.data_uploaded = True
    st.rerun()


def _apply_filters(
    projection_data: pd.DataFrame, position_filter: list, team_filter: list | None, price_range: tuple
) -> pd.DataFrame:
    filtered_df = projection_data[projection_data['Position'].isin(position_filter)]

    filtered_df = filtered_df[filtered_df['Team'].isin(team_filter)]

    return filtered_df[(filtered_df['PRICE'] >= price_range[0]) & (filtered_df['PRICE'] <= price_range[1])]


def _display_full_data_tab(projection_data: pd.DataFrame) -> None:
    st.markdown('### All Player Data')
    projection_data = projection_data[
        ['Name', 'Position', 'Team', 'PRICE', 'xMins']
        + [col for col in projection_data.columns if 'GW' in col]
        + ['Total', '/£M']
    ]

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        position_filter = st.multiselect('Filter by Position', options=POSSIBLE_POSITIONS, default=POSSIBLE_POSITIONS)

    projection_data['Team'] = projection_data['Team'].astype(str)
    with col2:
        teams = sorted(projection_data['Team'].unique()) if 'Team' in projection_data.columns else []
        team_filter = st.multiselect('Filter by Team', options=teams, default=teams) if teams else None

    with col3:
        price_range = st.slider(
            'Price Range (£M)',
            min_value=float(projection_data['PRICE'].min()),
            max_value=float(projection_data['PRICE'].max()),
            value=(float(projection_data['PRICE'].min()), float(projection_data['PRICE'].max())),
        )

    filtered_df = _apply_filters(projection_data, position_filter, team_filter, price_range)

    display_df = filtered_df.sort_values('Total', ascending=False)

    st.dataframe(display_df, width='stretch', height=400)


def _display_player_info(player_data: pd.Series) -> None:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('**Player Info**')
        info_cols = ['Position', 'PRICE', 'Team']
        cols_info = st.columns(len(info_cols))
        for idx, col in enumerate(info_cols):
            if col in player_data.index:
                with cols_info[idx]:
                    st.metric(label=col.title(), value=player_data[col])

    with col2:
        if 'in_my_team' in player_data.index and player_data['in_my_team']:
            st.success('✓ In Your Team')


def _display_override_inputs(player_data: pd.Series, selected_player: str, gw_columns: list) -> dict:
    st.markdown('---')
    st.markdown('**Override Values**')

    current_overrides = st.session_state.player_overrides.get(selected_player, {})
    override_columns = [*gw_columns]
    cols = st.columns(len(override_columns))

    new_overrides = {}
    for idx, col_name in enumerate(override_columns):
        with cols[idx]:
            current_val = player_data.get(col_name, 0)
            override_val = current_overrides.get(col_name, None)

            st.caption(f'{col_name}')
            st.text(f'Current: {current_val:.1f}')

            new_val = st.number_input(
                'Override',
                min_value=0.0,
                max_value=99.9,
                value=float(override_val) if override_val is not None else float(current_val),
                step=0.1,
                key=f'override_{selected_player}_{col_name}',
                label_visibility='collapsed',
            )

            if new_val != current_val:
                new_overrides[col_name] = new_val

    return new_overrides


def _display_player_actions(selected_player: str, new_overrides: dict) -> None:
    col_button1, col_button2 = st.columns(2)

    with col_button1:
        if st.button('✅ Confirm Overrides', type='primary', use_container_width=True):
            if new_overrides:
                st.session_state.player_overrides[selected_player] = new_overrides
                st.success(f'✅ Confirmed overrides for {selected_player}')
            else:
                st.info('No changes detected')

    with col_button2:
        if st.button('🔄 Reset', use_container_width=True) and selected_player in st.session_state.player_overrides:
            del st.session_state.player_overrides[selected_player]
            st.success(f'✅ Reset overrides for {selected_player}')


def _display_overrides_summary(next_gameweek: int) -> None:
    if not st.session_state.player_overrides:
        return

    st.markdown('---')
    st.markdown('### Current Overrides Summary')

    override_summary = []
    for player_name, overrides in st.session_state.player_overrides.items():
        override_summary.append({'Player': player_name, 'Overridden Columns': ', '.join(overrides.keys())})

    st.dataframe(pd.DataFrame(override_summary), hide_index=True, use_container_width=True)

    col_action1, col_action2 = st.columns(2)

    with col_action1:
        if st.button('💾 Save Overrides', type='primary', use_container_width=True):
            _save_overrides(st.session_state.player_overrides, next_gameweek)
            st.success('✅ Saved overrides! They will be applied automatically when loading data.')
            _load_projection_data.clear()

    with col_action2:
        if st.button('🗑️ Clear All Overrides', type='secondary', use_container_width=True):
            st.session_state.player_overrides = {}
            _save_overrides({}, next_gameweek)
            _load_projection_data.clear()
            st.success('✅ Cleared all overrides')


def _display_override_values_tab(projection_data: pd.DataFrame, next_gameweek: int) -> None:
    st.markdown('### Override Player Values')
    st.markdown('Adjust individual player projections to reflect your own analysis or information.')

    with st.expander('📖 Instructions', expanded=False):
        st.markdown("""
        ### How to Override Player Values

        **Purpose:** Modify projected points for specific players based on your own analysis,
        injury news, or other factors.

        **Steps:**
        1. **Select a Player** from the dropdown menu
        2. **Adjust Values** for specific gameweeks by entering new projected points
        3. **Confirm Overrides** to save changes for that player (not applied to the projections yet)
        4. **Review Summary** below to see all pending overrides
        5. **Save Overrides** to file for future sessions
        """)

    gw_columns = [col for col in projection_data.columns if col.startswith('GW')]
    gw_columns = sorted(gw_columns, key=lambda x: int(x[2:])) if gw_columns else []

    if 'player_overrides' not in st.session_state:
        st.session_state.player_overrides = _load_overrides(next_gameweek)

    player_names = sorted(projection_data['Name'].tolist())
    selected_player = st.selectbox(
        'Select Player to Override',
        options=['', *player_names],
        help='Choose a player to modify their projected values',
    )

    if selected_player:
        player_data = projection_data[projection_data['Name'] == selected_player].iloc[0]
        _display_player_info(player_data)
        new_overrides = _display_override_inputs(player_data, selected_player, gw_columns)
        _display_player_actions(selected_player, new_overrides)

    _display_overrides_summary(next_gameweek)


def _apply_overrides_to_dataframe(projection_data: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    projection_data_modified = projection_data.copy()

    for player_name, player_overrides in overrides.items():
        player_idx = projection_data_modified[projection_data_modified['Name'] == player_name].index
        if len(player_idx) > 0:
            idx = player_idx[0]
            for col_name, new_value in player_overrides.items():
                if col_name in projection_data_modified.columns:
                    projection_data_modified.loc[idx, col_name] = new_value

    return projection_data_modified


def _display_existing_data(csv_path: Path, next_gameweek: int | None) -> None:
    just_uploaded = st.session_state.get('data_uploaded', False)

    mod_time = csv_path.stat().st_mtime
    formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))

    if just_uploaded:
        st.success(f'✅ Data available for GW{next_gameweek} & Last updated: {formatted_time} (just now)')
        st.session_state.data_uploaded = False
    else:
        st.success(f'✅ Data available for GW{next_gameweek} & Last updated: {formatted_time}')

    overrides = _load_overrides(next_gameweek)
    projection_data = _load_projection_data(csv_path, overrides)

    if overrides:
        st.info(f'ℹ️ {len(overrides)} player override(s) applied to the data')  # noqa: RUF001

    tab1, tab2, tab3 = st.tabs(['📤 Upload Data', '📊 All Data', '✏️ Override Values'])

    with tab1:
        _display_upload_tab(next_gameweek, has_existing_data=True)

    with tab2:
        _display_full_data_tab(projection_data)

    with tab3:
        _display_override_values_tab(projection_data, next_gameweek)


def write() -> None:
    st.title('FPL - Projection Data')

    PROJECTIONS_PATH.mkdir(parents=True, exist_ok=True)

    next_gameweek = get_next_gameweek()
    has_existing_data, csv_path = _check_existing_data(next_gameweek)

    if has_existing_data:
        _display_existing_data(csv_path, next_gameweek)
    else:
        st.info('No projection data found. Please upload data below.')
        _display_upload_tab(next_gameweek, has_existing_data=False)


if __name__ == '__main__':
    write()
