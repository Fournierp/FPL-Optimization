import contextlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from highlight_text import fig_text
from matplotlib import patches
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from src.pages.select_chips import get_chip_parameters
from src.pages.vanilla import PROJECTIONS_PATH, get_basic_parameters
from src.team_optimization import TeamOptimization
from src.utils import get_next_gameweek


def get_analysis_parameters() -> dict:
    with st.expander('Parameters', expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            repeats = st.slider('Number of Experiments', min_value=1, max_value=25, value=5)
        with col2:
            iterations = st.slider('Iterations per exp.', min_value=1, max_value=25, value=7)

    return {'repeats': repeats, 'iterations': iterations}


def load_analysis_data(season: int) -> tuple:
    player_names = pd.read_csv(
        f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/player_idlist.csv'
    ).set_index('id')

    with Path('tmp/hashes.json').open('r') as f:
        hashes = json.load(f)

    df = pd.read_csv('tmp/podium.csv')

    return player_names, hashes, df


def process_results_dataframe(df: pd.DataFrame, hashes: dict, iterations: int) -> tuple:
    df['Total'] = df.apply(lambda x: sum(x[str(col)] for col in np.arange(1, iterations + 1)), axis=1)

    (df['Transfer in'], df['Transfer out']) = zip(
        *df['Unnamed: 0'].apply(lambda x: (hashes[str(x)][0], hashes[str(x)][1])), strict=False
    )

    max_cols = [col for col in df.columns if 'EV_' in col]
    evs = np.unique(df[max_cols].values)

    df['Mean'] = df.apply(lambda x: -sum(x[col] if pd.notna(x[col]) else 0 for col in max_cols) / x['Total'], axis=1)
    df['Std'] = df.apply(
        lambda x: np.sqrt(
            np.sum(np.power([x[col] + x['Mean'] if pd.notna(x[col]) else 0 for col in max_cols], 2)) / x['Total']
        ),
        axis=1,
    )

    df[['Total', '1', '2', '3']] = df[['Total', '1', '2', '3']].astype('int32')

    return df, max_cols, evs


def display_percentages_tables(percent: pd.DataFrame) -> None:
    with st.expander('Percentages', expanded=False):
        st.write('Goalkeepers')
        st.dataframe(
            percent.loc[percent.Position == 'GK']
            .sort_values(by=['Appearences', 'Mean'], ascending=[False, False])[['Player', 'Appearences', 'Mean', 'Std']]
            .reset_index(drop=True)
        )

        st.write('Defenders')
        st.dataframe(
            percent.loc[percent.Position == 'DF']
            .sort_values(by=['Appearences', 'Mean'], ascending=[False, False])[['Player', 'Appearences', 'Mean', 'Std']]
            .reset_index(drop=True)
        )

        st.write('Midfielders')
        st.dataframe(
            percent.loc[percent.Position == 'MD']
            .sort_values(by=['Appearences', 'Mean'], ascending=[False, False])[['Player', 'Appearences', 'Mean', 'Std']]
            .reset_index(drop=True)
        )

        st.write('Forwards')
        st.dataframe(
            percent.loc[percent.Position == 'FW']
            .sort_values(by=['Appearences', 'Mean'], ascending=[False, False])[['Player', 'Appearences', 'Mean', 'Std']]
            .reset_index(drop=True)
        )


def enrich_player_data(percent: pd.DataFrame, season: int, start: int, xpts: pd.DataFrame) -> tuple:
    photos = pd.read_csv(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/players_raw.csv')[
        ['first_name', 'second_name', 'photo', 'team']
    ]

    percent['Team'] = percent.apply(
        lambda x: photos.loc[photos.first_name + ' ' + photos.second_name == x.Player]['team'].to_numpy()[0], axis=1
    )

    percent = pd.merge(
        percent,
        pd.read_csv(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/teams.csv')[['id', 'short_name']],
        left_on='Team',
        right_on='id',
    ).drop(['Team', 'id'], axis=1)

    # Add upcoming fixtures
    fixtures = pd.read_csv(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/fixtures.csv')
    fixtures = fixtures.loc[fixtures.event == start][['team_h', 'team_a']]
    fixtures = (
        pd.merge(
            fixtures,
            pd.read_csv(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/teams.csv')[['id', 'short_name']],
            left_on='team_h',
            right_on='id',
        )
        .drop(['team_h', 'id'], axis=1)
        .rename(columns={'short_name': 'team_h'})
    )

    fixtures = (
        pd.merge(
            fixtures,
            pd.read_csv(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/teams.csv')[['id', 'short_name']],
            left_on='team_a',
            right_on='id',
        )
        .drop(['team_a', 'id'], axis=1)
        .rename(columns={'short_name': 'team_a'})
    )

    def get_fixtures(x: str, y: pd.Series) -> str | None:
        if y.team_h == x:
            return y.team_a
        if y.team_a == x:
            return y.team_h.lower()
        return None

    percent['GW'] = percent['short_name'].apply(
        lambda x: ' + '.join(
            [game for game in fixtures.apply(lambda y: get_fixtures(x, y), axis=1).to_numpy() if game is not None]
        )
    )

    # Add xpts data
    percent = pd.merge(percent, xpts, left_on='Player', right_on='Player')

    return percent, photos


def draw_position_row(  # noqa: PLR0913
    ax,  # noqa: ANN001
    percent: pd.DataFrame,
    position: str,
    y_pos: float,
    num_players: int,
    x_offset: float,
    width: float,
    repeats: int,
    start: int,
    photos: pd.DataFrame,
    url: str,
    name_overrides: dict | None = None,
) -> None:
    df_pos = (
        percent.loc[percent.Position == position]
        .sort_values(by=['Appearences', 'Mean'], ascending=[False, False])
        .reset_index(drop=True)
        .head(num_players)
    )

    for j, row in df_pos.iterrows():
        rectangle = patches.Rectangle((x_offset + j * width, y_pos), 17, 0.25, facecolor='#656B73')
        ax.add_patch(rectangle)
        rx, ry = rectangle.get_xy()
        cx = rx + rectangle.get_width() / 2.0
        cy = ry + rectangle.get_height() / 2.0

        # Apply name overrides if provided
        name = row['Player']
        if name_overrides and name in name_overrides:
            name = name_overrides[name]
        elif '-' in name:
            name = name.split(' ')[-1]

        ax.annotate(
            f'{name}\n{row["GW"]}\n{np.round(row["Appearences"] / repeats * 100)} | xPts: {row[f"{start}_Pts"]}',
            (cx, cy),
            color='w',
            weight='bold',
            fontsize=11,
            ha='center',
            va='center',
        )

        # Plot the portrait
        imscatter(
            x=cx,
            y=y_pos + 0.25 + 0.31,
            image=url.format(
                index=(
                    photos.loc[photos.first_name + ' ' + photos.second_name == row['Player']]['photo'].to_numpy()[0][
                        :-4
                    ]
                )
            ),
            ax=ax,
            zoom=0.4,
        )


def display_freehit_visualization(percent: pd.DataFrame, repeats: int, start: int, season: int) -> None:
    photos = pd.read_csv(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/players_raw.csv')[
        ['first_name', 'second_name', 'photo', 'team']
    ]
    url = 'https://resources.premierleague.com/premierleague/photos/players/110x140/p{index}.png'

    fig, ax = plt.subplots(figsize=(12, 14))

    # Define name overrides for specific players
    name_overrides = {
        'Bruno Miguel Borges Fernandes': 'Bruno Fernandes',
        'Cristiano Ronaldo dos Santos Aveiro': 'Cristiano Ronaldo',
    }

    # Draw each position
    draw_position_row(ax, percent, 'GK', 3, 2, 25, 35, repeats, start, photos, url)
    draw_position_row(ax, percent, 'DF', 2, 5, 2.5, 20, repeats, start, photos, url)
    draw_position_row(ax, percent, 'MD', 1, 5, 2.5, 20, repeats, start, photos, url, name_overrides)
    draw_position_row(ax, percent, 'FW', 0, 3, 17.5, 25, repeats, start, photos, url, name_overrides)

    ax.set_ylim(0, 4.25)
    ax.set_xlim(0, 100)
    ax.axis('off')

    fig_text(
        x=0.14,
        y=0.855,
        s='<Players appearing most on randomized FH Team>',
        highlight_textprops=[{'fontweight': 'bold'}],
        fontsize=24,
        fontfamily='DejaVu Sans',
        color='w',
    )

    st.pyplot(fig, ax)
    plt.close(fig)


def display_transfer_analysis(df, max_cols, evs, player_names) -> None:  # noqa: ANN001
    df['Transfer'] = df[['Transfer in', 'Transfer out']].apply(lambda x: write_transfer(x, player_names), axis=1)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_ylim(0, 5.5)
    ax.set_xlim(0, 25 + (4 + 1) * 16 + 2.5)
    ax.axis('off')
    header_pos = 5.25

    for j, row in df.head(7)[::-1].reset_index().iterrows():
        rectangle = patches.Rectangle((0, j * 0.75), 25, 0.5, facecolor='#656B73')
        ax.add_patch(rectangle)
        rx, ry = rectangle.get_xy()
        cx = rx + rectangle.get_width() / 2.0
        cy = ry + rectangle.get_height() / 2.0
        ax.annotate(row['Transfer'], (cx, cy), color='w', weight='bold', fontsize=14, ha='center', va='center')

        newax = fig.add_axes([0.63, 0.11 + j * 0.105, 0.25, 0.07])
        newax.boxplot(
            [-row[col] for col in max_cols if pd.notna(row[col]) and row[col] != 0],
            vert=False,
            patch_artist=True,
            capprops={'color': '#656B73'},
            boxprops={'facecolor': '#656B73', 'color': '#656B73'},
            whiskerprops={'color': '#656B73'},
            flierprops={'markerfacecolor': 'r'},
            medianprops={'color': 'w'},
            widths=0.6,
        )

        newax.set_xlim(-np.max(evs[evs != 0]) - 1, -np.min(evs[evs != 0]) + 1)

    ax.text(cx, header_pos, 'Transfer', fontsize=16, weight='bold', ha='center')

    ax.plot([0, 25], [5.125, 5.125], ls='-', lw='2.5', c='grey')

    for i, col in enumerate(['Total', '1', '2', '3']):
        for j, row in df.head(7)[::-1].reset_index().iterrows():
            rectangle = patches.Rectangle((30 + i * 10, j * 0.75), 5, 0.5, facecolor='#656B73')
            ax.add_patch(rectangle)
            rx, ry = rectangle.get_xy()
            cx = rx + rectangle.get_width() / 2.0
            cy = ry + rectangle.get_height() / 2.0
            ax.annotate(row[col], (cx, cy), color='w', weight='bold', fontsize=14, ha='center', va='center')

        ax.text(cx, header_pos, col, fontsize=16, weight='bold', ha='center')

        ax.plot([30 + i * 10, 30 + i * 10 + 5], [5.125, 5.125], ls='-', lw='2.5', c='grey')

    st.pyplot(fig, ax)
    plt.close(fig)


def run_sensitivity_analysis(  # noqa: PLR0913
    params: dict,
    chips: dict,
    analysis: dict,
    start: int,
    team_id: int,
    season: int,
    xpts: pd.DataFrame,
    projection_data: pd.DataFrame,
) -> None:
    if chips['fh_gw'] == 0 and (params['horizon'] > 1 or analysis['iterations'] > 1):
        st.warning('Should not have more than 1 iteration or longer than 1 gw horizon')
        return
    if chips['fh_gw'] == 0 and (params['horizon'] > 1 or analysis['iterations'] > 1):
        st.warning('Should not have more than 1 iteration or longer than 1 gw horizon')
        return

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

    my_bar = st.progress(0)

    for progress, _ in enumerate(
        to.sensitivity_analysis(
            repeats=analysis['repeats'],
            iterations=analysis['iterations'],
            parameters={
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
            },
        ),
        start=1,
    ):
        my_bar.progress(progress / analysis['repeats'])

    # Load and process results
    player_names, hashes, df = load_analysis_data(season)
    df, max_cols, evs = process_results_dataframe(df, hashes, analysis['iterations'])

    if chips['fh_gw'] is not None or chips['wc_gw'] is not None:
        # Freehit/Wildcard analysis
        freehit_teams = pd.DataFrame(
            zip(*df['Transfer in'].apply(lambda x: write_freehit(x, player_names)), strict=False)
        ).T

        # Gather player appearance data
        percent = pd.DataFrame(columns=['Player', 'Position', 'Appearences'])
        percent['Player'] = np.unique(freehit_teams)
        player_pos = pd.read_csv(f'data/fpl_official/vaastav/data/{season}-{season % 2000 + 1}/cleaned_players.csv')[
            ['first_name', 'second_name', 'element_type']
        ]
        percent['Position'] = percent['Player'].apply(
            lambda x: player_pos.loc[player_pos.first_name + ' ' + player_pos.second_name == x][
                'element_type'
            ].to_numpy()[0]
        )
        percent['Appearences'] = percent['Player'].apply(lambda x: np.sum(np.sum(freehit_teams == x)))

        percent['Mean'] = percent.apply(
            lambda x: np.sum(np.sum(freehit_teams == x['Player'], axis=1) * df['Mean']) / x['Appearences'], axis=1
        )
        percent['Std'] = percent.apply(
            lambda x: np.sum(np.sum(freehit_teams == x['Player'], axis=1) * df['Std']) / x['Appearences'], axis=1
        )

        display_percentages_tables(percent)

        # Enrich and visualize
        xpts_enriched = pd.merge(xpts, player_names, left_index=True, right_index=True)
        xpts_enriched['Player'] = xpts_enriched.apply(lambda x: x.first_name + ' ' + x.second_name, axis=1)

        percent, _ = enrich_player_data(percent, season, start, xpts_enriched)
        display_freehit_visualization(percent, analysis['repeats'], start, season)

    else:
        # Transfer analysis
        display_transfer_analysis(df, max_cols, evs, player_names)


def write() -> None:
    st.title('FPL - Sensitivity Analysis Model')

    plt.style.use('.streamlit/style.mplstyle')
    start = get_next_gameweek()

    csv_path = PROJECTIONS_PATH / f'enriched_expected_points_GW{start}.csv'
    projection_data = pd.read_csv(csv_path)

    max_horizon = len([col for col in projection_data.columns if 'GW' in col])

    params = get_basic_parameters(max_horizon)
    chips = get_chip_parameters(start, params['horizon'])
    analysis = get_analysis_parameters()

    if st.button('Run Optimization'):
        with Path('info.json').open() as f:
            info = json.load(f)

            season = info['season']

        with st.spinner('Running Optimization ...'):
            run_sensitivity_analysis(
                params,
                chips,
                analysis,
                start,
                st.session_state.fpl_team_id,
                season,
                projection_data[[f'GW{start}']],
                projection_data,
            )


def write_transfer(x: dict, player_names: pd.DataFrame) -> str:
    if len(x['Transfer in']) == len(x['Transfer out']) == 0:
        return 'Roll Transfer'

    out_list = [player_names.loc[transfer_out]['second_name'] for transfer_out in x['Transfer out']]
    in_list = [player_names.loc[transfer_in]['second_name'] for transfer_in in x['Transfer in']]

    return ' + '.join(out_list) + ' -> ' + ' + '.join(in_list)


def write_freehit(x: dict, player_names: pd.DataFrame) -> list:
    return [
        player_names.loc[fh_player]['first_name'] + ' ' + player_names.loc[fh_player]['second_name'] for fh_player in x
    ]


def imscatter(x: float, y: float, image: str, ax=None, zoom: float = 1) -> list:  # noqa: ANN001
    """stackoverflow.com/questions/35651932/plotting-img-with-matplotlib/35651933"""
    if ax is None:
        ax = plt.gca()

    with contextlib.suppress(TypeError):
        image = plt.imread(image)

    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y, strict=False):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists
