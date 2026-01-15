import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.path as mpath
import numpy as np
import pandas as pd
import sasoptpy as so
from matplotlib import patches

from src.data_collection import fetch_bootstrap_data, fetch_team_picks, get_fpl_players, get_my_team_prices


def load_team_id() -> dict | None:
    info_path = Path('info.json')
    if info_path.exists():
        with info_path.open() as f:
            return json.load(f)['team-id']
    return None


def get_team(team_id: int, gameweek: int) -> tuple[list[int], int]:
    team_picks = fetch_team_picks(team_id, gameweek)

    # Scrape GW before FH to get the team from GW prior
    if team_picks['active_chip'] == 'freehit':
        team_picks = fetch_team_picks(team_id, gameweek - 1)
    # Adjust the player id with fplreview indices
    return [i['element'] for i in team_picks['picks']], team_picks['entry_history']['bank']


def randomize(seed: int, projection_df: pd.DataFrame, start: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    gws = np.arange(start, start + len([col for col in projection_df.columns if 'GW' in col]))

    for w in gws:
        noise = (
            projection_df[f'{w}_Pts']
            * (92 - projection_df[f'{w}_xMins'])
            / 134
            * rng.standard_normal(size=len(projection_df))
        )
        projection_df[f'{w}_Pts'] = projection_df[f'{w}_Pts'] + noise

    return projection_df


def get_transfer_history(team_id: int, last_gameweek: int) -> list[int]:
    transfers = []
    # Reversing GW history until a chip is played or 2+ transfers were made
    for gameweek in range(last_gameweek, 0, -1):
        team_picks = fetch_team_picks(team_id, gameweek)
        transfer = team_picks['entry_history']['event_transfers']
        chip = team_picks['active_chip']

        if chip is not None and chip not in {'3xc', 'bboost'}:
            transfer = 2
        transfers.append(transfer)
        if transfer > 1:
            break

    return transfers


def get_rolling(team_id: int, last_gameweek: int) -> tuple[int, int]:
    transfers = get_transfer_history(team_id, last_gameweek)

    # Start from gw where last chip used or when hits were taken
    # Reset FT count
    rolling = 0
    for transfer in reversed(transfers):
        # Transfer logic
        rolling = min(max(rolling + 1 - transfer, 0), 1)

    return rolling, transfers[0]


def get_chips(team_id: int, last_gameweek: int) -> tuple[int, int, int, int]:
    freehit, wildcard, bboost, threexc = 0, 0, 0, 0
    fh_count = 0
    # Reversing GW history until a chip is played or 2+ transfers were made
    for gameweek in range(last_gameweek, 0, -1):
        team_picks = fetch_team_picks(team_id, gameweek)
        chip = team_picks['active_chip']

        if chip == '3xc':
            threexc = gameweek
        if chip == 'bboost':
            bboost = gameweek
        if chip == 'wildcard' and wildcard == 0:
            wildcard = gameweek
        if chip == 'freehit':
            freehit = gameweek
            fh_count += 1

    # Handle the WC reset at GW 20
    reset_gameweek = 20
    if wildcard <= reset_gameweek and last_gameweek >= reset_gameweek:
        wildcard = 0

    return freehit, wildcard, bboost, threexc


def get_next_gameweek() -> int | None:
    data = fetch_bootstrap_data()

    for idx, gw in enumerate(data['events']):
        if gw['is_next']:
            return idx + 1
    return None


def get_season() -> int:
    with Path('info.json').open() as f:
        season_data = json.load(f)

    return season_data['season']


def get_ownership_data() -> pd.DataFrame:
    gw = get_next_gameweek() - 1
    season = get_season()
    df = pd.read_csv(f'data/fpl_official/{season}-{season % 2000 + 1}/gameweek/{gw}/player_ownership.csv')[
        ['id', 'Top_100', 'Top_1K', 'Top_10K', 'Top_50K', 'Top_100K', 'Top_250K']
    ]
    df['id'] = df['id']
    return df.set_index('id')


def pretty_print(  # noqa: C901, PLR0912, PLR0913, PLR0915
    data: pd.DataFrame,
    start: int,
    period: int,
    team: so.Variable,
    team_fh: so.Variable,
    starter: so.Variable,
    bench: so.Variable,
    captain: so.Variable,
    vicecaptain: so.Variable,
    buy: so.Variable,
    sell: so.Variable,
    free_transfers: so.Variable,
    hits: so.Variable,
    in_the_bank: so.Variable,
    objective_value: so.Objective,
    freehit: so.Variable,
    wildcard: so.Variable,
    bboost: so.Variable,
    threexc: so.Variable,
    nb_suboptimal: int = 0,
    ownership: bool = False,
) -> tuple[pd.DataFrame, list[str | None], float, float]:
    cols = ['GW', 'Name', 'Position', 'Team', 'selling_price', 'xP', 'xMins', 'Start', 'Bench', 'Cap', 'Vice']
    if ownership:
        cols = +['Ownership']

    df = pd.DataFrame([], columns=cols)
    total_ev = 0
    chip_strat = []

    for w in np.arange(start, start + period):
        print(f'GW: {w} - FT: {int(free_transfers[w].get_value())}')
        for p in data.index.tolist():
            if not freehit[w].get_value():
                if team[p, w].get_value():
                    if not starter[p, w].get_value():
                        bo = [-1] + [bench[p, w, o].get_value() for o in [0, 1, 2, 3]]
                    else:
                        bo = [0]

                    df_dict = {
                        'GW': w,
                        'Name': data.loc[p]['Name'],
                        'Position': data.loc[p]['Position'],
                        'Team': data.loc[p]['Team'],
                        'selling_price': data.loc[p]['selling_price'],
                        'xP': data.loc[p][f'GW{w}'],
                        'xMins': data.loc[p]['xMins'],
                        'Start': int(starter[p, w].get_value()),
                        'Bench': int(np.argmax(bo)),
                        'Cap': int(captain[p, w].get_value()),
                        'Vice': int(vicecaptain[p, w].get_value()),
                    }
                    if ownership:
                        df_dict['Ownership'] = data.loc[p]['Top_100']

                    df = pd.concat([df, pd.DataFrame([df_dict])], ignore_index=True)

            elif team_fh[p, w].get_value():
                bo = [-1] + [bench[p, w, o].get_value() for o in [0, 1, 2, 3]] if not starter[p, w].get_value() else [0]

                df_dict = {
                    'GW': w,
                    'Name': data.loc[p]['Name'],
                    'Position': data.loc[p]['Position'],
                    'Team': data.loc[p]['Team'],
                    'selling_price': data.loc[p]['selling_price'],
                    'xP': data.loc[p][str(w) + '_Pts'],
                    'xMins': data.loc[p][str(w) + '_xMins'],
                    'Start': int(starter[p, w].get_value()),
                    'Bench': int(np.argmax(bo)),
                    'Cap': int(captain[p, w].get_value()),
                    'Vice': int(vicecaptain[p, w].get_value()),
                }
                if ownership:
                    df_dict['Ownership'] = data.loc[p]['Top_100']

                df = pd.concat([df, pd.DataFrame([df_dict])], ignore_index=True)

            if buy[p, w].get_value():
                print(f'Buy: {data.loc[p, "Name"]}')
            if sell[p, w].get_value():
                print(f'Sell: {data.loc[p, "Name"]}')

        chip = ''
        av = ''
        if freehit[w].get_value():
            chip = ' - Chip: Freehit'
            chip_strat.append('FH')
        elif wildcard[w].get_value():
            chip = ' - Chip: Wildcard'
            chip_strat.append('WC')
        elif bboost[w].get_value():
            chip = ' - Chip: Bench Boost'
            val = np.sum(df.loc[(df['GW'] == w), 'xP']) - np.sum(df.loc[(df['Start'] == 1) & (df['GW'] == w), 'xP'])
            av = f' - Added value: {val}'
            chip_strat.append('BB')
        elif so.expr_sum(threexc[p, w] for p in data.index.tolist()).get_value():
            chip = ' - Chip: Triple Captain'
            val = np.sum(df.loc[(df['Cap'] == 1) & (df['GW'] == w), 'xP'])
            av = f' - Added value: {val}'
            chip_strat.append('TC')
        else:
            chip_strat.append(None)

        xpts_val = np.sum(df.loc[(df['Start'] == 1) & (df['GW'] == w), 'xP']) - hits[w].get_value() * 4 * (
            0 if wildcard[w].get_value() else 1
        ) * (0 if freehit[w].get_value() else 1)
        total_ev += xpts_val
        hits_val = (
            int(hits[w].get_value()) * (0 if wildcard[w].get_value() else 1) * (0 if freehit[w].get_value() else 1)
        )

        print()
        print(
            df.loc[w == df.GW]
            .sort_values(by=['Position'], key=lambda x: x.map({'G': 0, 'D': 1, 'M': 2, 'F': 3}))
            .sort_values(by=['Start', 'Bench'], ascending=[False, True])
        )

        print(f'xPts: {xpts_val:.2f} - Hits: {hits_val}' + chip + av + f' - ITB: {in_the_bank[w].get_value() / 10:.1f}')
        print(' ____ ')

    df = df.sort_values(by=['Position'], key=lambda x: x.map({'G': 0, 'D': 1, 'M': 2, 'F': 3})).sort_values(
        by=['GW', 'Start', 'Bench'], ascending=[True, False, True]
    )
    df.to_csv(f'tmp/{nb_suboptimal}.csv')

    print(f'EV: {total_ev:.2f}  |  Objective Val: {-objective_value:.2f}')

    return df, chip_strat, total_ev, -objective_value


def convert_txt_to_csv(input_file: str, output_file: str) -> None:
    # Convert the raw text data (from the FPL Review Free Projection Data) into a structured CSV format
    with Path(input_file).open('r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        msg = 'Input file is empty'
        raise ValueError(msg)

    column_names = [col.strip() for col in lines[0].split('\t') if col.strip()]

    # Parse the data - pattern is: blank line, name, position/price, data line
    data_rows = []
    i = 1  # Skip the header line at index 0

    # Remove all blank lines upfront
    lines = [line for line in lines if line.strip()]

    while i < len(lines):
        line = lines[i].strip()

        player_name = line.strip()

        # Next line should be position and price
        if i + 1 >= len(lines):
            break
        position_price_line = lines[i + 1].strip()
        position_and_price = position_price_line.split()

        # Next line should contain position and price
        if len(position_and_price) <= 1:
            i += 1
            continue

        position = position_and_price[0]
        price = position_and_price[1]

        # Next line should be the projection data
        if i + 2 >= len(lines):
            break
        projection_data = lines[i + 2].strip().split()

        # Combine: name, position, price, then all data columns
        row = [player_name, position, price, *projection_data]
        data_rows.append(row)

        i += 3  # Move to the next player

    with Path(output_file).open('w', newline='') as f:
        writer = csv.writer(f)
        header = ['NAME', 'POSITION', 'PRICE', *column_names]
        writer.writerow(header)
        writer.writerows(data_rows)

    print(f'✓ Successfully converted {len(data_rows)} players to CSV')


def match_player_names(projection_df: pd.DataFrame, team_id: int, next_gameweek: int) -> pd.DataFrame:
    fpl_df = get_fpl_players()
    team_prices = get_my_team_prices(team_id, next_gameweek)

    enriched_data = projection_df.apply(
        lambda row: _enrich_player_data(
            row,
            fpl_df.loc[
                (fpl_df['position'] == row['POSITION'])
                & (fpl_df['now_cost'].between(row['PRICE'] * 10 - 1, row['PRICE'] * 10 + 1))
            ],
            team_prices,
        ),
        axis=1,
    )

    return projection_df.merge(
        pd.DataFrame(enriched_data.tolist()).set_index('index'),
        left_index=True,
        right_index=True,
        how='left',
    )


def _enrich_player_data(projection_data_player: pd.Series, fpl_df: pd.DataFrame, team_prices: dict) -> dict:
    match = find_best_string_match(projection_data_player['NAME'], fpl_df)

    if match is None:
        return {
            'index': projection_data_player.name,
            'fpl_id': None,
            'team_id': None,
            'Team': None,
            'purchase_price': None,
            'selling_price': None,
            'in_my_team': False,
        }

    # Select the best matching player by price if multiple matches found
    if isinstance(match, pd.DataFrame) and len(match) > 1:
        now_cost = projection_data_player['PRICE']
        match['price_diff'] = (match['now_cost'] / 10 - now_cost).abs()
        match = match.loc[match['price_diff'].idxmin()]

    player_id = match['player_id']
    now_cost = match['now_cost'] / 10
    enriched = {
        'index': projection_data_player.name,
        'fpl_id': player_id,
        'team_id': match['team_id'],
        'Team': match['Team'],
        'purchase_price': now_cost,
        'selling_price': now_cost,
        'in_my_team': False,
    }

    if player_id in team_prices:
        prices = team_prices[player_id]
        enriched.update(
            {'purchase_price': prices['purchase_price'], 'selling_price': prices['selling_price'], 'in_my_team': True}
        )

    return enriched


def find_best_string_match(player_name: str, player_df: pd.DataFrame, threshold: float = 0.6) -> pd.Series | None:
    best_match = None
    best_ratio = 0.0

    for _, player in player_df.iterrows():
        name_variants = [player['web_name'], player['second_name'], player['full_name']]

        for fpl_name in name_variants:
            ratio = _calculate_similarity(player_name, fpl_name)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = player

    return best_match if best_ratio > threshold else None


def _calculate_similarity(name1: str, name2: str) -> float:
    return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()


def bezier_path(p1: tuple, p2: tuple, color: str = 'white') -> patches.PathPatch:
    x1, y1 = p1
    x2, y2 = p2

    if y2 != y1:
        path_data = [
            (mpath.Path.MOVETO, (x1, y1)),
            (mpath.Path.CURVE3, (x1 + (x2 - x1) / 2, y1)),
            (mpath.Path.CURVE3, (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)),
            (mpath.Path.CURVE3, (x1 + (x2 - x1) / 2, y2)),
            (mpath.Path.CURVE3, (x2, y2)),
        ]
        codes, verts = zip(*path_data, strict=False)
        path = mpath.Path(verts, codes)
        patch = patches.PathPatch(path, ec=color, fc='none', zorder=2, lw=2)

    else:
        path_data = [
            (mpath.Path.MOVETO, (x1, y1)),
            (mpath.Path.LINETO, (x2, y2)),
        ]
        codes, verts = zip(*path_data, strict=False)
        path = mpath.Path(verts, codes)
        patch = patches.PathPatch(path, ec=color, fc='none', zorder=2, lw=2)

    return patch
