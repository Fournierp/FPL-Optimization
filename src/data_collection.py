import csv
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

FPL_API_URL = 'https://fantasy.premierleague.com/api/'


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
                & (fpl_df['now_cost'].between(row['PRICE'] * 10 - 2, row['PRICE'] * 10 + 2))
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


def get_fpl_players() -> pd.DataFrame:
    data = fetch_bootstrap_data()

    teams_short = {team['id']: team['short_name'] for team in data['teams']}
    positions = {pos['id']: pos['singular_name_short'] for pos in data['element_types']}

    player_data = [
        {
            'player_id': player['id'],
            'team_id': player['team'],
            'web_name': player['web_name'],
            'first_name': player['first_name'],
            'second_name': player['second_name'],
            'full_name': f'{player["first_name"]} {player["second_name"]}',
            'Team': teams_short[player['team']],
            'position': positions[player['element_type']],
            'now_cost': player['now_cost'],
        }
        for player in data['elements']
    ]
    player_data = pd.DataFrame(player_data)
    player_data['position'] = player_data['position'].map(
        {
            'GKP': 'GK',
            'DEF': 'DF',
            'MID': 'MD',
            'FWD': 'FW',
        }
    )

    return pd.DataFrame(player_data)


def get_my_team_prices(team_id: int, gameweek: int) -> dict[int, dict[str, float]]:
    picks_res = _fetch_team_picks(team_id, gameweek)
    transfers = _fetch_transfer_history(team_id)
    bootstrap = fetch_bootstrap_data()

    current_squad = [pick['element'] for pick in picks_res['picks']]
    purchase_prices = {}
    for transfer in transfers:
        player_id = transfer['element_in']
        purchase_price = transfer['element_in_cost']
        purchase_prices[player_id] = purchase_price
    current_prices = {p['id']: p['now_cost'] for p in bootstrap['elements']}

    team_prices = {}
    for player_id in current_squad:
        current_price = current_prices.get(player_id, 0)
        purchase_price = purchase_prices.get(player_id, current_price)
        selling_price = _calculate_selling_price(purchase_price, current_price)

        team_prices[player_id] = {
            'purchase_price': purchase_price / 10,
            'selling_price': selling_price / 10,
            'current_price': current_price / 10,
        }

    return team_prices


def _fetch_team_picks(team_id: int, gameweek: int) -> dict:
    picks_url = f'{FPL_API_URL}entry/{team_id}/event/{gameweek}/picks/'
    picks_res = requests.get(picks_url, timeout=10).json()

    if 'picks' not in picks_res and gameweek > 1:
        picks_url = f'{FPL_API_URL}entry/{team_id}/event/{gameweek - 1}/picks/'
        picks_res = requests.get(picks_url, timeout=10).json()

    if 'picks' not in picks_res:
        msg = f'Could not fetch picks for team {team_id} at GW{gameweek} or GW{gameweek - 1}'
        raise ValueError(msg)
    return picks_res


def _fetch_transfer_history(team_id: int) -> list[dict]:
    transfers_url = f'{FPL_API_URL}entry/{team_id}/transfers/'
    return requests.get(transfers_url, timeout=10).json()


def fetch_bootstrap_data() -> dict:
    bootstrap_url = f'{FPL_API_URL}bootstrap-static/'
    return requests.get(bootstrap_url, timeout=10).json()


def _calculate_selling_price(purchase_price: int, current_price: int) -> int:
    price_rise = current_price - purchase_price
    if price_rise > 0:
        return purchase_price + (price_rise // 2)
    return current_price


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
