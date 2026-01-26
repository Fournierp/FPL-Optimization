import pandas as pd
import requests

FPL_API_URL = 'https://fantasy.premierleague.com/api/'


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
    picks_res = fetch_team_picks(team_id, gameweek)
    transfers = _fetch_transfer_history(team_id)
    bootstrap = fetch_bootstrap_data()

    current_squad = [pick['element'] for pick in picks_res['picks']]
    purchase_prices = {}
    for transfer in reversed(transfers):
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


def fetch_team_picks(team_id: int, gameweek: int) -> dict:
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

