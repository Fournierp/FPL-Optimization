import json
import logging
import os
from pathlib import Path
from subprocess import DEVNULL, Popen

import numpy as np
import pandas as pd
import sasoptpy as so

from src.utils import (
    get_chips,
    get_ownership_data,
    get_rolling,
    get_team,
    pretty_print,
    randomize,
)

MAX_NUMBER_GOALKEEPERS = 2
MAX_NUMBER_DEFENDERS = 5
MAX_NUMBER_MIDFIELDERS = 5
MAX_NUMBER_FORWARDS = 3
MAX_NUMBER_PLAYERS_PER_TEAM = 3
MIN_FORMATION = {'GK': 1, 'DF': 3, 'MD': 2, 'FW': 1}


class TeamOptimization:
    def __init__(self, params: dict) -> None:
        self.filter_ev = params['filter_ev']
        self.horizon = params['horizon']
        self.ownership = params['ownership']
        self.predictions = params['predictions']
        self.start = params['start']

        self.get_data(params['team_id'])

        if params['noise']:
            self.random_noise(42)

    def get_data(self, team_id: int) -> None:
        # Projection data
        self.team_names = self.predictions.Team.unique().tolist()
        self.data = self.predictions.copy()
        self.data = self.data.dropna(subset=['fpl_id'])
        self.data['fpl_id'] = self.data['fpl_id'].astype(int)
        self.data = self.data.set_index('fpl_id')
        # One hot encoded values for the constraints
        self.data = pd.concat([self.data, pd.get_dummies(self.data.Position, dtype=int)], axis=1)
        self.data = pd.concat([self.data, pd.get_dummies(self.data.Team, dtype=int)], axis=1)

        # FPL data
        if self.start != 1:
            if self.ownership:
                ownership = get_ownership_data()
                self.data = pd.concat([self.data, ownership], axis=1, join='inner')

            self.initial_team, self.bank = get_team(team_id, self.start - 1)
            (self.freehit_used, self.wildcard_used, self.bboost_used, self.threexc_used) = get_chips(
                team_id, self.start - 1
            )

            self.period = min(self.horizon, len([col for col in self.data.columns if 'GW' in col]))
            (self.rolling_transfer, self.transfer) = get_rolling(team_id, self.start - 1)

        else:
            self.initial_team, self.bank = [0 for i in range(15)], 100
            (self.freehit_used, self.wildcard_used, self.bboost_used, self.threexc_used) = 0, 0, 0, 0

            self.period = min(self.horizon, len([col for col in self.data.columns if 'GW' in col]))
            (self.rolling_transfer, self.transfer) = 0, 0

        self.budget = np.sum(self.data.loc[self.initial_team, 'selling_price']) + self.bank

        self.all_gameweeks = np.arange(self.start - 1, self.start + self.period)
        self.gameweeks = np.arange(self.start, self.start + self.period)

        # Sort DF by EV for efficient optimization
        self.data['total_ev'] = self.data[[col for col in self.data.columns if 'GW' in col]].sum(axis=1)
        self.data = self.data.sort_values(by=['total_ev'], ascending=[False])

        # Drop players that are not predicted to play much to reduce the search space
        if self.filter_ev is not None:
            print(
                f'Droped {self.data[self.data.total_ev <= self.filter_ev].shape[0]} players '
                f'because they have no projected points.'
            )
            self.data = self.data.drop(self.data[self.data.total_ev <= self.filter_ev].index)
        self.players = self.data.index.tolist()

        self.initial_team_df = pd.DataFrame([], columns=['GW', 'Name', 'Position', 'Team', 'selling_price'])

        for p in self.initial_team:
            self.initial_team_df = pd.concat(
                (
                    self.initial_team_df,
                    pd.DataFrame(
                        [
                            {
                                'GW': self.start - 1,
                                'Name': self.data.loc[p]['Name'],
                                'Position': self.data.loc[p]['Position'],
                                'Team': self.data.loc[p]['Team'],
                                'selling_price': self.data.loc[p]['selling_price'],
                            }
                        ]
                    ),
                ),
                ignore_index=True,
            )

    def random_noise(self, seed: int | None) -> None:
        self.data = randomize(seed, self.data, self.start)

    def build_model(self, params: dict) -> None:  # noqa: C901, PLR0912, PLR0915
        model_name = params.get('model_name', 'vanilla')
        freehit_gw = params.get('freehit_gw', -1)
        wildcard_gw = params.get('wildcard_gw', -1)
        bboost_gw = params.get('bboost_gw', -1)
        threexc_gw = params.get('threexc_gw', -1)
        objective_type = params.get('objective_type', 'decay')
        decay_gameweek = params.get('decay_gameweek', 0.9)
        vicecap_decay = params.get('vicecap_decay', 0.1)
        decay_bench = params.get('decay_bench', [0.1, 0.1, 0.1, 0.1])
        ft_val = params.get('ft_val', 0)
        itb_val = params.get('itb_val', 0)
        hit_val = params.get('hit_val', 6)
        goalkeeper_max_budget = params.get('goalkeeper_max_budget', 100)
        def_stack_limit = params.get('def_stack_limit', 3)

        gw_out_of_horizon_msg = 'Select a GW within the horizon.'
        if freehit_gw >= self.horizon:
            raise ValueError(gw_out_of_horizon_msg)
        if wildcard_gw >= self.horizon:
            raise ValueError(gw_out_of_horizon_msg)
        if bboost_gw >= self.horizon:
            raise ValueError(gw_out_of_horizon_msg)
        if threexc_gw >= self.horizon:
            raise ValueError(gw_out_of_horizon_msg)

        freehit_used_msg = 'Freehit chip was already used.'
        if self.freehit_used and freehit_gw >= 0:
            raise ValueError(freehit_used_msg)
        wildcard_used_msg = 'Wildcard chip was already used.'
        if self.wildcard_used and wildcard_gw >= 0:
            raise ValueError(wildcard_used_msg)
        bboost_used_msg = 'Bench boost chip was already used.'
        if self.bboost_used and bboost_gw >= 0:
            raise ValueError(bboost_used_msg)
        threexc_used_msg = 'Tripple captain chip was already used.'
        if self.threexc_used and threexc_gw >= 0:
            raise ValueError(threexc_used_msg)

        # Model
        self.model = so.Model(name=model_name)

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(self.players, self.all_gameweeks, name='team', vartype=so.binary)
        self.team_fh = self.model.add_variables(self.players, self.gameweeks, name='team_fh', vartype=so.binary)
        self.starter = self.model.add_variables(self.players, self.gameweeks, name='starter', vartype=so.binary)
        self.bench = self.model.add_variables(self.players, self.gameweeks, order, name='bench', vartype=so.binary)

        self.captain = self.model.add_variables(self.players, self.gameweeks, name='captain', vartype=so.binary)
        self.vicecaptain = self.model.add_variables(self.players, self.gameweeks, name='vicecaptain', vartype=so.binary)

        self.buy = self.model.add_variables(self.players, self.gameweeks, name='buy', vartype=so.binary)
        self.sell = self.model.add_variables(self.players, self.gameweeks, name='sell', vartype=so.binary)

        self.triple = self.model.add_variables(self.players, self.gameweeks, name='3xc', vartype=so.binary)
        self.bboost = self.model.add_variables(self.gameweeks, name='bb', vartype=so.binary)
        self.freehit = self.model.add_variables(self.gameweeks, name='fh', vartype=so.binary)
        self.wildcard = self.model.add_variables(self.gameweeks, name='wc', vartype=so.binary)

        self.aux = self.model.add_variables(self.players, self.all_gameweeks, name='aux', vartype=so.binary)
        self.free_transfers = self.model.add_variables(
            np.arange(self.start - 1, self.start + self.period + 1), name='ft', vartype=so.integer, lb=0
        )
        self.hits = self.model.add_variables(self.all_gameweeks, name='hits', vartype=so.integer, lb=0)
        self.in_the_bank = self.model.add_variables(self.all_gameweeks, name='itb', vartype=so.continuous, lb=0)

        # Objective: maximize total expected points
        starter = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(self.starter[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        cap = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(self.captain[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        vice = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(vicecap_decay * self.vicecaptain[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        bench = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(
                so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order) * self.data.loc[p, f'GW{w}']
                for p in self.players
            )
            for w in self.gameweeks
        )

        txc = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(2 * self.triple[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        hits = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1) * (hit_val * self.hits[w])
            for w in self.gameweeks
        )

        ftv = so.expr_sum(
            (np.power(decay_gameweek, w - self.start - 1) if objective_type == 'linear' else 1)
            * (ft_val * (self.free_transfers[w] - 1))
            for w in self.gameweeks[1:]
        )

        itbv = so.expr_sum(
            (np.power(decay_gameweek, w - self.start - 1) if objective_type == 'linear' else 1)
            * (itb_val * self.in_the_bank[w])
            for w in self.gameweeks
        )

        self.model.set_objective(
            -starter - cap - vice - bench - txc - ftv - itbv + hits, name='total_xp_obj', sense='N'
        )

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints((self.team[p, self.start - 1] == 1 for p in self.initial_team), name='initial_team')
        self.model.add_constraint(self.free_transfers[self.start] == self.rolling_transfer + 1, name='initial_ft')
        self.model.add_constraint(self.in_the_bank[self.start - 1] == self.bank, name='initial_itb')

        # Constraints
        # Chips
        # The chips must not be used more than once
        self.model.add_constraint(
            so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) <= int(not self.threexc_used),
            name='tc_once',
        )
        self.model.add_constraint(
            so.expr_sum(self.bboost[w] for w in self.gameweeks) <= int(not self.bboost_used), name='bb_once'
        )
        self.model.add_constraint(
            so.expr_sum(self.freehit[w] for w in self.gameweeks) <= int(not self.freehit_used), name='fh_once'
        )
        self.model.add_constraint(
            so.expr_sum(self.wildcard[w] for w in self.gameweeks) <= int(not self.wildcard_used), name='wc_once'
        )

        # The chips must not be used on the same GW
        self.model.add_constraints(
            (
                so.expr_sum(self.triple[p, w] for p in self.players)
                + self.bboost[w]
                + self.freehit[w]
                + self.wildcard[w]
                <= 1
                for w in self.gameweeks
            ),
            name='chip_once',
        )

        # The chips must be used on the selected GW
        if bboost_gw + 1:
            self.model.add_constraint(self.bboost[self.start + bboost_gw] == 1, name='bboost_gw')
        else:
            self.model.add_constraint(so.expr_sum(self.bboost[w] for w in self.gameweeks) == 0, name='bboost_unused')

        if threexc_gw + 1:
            self.model.add_constraint(
                so.expr_sum(self.triple[p, self.start + threexc_gw] for p in self.players) == 1, name='triple_gw'
            )
        else:
            self.model.add_constraint(
                so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) == 0, name='triple_unused'
            )

        if freehit_gw + 1:
            self.model.add_constraint(self.freehit[self.start + freehit_gw] == 1, name='freehit_gw')
        else:
            self.model.add_constraint(so.expr_sum(self.freehit[w] for w in self.gameweeks) == 0, name='freehit_unused')

        if wildcard_gw + 1:
            self.model.add_constraint(self.wildcard[self.start + wildcard_gw] == 1, name='wildcard_gw')
        else:
            self.model.add_constraint(
                so.expr_sum(self.wildcard[w] for w in self.gameweeks) == 0, name='wildcard_unused'
            )

        # Team
        # The number of players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'GK'] for p in self.players) == MAX_NUMBER_GOALKEEPERS
                for w in self.gameweeks
            ),
            name='gk_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'DF'] for p in self.players) == MAX_NUMBER_DEFENDERS
                for w in self.gameweeks
            ),
            name='def_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'MD'] for p in self.players) == MAX_NUMBER_MIDFIELDERS
                for w in self.gameweeks
            ),
            name='mid_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'FW'] for p in self.players) == MAX_NUMBER_FORWARDS
                for w in self.gameweeks
            ),
            name='for_limit',
        )

        # The number of players from a team must exceed three
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, team_name] for p in self.players)
                <= MAX_NUMBER_PLAYERS_PER_TEAM
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='team_limit',
        )

        # The number of Freehit players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'GK'] for p in self.players)
                == MAX_NUMBER_GOALKEEPERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='gk_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'DF'] for p in self.players)
                == MAX_NUMBER_DEFENDERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='def_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'MD'] for p in self.players)
                == MAX_NUMBER_MIDFIELDERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='mid_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'FW'] for p in self.players)
                == MAX_NUMBER_FORWARDS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='for_limit_fh',
        )

        # The number of Freehit players from a team must not exceed three
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, team_name] for p in self.players)
                <= 3 * self.freehit[w]
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='team_limit_fh',
        )

        # Starters
        # The formation must be valid i.e. Minimum one goalkeeper,
        # 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints(
            (so.expr_sum(self.starter[p, w] for p in self.players) == 11 + 4 * self.bboost[w] for w in self.gameweeks),
            name='11_starters',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'GK'] for p in self.players)
                == MIN_FORMATION['GK'] + self.bboost[w]
                for w in self.gameweeks
            ),
            name='gk_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'DF'] for p in self.players) >= MIN_FORMATION['DF']
                for w in self.gameweeks
            ),
            name='def_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'MD'] for p in self.players) >= MIN_FORMATION['MD']
                for w in self.gameweeks
            ),
            name='mid_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'FW'] for p in self.players) >= MIN_FORMATION['FW']
                for w in self.gameweeks
            ),
            name='for_min',
        )

        # Linearization constraints to limit the Freehit Team
        self.model.add_constraints(
            (self.starter[p, w] <= self.team_fh[p, w] + self.aux[p, w] for p in self.players for w in self.gameweeks),
            name='4.24',
        )
        self.model.add_constraints(
            (self.aux[p, w] <= self.team[p, w] for p in self.players for w in self.gameweeks), name='4.25'
        )
        self.model.add_constraints(
            (self.aux[p, w] <= 1 - self.freehit[w] for p in self.players for w in self.gameweeks), name='4.26'
        )

        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, team_name] * (self.data.loc[p, 'DF'] + self.data.loc[p, 'GK'])
                    for p in self.players
                )
                <= def_stack_limit * self.freehit[w]
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='def_stack_limit_fh',
        )

        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, team_name] * (self.data.loc[p, 'DF'] + self.data.loc[p, 'GK'])
                    for p in self.players
                )
                <= def_stack_limit
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='def_stack_limit',
        )

        # Captain
        # One captain (or one triple cap) must be picked once
        self.model.add_constraints(
            (so.expr_sum(self.captain[p, w] + self.triple[p, w] for p in self.players) == 1 for w in self.gameweeks),
            name='one_captain',
        )
        # One vice captain must be picked once
        self.model.add_constraints(
            (so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1 for w in self.gameweeks),
            name='one_vicecaptain',
        )
        # The captain, vice captain and triple captain must be starters and
        # must not be the same player
        self.model.add_constraints(
            (
                self.captain[p, w] + self.triple[p, w] + self.vicecaptain[p, w] <= self.starter[p, w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='cap_in_starters',
        )

        # Substitutions
        # The first substitute is a single goalkeeper
        self.model.add_constraints(
            (
                so.expr_sum(self.bench[p, w, 0] for p in self.players if self.data.loc[p, 'GK'] == 1)
                == 1 - self.bboost[w]
                for w in self.gameweeks
            ),
            name='one_bench_gk',
        )
        # There must be a single substitute per bench spot
        self.model.add_constraints(
            (so.expr_sum(self.bench[p, w, o] for p in self.players) <= 1 for w in self.gameweeks for o in [0, 1, 2, 3]),
            name='one_per_bench_spot',
        )

        # The players not started in the team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order)
                <= self.team[p, w] + 10000 * self.freehit[w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='bench_team',
        )
        # The players not started in the freehit team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order)
                <= self.team_fh[p, w] + 10000 * (1 - self.freehit[w])
                for p in self.players
                for w in self.gameweeks
            ),
            name='bench_team_fh',
        )

        # Budget
        sold_amount = {
            w: so.expr_sum(self.sell[p, w] * self.data.loc[p, 'selling_price'] for p in self.players)
            for w in self.gameweeks
        }
        bought_amount = {
            w: so.expr_sum(self.buy[p, w] * self.data.loc[p, 'purchase_price'] for p in self.players)
            for w in self.gameweeks
        }
        # The cost of the squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w] == self.in_the_bank[w - 1] + sold_amount[w] - bought_amount[w]
                for w in self.gameweeks
            ),
            name='budget',
        )
        # The team must be the same as the previous GW plus/minus transfers
        self.model.add_constraints(
            (
                self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] == self.team[p, w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='team_similarity',
        )
        # The player must not be sold and bought simultaneously
        self.model.add_constraints(
            (self.sell[p, w] + self.buy[p, w] <= 1 for p in self.players for w in self.gameweeks),
            name='single_buy_or_sell',
        )

        # The cost of the freehit squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w - 1]
                + so.expr_sum(self.team[p, w - 1] * self.data.loc[p, 'selling_price'] for p in self.players)
                >= so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'purchase_price'] for p in self.players)
                for w in self.gameweeks
            ),
            name='budget_fh',
        )
        # On Freehit GW the number of transfers must be zero
        self.model.add_constraints(
            (so.expr_sum(self.sell[p, w] for p in self.players) <= 15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_sold_fh',
        )
        self.model.add_constraints(
            (so.expr_sum(self.buy[p, w] for p in self.players) <= 15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_bought_fh',
        )

        # The number of players bought and sold is equal
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) == so.expr_sum(self.sell[p, w] for p in self.players)
                for w in self.gameweeks
            ),
            name='equal_transfers',
        )

        goalkeeper_cost = {
            w: so.expr_sum(
                self.team[p, w] * self.data.loc[p, 'purchase_price'] * self.data.loc[p, 'GK'] for p in self.players
            )
            for w in self.gameweeks
        }
        self.model.add_constraints(
            (goalkeeper_cost[w] <= goalkeeper_max_budget * 10 for w in self.gameweeks), name='goalkeeper_budget'
        )

        goalkeeper_cost_fh = {
            w: so.expr_sum(
                self.team_fh[p, w] * self.data.loc[p, 'purchase_price'] * self.data.loc[p, 'GK'] for p in self.players
            )
            for w in self.gameweeks
        }
        self.model.add_constraints(
            (goalkeeper_cost_fh[w] <= goalkeeper_max_budget * 10 for w in self.gameweeks), name='goalkeeper_budget_fh'
        )

        # Transfers
        # The rolling transfer must be equal to the number of free
        # transfers not used (+ 1)
        self.model.add_constraints(
            (
                15 * self.wildcard[w]
                + self.free_transfers[w]
                - so.expr_sum(self.buy[p, w] for p in self.players)
                + 1
                + self.hits[w]
                >= self.free_transfers[w + 1]
                for w in self.gameweeks
            ),
            name='rolling_ft_rel',
        )
        # The hits value is zero only when the number of FT is 2
        self.model.add_constraints(
            (10000 * (2 - self.free_transfers[w + 1]) >= self.hits[w] for w in self.gameweeks), name='4.42'
        )
        # The minimum number of FT is 1
        self.model.add_constraints((self.free_transfers[w + 1] >= 1 for w in self.gameweeks), name='min_ft')
        # The maximum number of FT is 2 on regular GWs
        self.model.add_constraints(
            (self.free_transfers[w + 1] <= 2 - self.wildcard[w] - self.freehit[w] for w in self.gameweeks),
            name='max_ft',
        )

    def differential_model(self, nb_differentials: int = 3, threshold: int = 10, target: str = 'Top_100K') -> None:
        self.data['Differential'] = np.where(self.data[target] < threshold, 1, 0)
        # A min numberof starter players must be differentials
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'Differential'] for p in self.players)
                >= nb_differentials
                for w in self.gameweeks
            ),
            name='differentials',
        )

    def biased_model(self, love: dict, hate: dict, hit_limit: dict, two_ft_gw: list) -> None:  # noqa: C901, PLR0912, PLR0915
        gw_not_exist_msg = 'Gameweek selected does not exist.'
        player_not_exist_msg = 'Player selected to buy does not exist.'
        for bias, bias_value in love.items():
            if bias == 'buy' and bias_value:
                if not all(w in self.gameweeks for (_, w) in love['buy']):
                    raise ValueError(gw_not_exist_msg)
                if not all(bias[0] in self.players for bias in love['buy']):
                    raise ValueError(player_not_exist_msg)
                # The forced-buy player must be bought
                self.model.add_constraints((self.buy[p, w] == 1 for (p, w) in bias_value), name='force_buy')
            if bias == 'start' and bias_value:
                if not all(w in self.gameweeks for (_, w) in love['start']):
                    raise ValueError(gw_not_exist_msg)
                player_start_not_exist_msg = 'Player selected to start does not exist.'
                if not all(bias[0] in self.players for bias in love['start']):
                    raise ValueError(player_start_not_exist_msg)
                # The forced-in team player must be in the team
                self.model.add_constraints((self.team[p, w] == 1 for (p, w) in bias_value), name='force_in')
            if bias == 'team' and bias_value:
                if not all(w in self.gameweeks for (_, w) in love['team']):
                    raise ValueError(gw_not_exist_msg)
                player_team_not_exist_msg = 'Player selected to be in the team does not exist.'
                if not all(bias[0] in self.players for bias in love['team']):
                    raise ValueError(player_team_not_exist_msg)
                # The forced-in starter player must be a starter
                self.model.add_constraints((self.starter[p, w] == 1 for (p, w) in bias_value), name='force_starter')
            if bias == 'cap' and bias_value:
                if not all(w in self.gameweeks for (_, w) in love['cap']):
                    raise ValueError(gw_not_exist_msg)
                player_cap_not_exist_msg = 'Player selected to be the captain does not exist.'
                if not all(bias[0] in self.players for bias in love['cap']):
                    raise ValueError(player_cap_not_exist_msg)
                # The forced-in cap player must be the captain
                self.model.add_constraints((self.captain[p, w] == 1 for (p, w) in bias_value), name='force_captain')

        for bias, bias_value in hate.items():
            if bias == 'sell' and bias_value:
                if not all(w in self.gameweeks for (_, w) in hate['sell']):
                    raise ValueError(gw_not_exist_msg)
                player_sell_not_exist_msg = 'Player selected to sell does not exist.'
                if not all(bias[0] in self.players for bias in hate['sell']):
                    raise ValueError(player_sell_not_exist_msg)
                # The forced-out player must be sold
                self.model.add_constraints((self.sell[p, w] == 1 for (p, w) in bias_value), name='force_sell')
            if bias == 'bench' and bias_value:
                if not all(w in self.gameweeks for (_, w) in hate['bench']):
                    raise ValueError(gw_not_exist_msg)
                if not all(bias[0] in self.players for bias in hate['bench']):
                    raise ValueError(player_start_not_exist_msg)
                # The forced-out of starter player must not be starting
                self.model.add_constraints(
                    (self.starter[p, w] == 0 for (p, w) in bias_value), name='force_bench'
                )  # Force player out by a certain gw
            if bias == 'team' and bias_value:
                if not all(w in self.gameweeks for (_, w) in hate['team']):
                    raise ValueError(gw_not_exist_msg)
                player_out_team_msg = 'Player selected to be out of the team does not exist.'
                if not all(bias[0] in self.players for bias in hate['team']):
                    raise ValueError(player_out_team_msg)
                # The forced-out of team player must not be in team
                self.model.add_constraints((self.team[p, w] == 0 for (p, w) in bias_value), name='force_out')

        for bias in hit_limit:  # noqa: PLC0206
            if bias == 'max' and hit_limit[bias]:
                if not all(w in self.gameweeks for (w, _) in hit_limit['max']):
                    raise ValueError(gw_not_exist_msg)
                # The number of hits under the maximum
                self.model.add_constraints(
                    (self.hits[w] <= max_hit for (w, max_hit) in hit_limit[bias]), name='hits_max'
                )
            if bias == 'eq' and hit_limit[bias]:
                if not all(w in self.gameweeks for (w, _) in hit_limit['eq']):
                    raise ValueError(gw_not_exist_msg)
                # The number of hits equal to the choice
                self.model.add_constraints((self.hits[w] == nb_hit for (w, nb_hit) in hit_limit[bias]), name='hits_eq')
            if bias == 'min' and hit_limit[bias]:
                if not all(w in self.gameweeks for (w, _) in hit_limit['min']):
                    raise ValueError(gw_not_exist_msg)
                # The number of hits above the minumum
                self.model.add_constraints(
                    (self.hits[w] >= min_hit for (w, min_hit) in hit_limit[bias]), name='hits_min'
                )

        for gw in two_ft_gw:
            if not (gw > self.start and gw <= self.start + self.horizon):
                raise ValueError(gw_not_exist_msg)
            # Force rolling free transfer
            max_rolled_free_transfers = 2
            self.model.add_constraint(self.free_transfers[gw] == max_rolled_free_transfers, name=f'force_roll_{gw}')

    def automated_chips_model(self, params: dict) -> None:  # noqa: PLR0915
        # Extract parameters with defaults
        objective_type = params.get('objective_type', 'decay')
        decay_gameweek = params.get('decay_gameweek', 0.9)
        vicecap_decay = params.get('vicecap_decay', 0.1)
        decay_bench = params.get('decay_bench', [0.1, 0.1, 0.1, 0.1])
        ft_val = params.get('ft_val', 0)
        itb_val = params.get('itb_val', 0)
        hit_val = params.get('hit_val', 6)
        goalkeeper_max_budget = params.get('goalkeeper_max_budget', 100)
        def_stack_limit = params.get('def_stack_limit', 3)
        triple_val = params.get('triple_val', 12)
        bboost_val = params.get('bboost_val', 14)
        freehit_val = params.get('freehit_val', 18)
        wildcard_val = params.get('wildcard_val', 18)
        # Model
        self.model = so.Model(name='auto_chips_model')

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(self.players, self.all_gameweeks, name='team', vartype=so.binary)
        self.team_fh = self.model.add_variables(self.players, self.gameweeks, name='team_fh', vartype=so.binary)
        self.starter = self.model.add_variables(self.players, self.gameweeks, name='starter', vartype=so.binary)
        self.bench = self.model.add_variables(self.players, self.gameweeks, order, name='bench', vartype=so.binary)

        self.captain = self.model.add_variables(self.players, self.gameweeks, name='captain', vartype=so.binary)
        self.vicecaptain = self.model.add_variables(self.players, self.gameweeks, name='vicecaptain', vartype=so.binary)

        self.buy = self.model.add_variables(self.players, self.gameweeks, name='buy', vartype=so.binary)
        self.sell = self.model.add_variables(self.players, self.gameweeks, name='sell', vartype=so.binary)

        self.triple = self.model.add_variables(self.players, self.gameweeks, name='3xc', vartype=so.binary)
        self.bboost = self.model.add_variables(self.gameweeks, name='bb', vartype=so.binary)
        self.freehit = self.model.add_variables(self.gameweeks, name='fh', vartype=so.binary)
        self.wildcard = self.model.add_variables(self.gameweeks, name='wc', vartype=so.binary)

        self.aux = self.model.add_variables(self.players, self.all_gameweeks, name='aux', vartype=so.binary)
        self.free_transfers = self.model.add_variables(
            np.arange(self.start - 1, self.start + self.period + 1), name='ft', vartype=so.integer, lb=0
        )
        self.hits = self.model.add_variables(self.all_gameweeks, name='hits', vartype=so.integer, lb=0)
        self.in_the_bank = self.model.add_variables(self.all_gameweeks, name='itb', vartype=so.continuous, lb=0)

        # Objective: maximize total expected points
        starter = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(self.starter[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        cap = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(self.captain[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        vice = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(vicecap_decay * self.vicecaptain[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        bench = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(
                so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order) * self.data.loc[p, f'GW{w}']
                for p in self.players
            )
            for w in self.gameweeks
        )

        txc = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(2 * self.triple[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        hits = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1) * (hit_val * self.hits[w])
            for w in self.gameweeks
        )

        ftv = so.expr_sum(
            (np.power(decay_gameweek, w - self.start - 1) if objective_type == 'linear' else 1)
            * (ft_val * (self.free_transfers[w] - 1))
            for w in self.gameweeks[1:]
        )

        itbv = so.expr_sum(
            (np.power(decay_gameweek, w - self.start - 1) if objective_type == 'linear' else 1)
            * (itb_val * self.in_the_bank[w])
            for w in self.gameweeks
        )

        triple_penalty = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * (triple_val * so.expr_sum(self.triple[p, w] for p in self.players))
            for w in self.gameweeks
        )

        bboost_penalty = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * (bboost_val * self.bboost[w])
            for w in self.gameweeks
        )

        freehit_penalty = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * (freehit_val * self.freehit[w])
            for w in self.gameweeks
        )

        wildcard_penalty = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * (wildcard_val * self.wildcard[w])
            for w in self.gameweeks
        )

        self.model.set_objective(
            -starter
            - cap
            - vice
            - bench
            - txc
            - ftv
            - itbv
            + hits
            + triple_penalty
            + bboost_penalty
            + freehit_penalty
            + wildcard_penalty,
            name='total_xp_obj',
            sense='N',
        )

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints((self.team[p, self.start - 1] == 1 for p in self.initial_team), name='initial_team')
        self.model.add_constraint(self.free_transfers[self.start] == self.rolling_transfer + 1, name='initial_ft')
        self.model.add_constraint(self.in_the_bank[self.start - 1] == self.bank, name='initial_itb')

        # Constraints
        # Chips
        # The chips must not be used more than once
        self.model.add_constraint(
            so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) <= int(not self.threexc_used),
            name='tc_once',
        )
        self.model.add_constraint(
            so.expr_sum(self.bboost[w] for w in self.gameweeks) <= int(not self.bboost_used), name='bb_once'
        )
        self.model.add_constraint(
            so.expr_sum(self.freehit[w] for w in self.gameweeks) <= int(not self.freehit_used), name='fh_once'
        )
        self.model.add_constraint(
            so.expr_sum(self.wildcard[w] for w in self.gameweeks) <= int(not self.wildcard_used), name='wc_once'
        )

        # The chips must not be used on the same GW
        self.model.add_constraints(
            (
                so.expr_sum(self.triple[p, w] for p in self.players)
                + self.bboost[w]
                + self.freehit[w]
                + self.wildcard[w]
                <= 1
                for w in self.gameweeks
            ),
            name='chip_once',
        )

        # Team
        # The number of players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'GK'] for p in self.players) == MAX_NUMBER_GOALKEEPERS
                for w in self.gameweeks
            ),
            name='gk_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'DF'] for p in self.players) == MAX_NUMBER_DEFENDERS
                for w in self.gameweeks
            ),
            name='def_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'MD'] for p in self.players) == MAX_NUMBER_MIDFIELDERS
                for w in self.gameweeks
            ),
            name='mid_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'FW'] for p in self.players) == MAX_NUMBER_FORWARDS
                for w in self.gameweeks
            ),
            name='for_limit',
        )

        # The number of players from a team must exceed three
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, team_name] for p in self.players)
                <= MAX_NUMBER_PLAYERS_PER_TEAM
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='team_limit',
        )

        # The number of Freehit players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'GK'] for p in self.players)
                == MAX_NUMBER_GOALKEEPERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='gk_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'DF'] for p in self.players)
                == MAX_NUMBER_DEFENDERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='def_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'MD'] for p in self.players)
                == MAX_NUMBER_MIDFIELDERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='mid_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'FW'] for p in self.players)
                == MAX_NUMBER_FORWARDS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='for_limit_fh',
        )

        # The number of Freehit players from a team must not exceed three
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, team_name] for p in self.players)
                <= 3 * self.freehit[w]
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='team_limit_fh',
        )

        # Starters
        # The formation must be valid i.e. Minimum one goalkeeper,
        # 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints(
            (so.expr_sum(self.starter[p, w] for p in self.players) == 11 + 4 * self.bboost[w] for w in self.gameweeks),
            name='11_starters',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'GK'] for p in self.players)
                == MIN_FORMATION['GK'] + self.bboost[w]
                for w in self.gameweeks
            ),
            name='gk_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'DF'] for p in self.players) >= MIN_FORMATION['DF']
                for w in self.gameweeks
            ),
            name='def_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'MD'] for p in self.players) >= MIN_FORMATION['MD']
                for w in self.gameweeks
            ),
            name='mid_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'FW'] for p in self.players) >= MIN_FORMATION['FW']
                for w in self.gameweeks
            ),
            name='for_min',
        )

        # Linearization constraints to limit the Freehit Team
        self.model.add_constraints(
            (self.starter[p, w] <= self.team_fh[p, w] + self.aux[p, w] for p in self.players for w in self.gameweeks),
            name='4.24',
        )
        self.model.add_constraints(
            (self.aux[p, w] <= self.team[p, w] for p in self.players for w in self.gameweeks), name='4.25'
        )
        self.model.add_constraints(
            (self.aux[p, w] <= 1 - self.freehit[w] for p in self.players for w in self.gameweeks), name='4.26'
        )

        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, team_name] * (self.data.loc[p, 'DF'] + self.data.loc[p, 'GK'])
                    for p in self.players
                )
                <= def_stack_limit * self.freehit[w]
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='def_stack_limit_fh',
        )

        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, team_name] * (self.data.loc[p, 'DF'] + self.data.loc[p, 'GK'])
                    for p in self.players
                )
                <= def_stack_limit
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='def_stack_limit',
        )

        # Captain
        # One captain (or one triple cap) must be picked once
        self.model.add_constraints(
            (so.expr_sum(self.captain[p, w] + self.triple[p, w] for p in self.players) == 1 for w in self.gameweeks),
            name='one_captain',
        )
        # One vice captain must be picked once
        self.model.add_constraints(
            (so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1 for w in self.gameweeks),
            name='one_vicecaptain',
        )
        # The captain, vice captain and triple captain must be starters and
        # must not be the same player
        self.model.add_constraints(
            (
                self.captain[p, w] + self.triple[p, w] + self.vicecaptain[p, w] <= self.starter[p, w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='cap_in_starters',
        )

        # Substitutions
        # The first substitute is a single goalkeeper
        self.model.add_constraints(
            (
                so.expr_sum(self.bench[p, w, 0] for p in self.players if self.data.loc[p, 'GK'] == 1) <= 1
                for w in self.gameweeks
            ),
            name='one_bench_gk',
        )
        # There must be a single substitute per bench spot
        self.model.add_constraints(
            (so.expr_sum(self.bench[p, w, o] for p in self.players) <= 1 for w in self.gameweeks for o in [1, 2, 3]),
            name='one_per_bench_spot',
        )

        # The players not started in the team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order)
                <= self.team[p, w] + 10000 * self.freehit[w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='bench_team',
        )
        # The players not started in the freehit team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order)
                <= self.team_fh[p, w] + 10000 * (1 - self.freehit[w])
                for p in self.players
                for w in self.gameweeks
            ),
            name='bench_team_fh',
        )

        # Budget
        sold_amount = {
            w: so.expr_sum(self.sell[p, w] * self.data.loc[p, 'selling_price'] for p in self.players)
            for w in self.gameweeks
        }
        bought_amount = {
            w: so.expr_sum(self.buy[p, w] * self.data.loc[p, 'purchase_price'] for p in self.players)
            for w in self.gameweeks
        }
        # The cost of the squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w] == self.in_the_bank[w - 1] + sold_amount[w] - bought_amount[w]
                for w in self.gameweeks
            ),
            name='budget',
        )
        # The team must be the same as the previous GW plus/minus transfers
        self.model.add_constraints(
            (
                self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] == self.team[p, w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='team_similarity',
        )
        # The player must not be sold and bought simultaneously
        self.model.add_constraints(
            (self.sell[p, w] + self.buy[p, w] <= 1 for p in self.players for w in self.gameweeks),
            name='single_buy_or_sell',
        )

        # The cost of the freehit squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w - 1]
                + so.expr_sum(self.team[p, w - 1] * self.data.loc[p, 'selling_price'] for p in self.players)
                >= so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'purchase_price'] for p in self.players)
                for w in self.gameweeks
            ),
            name='budget_fh',
        )
        # On Freehit GW the number of transfers must be zero
        self.model.add_constraints(
            (so.expr_sum(self.sell[p, w] for p in self.players) <= 15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_sold_fh',
        )
        self.model.add_constraints(
            (so.expr_sum(self.buy[p, w] for p in self.players) <= 15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_bought_fh',
        )

        # The number of players bought and sold is equal
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) == so.expr_sum(self.sell[p, w] for p in self.players)
                for w in self.gameweeks
            ),
            name='equal_transfers',
        )

        goalkeeper_cost = {
            w: so.expr_sum(
                self.team[p, w] * self.data.loc[p, 'purchase_price'] * self.data.loc[p, 'GK'] for p in self.players
            )
            for w in self.gameweeks
        }
        self.model.add_constraints(
            (goalkeeper_cost[w] <= goalkeeper_max_budget for w in self.gameweeks), name='goalkeeper_budget'
        )

        goalkeeper_cost_fh = {
            w: so.expr_sum(
                self.team_fh[p, w] * self.data.loc[p, 'purchase_price'] * self.data.loc[p, 'GK'] for p in self.players
            )
            for w in self.gameweeks
        }
        self.model.add_constraints(
            (goalkeeper_cost_fh[w] <= goalkeeper_max_budget for w in self.gameweeks), name='goalkeeper_budget'
        )

        # Transfers
        # The rolling transfer must be equal to the number of free
        # transfers not used (+ 1)
        self.model.add_constraints(
            (
                15 * self.wildcard[w]
                + self.free_transfers[w]
                - so.expr_sum(self.buy[p, w] for p in self.players)
                + 1
                + self.hits[w]
                >= self.free_transfers[w + 1]
                for w in self.gameweeks
            ),
            name='rolling_ft_rel',
        )
        # The hits value is zero only when the number of FT is 2
        self.model.add_constraints(
            (10000 * (2 - self.free_transfers[w + 1]) >= self.hits[w] for w in self.gameweeks), name='4.42'
        )
        # The minimum number of FT is 1
        self.model.add_constraints((self.free_transfers[w + 1] >= 1 for w in self.gameweeks), name='min_ft')
        # The maximum number of FT is 2 on regular GWs
        self.model.add_constraints(
            (self.free_transfers[w + 1] <= 2 - self.wildcard[w] - self.freehit[w] for w in self.gameweeks),
            name='max_ft',
        )

    def advanced_wildcard(self, params: dict) -> tuple:  # noqa: C901, PLR0912, PLR0915
        freehit_gw = params.get('freehit_gw', -1)
        bboost_gw = params.get('bboost_gw', -1)
        threexc_gw = params.get('threexc_gw', -1)
        objective_type = params.get('objective_type', 'decay')
        decay_gameweek = params.get('decay_gameweek', [0.9, 0.8, 0.7])
        vicecap_decay = params.get('vicecap_decay', 0.1)
        decay_bench = params.get('decay_bench', [0.1, 0.1, 0.1, 0.1])
        ft_val = params.get('ft_val', 0)
        itb_val = params.get('itb_val', 0)
        hit_val = params.get('hit_val', 6)
        goalkeeper_max_budget = params.get('goalkeeper_max_budget', 100)
        def_stack_limit = params.get('def_stack_limit', 3)

        gw_out_of_horizon_msg = 'Select a GW within the horizon.'
        if freehit_gw >= self.horizon:
            raise ValueError(gw_out_of_horizon_msg)
        if bboost_gw >= self.horizon:
            raise ValueError(gw_out_of_horizon_msg)
        if threexc_gw >= self.horizon:
            raise ValueError(gw_out_of_horizon_msg)

        freehit_used_msg = 'Freehit chip was already used.'
        if self.freehit_used and freehit_gw >= 0:
            raise ValueError(freehit_used_msg)
        bboost_used_msg = 'Bench boost chip was already used.'
        if self.bboost_used and bboost_gw >= 0:
            raise ValueError(bboost_used_msg)
        threexc_used_msg = 'Tripple captain chip was already used.'
        if self.threexc_used and threexc_gw >= 0:
            raise ValueError(threexc_used_msg)

        # Longterm Model
        model_name = 'longterm'
        self.model = so.Model(name=model_name + '_model')

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(self.players, self.all_gameweeks, name='team', vartype=so.binary)
        self.team_fh = self.model.add_variables(self.players, self.gameweeks, name='team_fh', vartype=so.binary)
        self.starter = self.model.add_variables(self.players, self.gameweeks, name='starter', vartype=so.binary)
        self.bench = self.model.add_variables(self.players, self.gameweeks, order, name='bench', vartype=so.binary)

        self.captain = self.model.add_variables(self.players, self.gameweeks, name='captain', vartype=so.binary)
        self.vicecaptain = self.model.add_variables(self.players, self.gameweeks, name='vicecaptain', vartype=so.binary)

        self.buy = self.model.add_variables(self.players, self.gameweeks, name='buy', vartype=so.binary)
        self.sell = self.model.add_variables(self.players, self.gameweeks, name='sell', vartype=so.binary)

        self.triple = self.model.add_variables(self.players, self.gameweeks, name='3xc', vartype=so.binary)
        self.bboost = self.model.add_variables(self.gameweeks, name='bb', vartype=so.binary)
        self.freehit = self.model.add_variables(self.gameweeks, name='fh', vartype=so.binary)
        self.wildcard = self.model.add_variables(self.gameweeks, name='wc', vartype=so.binary)

        self.aux = self.model.add_variables(self.players, self.all_gameweeks, name='aux', vartype=so.binary)
        self.free_transfers = self.model.add_variables(
            np.arange(self.start - 1, self.start + self.period + 1), name='ft', vartype=so.integer, lb=0
        )
        self.hits = self.model.add_variables(self.all_gameweeks, name='hits', vartype=so.integer, lb=0)
        self.in_the_bank = self.model.add_variables(self.all_gameweeks, name='itb', vartype=so.continuous, lb=0)

        # Objective: maximize total expected points
        starter = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(self.starter[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        cap = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(self.captain[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        vice = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(vicecap_decay * self.vicecaptain[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        bench = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(
                so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order) * self.data.loc[p, f'GW{w}']
                for p in self.players
            )
            for w in self.gameweeks
        )

        txc = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(2 * self.triple[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        hits = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1) * (hit_val * self.hits[w])
            for w in self.gameweeks
        )

        itbv = so.expr_sum(
            (np.power(decay_gameweek, w - self.start - 1) if objective_type == 'linear' else 1)
            * (itb_val * self.in_the_bank[w])
            for w in self.gameweeks
        )

        self.model.set_objective(-starter - cap - vice - bench - txc - itbv + hits, name='total_xp_obj', sense='N')

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints((self.team[p, self.start - 1] == 1 for p in self.initial_team), name='initial_team')
        self.model.add_constraint(self.in_the_bank[self.start - 1] == self.bank, name='initial_itb')

        # Constraints
        # Chips
        # The chips must not be used more than once
        self.model.add_constraint(
            so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) <= int(not self.threexc_used),
            name='tc_once',
        )
        self.model.add_constraint(
            so.expr_sum(self.bboost[w] for w in self.gameweeks) <= int(not self.bboost_used), name='bb_once'
        )
        self.model.add_constraint(
            so.expr_sum(self.freehit[w] for w in self.gameweeks) <= int(not self.freehit_used), name='fh_once'
        )
        self.model.add_constraint(
            so.expr_sum(self.wildcard[w] for w in self.gameweeks) <= int(not self.wildcard_used), name='wc_once'
        )

        # The chips must not be used on the same GW
        self.model.add_constraints(
            (
                so.expr_sum(self.triple[p, w] for p in self.players)
                + self.bboost[w]
                + self.freehit[w]
                + self.wildcard[w]
                <= 1
                for w in self.gameweeks
            ),
            name='chip_once',
        )

        # The chips must be used on the selected GW
        # For printing
        self.model.add_constraint((self.wildcard[self.start] == 1), name='wildcard_gw')

        if bboost_gw + 1:
            self.model.add_constraint(self.bboost[self.start + bboost_gw] == 1, name='bboost_gw')
        else:
            self.model.add_constraint(so.expr_sum(self.bboost[w] for w in self.gameweeks) == 0, name='bboost_unused')

        if threexc_gw + 1:
            self.model.add_constraint(
                so.expr_sum(self.triple[p, self.start + threexc_gw] for p in self.players) == 1, name='triple_gw'
            )
        else:
            self.model.add_constraint(
                so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) == 0, name='triple_unused'
            )

        if freehit_gw + 1:
            self.model.add_constraint(self.freehit[self.start + freehit_gw] == 1, name='freehit_gw')
        else:
            self.model.add_constraint(so.expr_sum(self.freehit[w] for w in self.gameweeks) == 0, name='freehit_unused')

        # Team
        # The number of players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'GK'] for p in self.players) == MAX_NUMBER_GOALKEEPERS
                for w in self.gameweeks
            ),
            name='gk_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'DF'] for p in self.players) == MAX_NUMBER_DEFENDERS
                for w in self.gameweeks
            ),
            name='def_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'MD'] for p in self.players) == MAX_NUMBER_MIDFIELDERS
                for w in self.gameweeks
            ),
            name='mid_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'FW'] for p in self.players) == MAX_NUMBER_FORWARDS
                for w in self.gameweeks
            ),
            name='for_limit',
        )

        # The number of players from a team must exceed three
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, team_name] for p in self.players)
                <= MAX_NUMBER_PLAYERS_PER_TEAM
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='team_limit',
        )

        # The number of Freehit players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'GK'] for p in self.players)
                == MAX_NUMBER_GOALKEEPERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='gk_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'DF'] for p in self.players)
                == MAX_NUMBER_DEFENDERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='def_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'MD'] for p in self.players)
                == MAX_NUMBER_MIDFIELDERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='mid_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'FW'] for p in self.players)
                == MAX_NUMBER_FORWARDS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='for_limit_fh',
        )

        # The number of Freehit players from a team must not exceed three
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, team_name] for p in self.players)
                <= 3 * self.freehit[w]
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='team_limit_fh',
        )

        # Starters
        # The formation must be valid i.e. Minimum one goalkeeper,
        # 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints(
            (so.expr_sum(self.starter[p, w] for p in self.players) == 11 + 4 * self.bboost[w] for w in self.gameweeks),
            name='11_starters',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'GK'] for p in self.players) == 1 + self.bboost[w]
                for w in self.gameweeks
            ),
            name='gk_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'DF'] for p in self.players) >= MIN_FORMATION['DF']
                for w in self.gameweeks
            ),
            name='def_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'MD'] for p in self.players) >= MIN_FORMATION['MD']
                for w in self.gameweeks
            ),
            name='mid_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'FW'] for p in self.players) >= 1
                for w in self.gameweeks
            ),
            name='for_min',
        )

        # Linearization constraints to limit the Freehit Team
        self.model.add_constraints(
            (self.starter[p, w] <= self.team_fh[p, w] + self.aux[p, w] for p in self.players for w in self.gameweeks),
            name='4.24',
        )
        self.model.add_constraints(
            (self.aux[p, w] <= self.team[p, w] for p in self.players for w in self.gameweeks), name='4.25'
        )
        self.model.add_constraints(
            (self.aux[p, w] <= 1 - self.freehit[w] for p in self.players for w in self.gameweeks), name='4.26'
        )

        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, team_name] * (self.data.loc[p, 'DF'] + self.data.loc[p, 'GK'])
                    for p in self.players
                )
                <= def_stack_limit * self.freehit[w]
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='def_stack_limit_fh',
        )

        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, team_name] * (self.data.loc[p, 'DF'] + self.data.loc[p, 'GK'])
                    for p in self.players
                )
                <= def_stack_limit
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='def_stack_limit',
        )

        # Captain
        # One captain (or one triple cap) must be picked
        self.model.add_constraints(
            (so.expr_sum(self.captain[p, w] + self.triple[p, w] for p in self.players) == 1 for w in self.gameweeks),
            name='one_captain',
        )
        # One vice captain must be picked
        self.model.add_constraints(
            (so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1 for w in self.gameweeks),
            name='one_vicecaptain',
        )
        # The captain, vice captain and triple captain must be starters and
        # must not be the same player
        self.model.add_constraints(
            (
                self.captain[p, w] + self.triple[p, w] + self.vicecaptain[p, w] <= self.starter[p, w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='cap_in_starters',
        )

        # Substitutions
        # The first substitute is a single goalkeeper
        self.model.add_constraints(
            (
                so.expr_sum(self.bench[p, w, 0] for p in self.players if self.data.loc[p, 'GK'] == 1) <= 1
                for w in self.gameweeks
            ),
            name='one_bench_gk',
        )
        # There must be a single substitute per bench spot
        self.model.add_constraints(
            (so.expr_sum(self.bench[p, w, o] for p in self.players) <= 1 for w in self.gameweeks for o in [1, 2, 3]),
            name='one_per_bench_spot',
        )

        # The players not started in the team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order)
                <= self.team[p, w] + 10000 * self.freehit[w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='bench_team',
        )
        # The players not started in the freehit team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order)
                <= self.team_fh[p, w] + 10000 * (1 - self.freehit[w])
                for p in self.players
                for w in self.gameweeks
            ),
            name='bench_team_fh',
        )

        # Budget
        sold_amount = {
            w: so.expr_sum(self.sell[p, w] * self.data.loc[p, 'selling_price'] for p in self.players)
            for w in self.gameweeks
        }
        bought_amount = {
            w: so.expr_sum(self.buy[p, w] * self.data.loc[p, 'purchase_price'] for p in self.players)
            for w in self.gameweeks
        }
        # The cost of the squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w] == self.in_the_bank[w - 1] + sold_amount[w] - bought_amount[w]
                for w in self.gameweeks
            ),
            name='budget',
        )
        # The team must be the same as the previous GW plus/minus transfers
        self.model.add_constraints(
            (
                self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] == self.team[p, w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='team_similarity',
        )
        # The player must not be sold and bought simultaneously
        self.model.add_constraints(
            (self.sell[p, w] + self.buy[p, w] <= 1 for p in self.players for w in self.gameweeks),
            name='single_buy_or_sell',
        )

        # The cost of the freehit squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w - 1]
                + so.expr_sum(self.team[p, w - 1] * self.data.loc[p, 'selling_price'] for p in self.players)
                >= so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'purchase_price'] for p in self.players)
                for w in self.gameweeks
            ),
            name='budget_fh',
        )
        # On Freehit GW the number of transfers must be zero
        self.model.add_constraints(
            (so.expr_sum(self.sell[p, w] for p in self.players) <= 15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_sold_fh',
        )
        self.model.add_constraints(
            (so.expr_sum(self.buy[p, w] for p in self.players) <= 15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_bought_fh',
        )

        # The number of players bought and sold is equal
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) == so.expr_sum(self.sell[p, w] for p in self.players)
                for w in self.gameweeks
            ),
            name='equal_transfers',
        )

        goalkeeper_cost = {
            w: so.expr_sum(
                self.team[p, w] * self.data.loc[p, 'purchase_price'] * self.data.loc[p, 'GK'] for p in self.players
            )
            for w in self.gameweeks
        }
        self.model.add_constraints(
            (goalkeeper_cost[w] <= goalkeeper_max_budget for w in self.gameweeks), name='goalkeeper_budget'
        )

        goalkeeper_cost_fh = {
            w: so.expr_sum(
                self.team_fh[p, w] * self.data.loc[p, 'purchase_price'] * self.data.loc[p, 'GK'] for p in self.players
            )
            for w in self.gameweeks
        }
        self.model.add_constraints(
            (goalkeeper_cost_fh[w] <= goalkeeper_max_budget for w in self.gameweeks), name='goalkeeper_budget'
        )

        # Transfers
        # The rolling transfer must be equal to the number of free
        # transfers not used (+ 1)
        self.model.add_constraints(
            (
                15 * self.wildcard[w]
                + self.free_transfers[w]
                - so.expr_sum(self.buy[p, w] for p in self.players)
                + 1
                + self.hits[w]
                >= self.free_transfers[w + 1]
                for w in self.gameweeks
            ),
            name='rolling_ft_rel',
        )
        # The hits value is zero only when the number of FT is 2
        self.model.add_constraints(
            (10000 * (2 - self.free_transfers[w + 1]) >= self.hits[w] for w in self.gameweeks), name='4.42'
        )
        # The minimum number of FT is 1
        self.model.add_constraints((self.free_transfers[w + 1] >= 1 for w in self.gameweeks), name='min_ft')
        # The maximum number of FT is 2 on regular GWs
        self.model.add_constraints(
            (self.free_transfers[w + 1] <= 2 - self.wildcard[w] - self.freehit[w] for w in self.gameweeks),
            name='max_ft',
        )

        # Enforce longterm planning by having no transfer past WC GW
        # NOTE: This messes up Rolling transfer logic but it matters not
        # Since we're not using FTV in objective function.
        self.model.add_constraints(
            (so.expr_sum(self.buy[p, w] for p in self.players) == 0 for w in self.gameweeks[1:]), name='no_transfer'
        )

        # Solve
        self.model.export_mps(filename=f'tmp/{model_name}.mps')
        command = f'cbc tmp/{model_name}.mps solve solu ' + f'tmp/{model_name}_solution.txt'

        process = Popen(command, shell=True, stdout=DEVNULL)  # noqa: S602
        process.wait()

        # Reset variables for next passes
        for v in self.model.get_variables():
            v.set_value(0)

        with Path(f'tmp/{model_name}_solution.txt').open('r') as f:
            for line in f:
                if 'objective value' in line:
                    continue
                words = line.split()
                var = self.model.get_variable(words[1])
                var.set_value(float(words[2]))

        pretty_print(
            self.data,
            self.start,
            self.period,
            self.team,
            self.team_fh,
            self.starter,
            self.bench,
            self.captain,
            self.vicecaptain,
            self.buy,
            self.sell,
            self.free_transfers,
            self.hits,
            self.in_the_bank,
            self.model.get_objective_value(),
            self.freehit,
            self.wildcard,
            self.bboost,
            self.triple,
            nb_suboptimal=model_name,
        )

        # GW
        print('\n----------')
        # Get all players selected by longterm WC
        wc_team = [p for p in self.players if self.team[p, self.start].get_value()]

        # Medium range planning Model
        model_name = 'medium'
        self.model = so.Model(name=model_name + '_model')

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(self.players, self.all_gameweeks, name='team', vartype=so.binary)
        self.team_fh = self.model.add_variables(self.players, self.gameweeks, name='team_fh', vartype=so.binary)
        self.starter = self.model.add_variables(self.players, self.gameweeks, name='starter', vartype=so.binary)
        self.bench = self.model.add_variables(self.players, self.gameweeks, order, name='bench', vartype=so.binary)

        self.captain = self.model.add_variables(self.players, self.gameweeks, name='captain', vartype=so.binary)
        self.vicecaptain = self.model.add_variables(self.players, self.gameweeks, name='vicecaptain', vartype=so.binary)

        self.buy = self.model.add_variables(self.players, self.gameweeks, name='buy', vartype=so.binary)
        self.sell = self.model.add_variables(self.players, self.gameweeks, name='sell', vartype=so.binary)

        self.triple = self.model.add_variables(self.players, self.gameweeks, name='3xc', vartype=so.binary)
        self.bboost = self.model.add_variables(self.gameweeks, name='bb', vartype=so.binary)
        self.freehit = self.model.add_variables(self.gameweeks, name='fh', vartype=so.binary)
        self.wildcard = self.model.add_variables(self.gameweeks, name='wc', vartype=so.binary)

        self.aux = self.model.add_variables(self.players, self.all_gameweeks, name='aux', vartype=so.binary)
        self.free_transfers = self.model.add_variables(
            np.arange(self.start - 1, self.start + self.period + 1), name='ft', vartype=so.integer, lb=0
        )
        self.hits = self.model.add_variables(self.all_gameweeks, name='hits', vartype=so.integer, lb=0)
        self.in_the_bank = self.model.add_variables(self.all_gameweeks, name='itb', vartype=so.continuous, lb=0)

        # Objective: maximize total expected points
        starter = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(self.starter[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        cap = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(self.captain[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        vice = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(vicecap_decay * self.vicecaptain[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        bench = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(
                so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order) * self.data.loc[p, f'GW{w}']
                for p in self.players
            )
            for w in self.gameweeks
        )

        txc = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(2 * self.triple[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        hits = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1) * (hit_val * self.hits[w])
            for w in self.gameweeks
        )

        ftv = so.expr_sum(
            (np.power(decay_gameweek, w - self.start - 1) if objective_type == 'linear' else 1)
            * (ft_val * (self.free_transfers[w] - 1))
            for w in self.gameweeks[1:]
        )

        itbv = so.expr_sum(
            (np.power(decay_gameweek, w - self.start - 1) if objective_type == 'linear' else 1)
            * (itb_val * self.in_the_bank[w])
            for w in self.gameweeks
        )

        self.model.set_objective(
            -starter - cap - vice - bench - txc - ftv - itbv + hits, name='total_xp_obj', sense='N'
        )

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints((self.team[p, self.start - 1] == 1 for p in self.initial_team), name='initial_team')
        self.model.add_constraint(self.in_the_bank[self.start - 1] == self.bank, name='initial_itb')
        number_players_for_longterm = 10
        self.model.add_constraint(
            (so.expr_sum(self.team[p, self.start] for p in wc_team) >= number_players_for_longterm),
            name='initial_wc_team',
        )

        # Constraints
        # Chips
        # The chips must not be used more than once
        self.model.add_constraint(
            so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) <= int(not self.threexc_used),
            name='tc_once',
        )
        self.model.add_constraint(
            so.expr_sum(self.bboost[w] for w in self.gameweeks) <= int(not self.bboost_used), name='bb_once'
        )
        self.model.add_constraint(
            so.expr_sum(self.freehit[w] for w in self.gameweeks) <= int(not self.freehit_used), name='fh_once'
        )
        self.model.add_constraint(
            so.expr_sum(self.wildcard[w] for w in self.gameweeks) <= int(not self.wildcard_used), name='wc_once'
        )

        # The chips must not be used on the same GW
        self.model.add_constraints(
            (
                so.expr_sum(self.triple[p, w] for p in self.players)
                + self.bboost[w]
                + self.freehit[w]
                + self.wildcard[w]
                <= 1
                for w in self.gameweeks
            ),
            name='chip_once',
        )

        # The chips must be used on the selected GW
        self.model.add_constraint((self.wildcard[self.start] == 1), name='wildcard_gw')

        if bboost_gw + 1:
            self.model.add_constraint(self.bboost[self.start + bboost_gw] == 1, name='bboost_gw')
        else:
            self.model.add_constraint(so.expr_sum(self.bboost[w] for w in self.gameweeks) == 0, name='bboost_unused')

        if threexc_gw + 1:
            self.model.add_constraint(
                so.expr_sum(self.triple[p, self.start + threexc_gw] for p in self.players) == 1, name='triple_gw'
            )
        else:
            self.model.add_constraint(
                so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) == 0, name='triple_unused'
            )

        if freehit_gw + 1:
            self.model.add_constraint(self.freehit[self.start + freehit_gw] == 1, name='freehit_gw')
        else:
            self.model.add_constraint(so.expr_sum(self.freehit[w] for w in self.gameweeks) == 0, name='freehit_unused')

        # Team
        # The number of players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'GK'] for p in self.players) == MAX_NUMBER_GOALKEEPERS
                for w in self.gameweeks
            ),
            name='gk_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'DF'] for p in self.players) == MAX_NUMBER_DEFENDERS
                for w in self.gameweeks
            ),
            name='def_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'MD'] for p in self.players) == MAX_NUMBER_MIDFIELDERS
                for w in self.gameweeks
            ),
            name='mid_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'FW'] for p in self.players) == MAX_NUMBER_FORWARDS
                for w in self.gameweeks
            ),
            name='for_limit',
        )

        # The number of players from a team must exceed three
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, team_name] for p in self.players)
                <= MAX_NUMBER_PLAYERS_PER_TEAM
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='team_limit',
        )

        # The number of Freehit players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'GK'] for p in self.players)
                == MAX_NUMBER_GOALKEEPERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='gk_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'DF'] for p in self.players)
                == MAX_NUMBER_DEFENDERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='def_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'MD'] for p in self.players)
                == MAX_NUMBER_MIDFIELDERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='mid_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'FW'] for p in self.players)
                == MAX_NUMBER_FORWARDS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='for_limit_fh',
        )

        # The number of Freehit players from a team must not exceed three
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, team_name] for p in self.players)
                <= 3 * self.freehit[w]
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='team_limit_fh',
        )

        # Starters
        # The formation must be valid i.e. Minimum one goalkeeper,
        # 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints(
            (so.expr_sum(self.starter[p, w] for p in self.players) == 11 + 4 * self.bboost[w] for w in self.gameweeks),
            name='11_starters',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'GK'] for p in self.players)
                == MIN_FORMATION['GK'] + self.bboost[w]
                for w in self.gameweeks
            ),
            name='gk_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'DF'] for p in self.players) >= MIN_FORMATION['DF']
                for w in self.gameweeks
            ),
            name='def_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'MD'] for p in self.players) >= MIN_FORMATION['MD']
                for w in self.gameweeks
            ),
            name='mid_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'FW'] for p in self.players) >= MIN_FORMATION['FW']
                for w in self.gameweeks
            ),
            name='for_min',
        )

        # Linearization constraints to limit the Freehit Team
        self.model.add_constraints(
            (self.starter[p, w] <= self.team_fh[p, w] + self.aux[p, w] for p in self.players for w in self.gameweeks),
            name='4.24',
        )
        self.model.add_constraints(
            (self.aux[p, w] <= self.team[p, w] for p in self.players for w in self.gameweeks), name='4.25'
        )
        self.model.add_constraints(
            (self.aux[p, w] <= 1 - self.freehit[w] for p in self.players for w in self.gameweeks), name='4.26'
        )

        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, team_name] * (self.data.loc[p, 'DF'] + self.data.loc[p, 'GK'])
                    for p in self.players
                )
                <= def_stack_limit * self.freehit[w]
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='def_stack_limit_fh',
        )

        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, team_name] * (self.data.loc[p, 'DF'] + self.data.loc[p, 'GK'])
                    for p in self.players
                )
                <= def_stack_limit
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='def_stack_limit',
        )

        # Captain
        # One captain (or one triple cap) must be picked
        self.model.add_constraints(
            (so.expr_sum(self.captain[p, w] + self.triple[p, w] for p in self.players) == 1 for w in self.gameweeks),
            name='one_captain',
        )
        # One vice captain must be picked
        self.model.add_constraints(
            (so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1 for w in self.gameweeks),
            name='one_vicecaptain',
        )
        # The captain, vice captain and triple captain must be starters and
        # must not be the same player
        self.model.add_constraints(
            (
                self.captain[p, w] + self.triple[p, w] + self.vicecaptain[p, w] <= self.starter[p, w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='cap_in_starters',
        )

        # Substitutions
        # The first substitute is a single goalkeeper
        self.model.add_constraints(
            (
                so.expr_sum(self.bench[p, w, 0] for p in self.players if self.data.loc[p, 'GK'] == 1) <= 1
                for w in self.gameweeks
            ),
            name='one_bench_gk',
        )
        # There must be a single substitute per bench spot
        self.model.add_constraints(
            (so.expr_sum(self.bench[p, w, o] for p in self.players) <= 1 for w in self.gameweeks for o in [1, 2, 3]),
            name='one_per_bench_spot',
        )

        # The players not started in the team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order)
                <= self.team[p, w] + 10000 * self.freehit[w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='bench_team',
        )
        # The players not started in the freehit team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order)
                <= self.team_fh[p, w] + 10000 * (1 - self.freehit[w])
                for p in self.players
                for w in self.gameweeks
            ),
            name='bench_team_fh',
        )

        # Budget
        sold_amount = {
            w: so.expr_sum(self.sell[p, w] * self.data.loc[p, 'selling_price'] for p in self.players)
            for w in self.gameweeks
        }
        bought_amount = {
            w: so.expr_sum(self.buy[p, w] * self.data.loc[p, 'purchase_price'] for p in self.players)
            for w in self.gameweeks
        }
        # The cost of the squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w] == self.in_the_bank[w - 1] + sold_amount[w] - bought_amount[w]
                for w in self.gameweeks
            ),
            name='budget',
        )
        # The team must be the same as the previous GW plus/minus transfers
        self.model.add_constraints(
            (
                self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] == self.team[p, w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='team_similarity',
        )
        # The player must not be sold and bought simultaneously
        self.model.add_constraints(
            (self.sell[p, w] + self.buy[p, w] <= 1 for p in self.players for w in self.gameweeks),
            name='single_buy_or_sell',
        )

        # The cost of the freehit squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w - 1]
                + so.expr_sum(self.team[p, w - 1] * self.data.loc[p, 'selling_price'] for p in self.players)
                >= so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'purchase_price'] for p in self.players)
                for w in self.gameweeks
            ),
            name='budget_fh',
        )
        # On Freehit GW the number of transfers must be zero
        self.model.add_constraints(
            (so.expr_sum(self.sell[p, w] for p in self.players) <= 15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_sold_fh',
        )
        self.model.add_constraints(
            (so.expr_sum(self.buy[p, w] for p in self.players) <= 15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_bought_fh',
        )

        # The number of players bought and sold is equal
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) == so.expr_sum(self.sell[p, w] for p in self.players)
                for w in self.gameweeks
            ),
            name='equal_transfers',
        )

        goalkeeper_cost = {
            w: so.expr_sum(
                self.team[p, w] * self.data.loc[p, 'purchase_price'] * self.data.loc[p, 'GK'] for p in self.players
            )
            for w in self.gameweeks
        }
        self.model.add_constraints(
            (goalkeeper_cost[w] <= goalkeeper_max_budget for w in self.gameweeks), name='goalkeeper_budget'
        )

        goalkeeper_cost_fh = {
            w: so.expr_sum(
                self.team_fh[p, w] * self.data.loc[p, 'purchase_price'] * self.data.loc[p, 'GK'] for p in self.players
            )
            for w in self.gameweeks
        }
        self.model.add_constraints(
            (goalkeeper_cost_fh[w] <= goalkeeper_max_budget for w in self.gameweeks), name='goalkeeper_budget'
        )

        # Transfers
        # The rolling transfer must be equal to the number of free
        # transfers not used (+ 1)
        self.model.add_constraints(
            (
                15 * self.wildcard[w]
                + self.free_transfers[w]
                - so.expr_sum(self.buy[p, w] for p in self.players)
                + 1
                + self.hits[w]
                >= self.free_transfers[w + 1]
                for w in self.gameweeks
            ),
            name='rolling_ft_rel',
        )
        # The hits value is zero only when the number of FT is 2
        self.model.add_constraints(
            (10000 * (2 - self.free_transfers[w + 1]) >= self.hits[w] for w in self.gameweeks), name='4.42'
        )
        # The minimum number of FT is 1
        self.model.add_constraints((self.free_transfers[w + 1] >= 1 for w in self.gameweeks), name='min_ft')
        # The maximum number of FT is 2 on regular GWs
        self.model.add_constraints(
            (self.free_transfers[w + 1] <= 2 - self.wildcard[w] - self.freehit[w] for w in self.gameweeks),
            name='max_ft',
        )

        # Enforce medium term planning by having 1 transfer past WC GW
        self.model.add_constraints((self.hits[w] <= 0 for w in self.gameweeks), name='hits_max')

        # Solve
        self.model.export_mps(filename=f'tmp/{model_name}.mps')
        command = f'cbc tmp/{model_name}.mps solve solu ' + f'tmp/{model_name}_solution.txt'

        process = Popen(command, shell=True, stdout=DEVNULL)  # noqa: S602
        process.wait()

        # Reset variables for next passes
        for v in self.model.get_variables():
            v.set_value(0)

        with Path(f'tmp/{model_name}_solution.txt').open('r') as f:
            for line in f:
                if 'objective value' in line:
                    continue
                words = line.split()
                var = self.model.get_variable(words[1])
                var.set_value(float(words[2]))

        pretty_print(
            self.data,
            self.start,
            self.period,
            self.team,
            self.team_fh,
            self.starter,
            self.bench,
            self.captain,
            self.vicecaptain,
            self.buy,
            self.sell,
            self.free_transfers,
            self.hits,
            self.in_the_bank,
            self.model.get_objective_value(),
            self.freehit,
            self.wildcard,
            self.bboost,
            self.triple,
            nb_suboptimal=model_name,
        )

        # GW
        print('\n----------')
        # Get all players selected by medium term WC
        wc_team = [p for p in self.players if self.team[p, self.start].get_value()]

        # Short range planning Model
        model_name = 'short'
        self.model = so.Model(name=model_name + '_model')

        order = [0, 1, 2, 3]
        # Variables
        self.team = self.model.add_variables(self.players, self.all_gameweeks, name='team', vartype=so.binary)
        self.team_fh = self.model.add_variables(self.players, self.gameweeks, name='team_fh', vartype=so.binary)
        self.starter = self.model.add_variables(self.players, self.gameweeks, name='starter', vartype=so.binary)
        self.bench = self.model.add_variables(self.players, self.gameweeks, order, name='bench', vartype=so.binary)

        self.captain = self.model.add_variables(self.players, self.gameweeks, name='captain', vartype=so.binary)
        self.vicecaptain = self.model.add_variables(self.players, self.gameweeks, name='vicecaptain', vartype=so.binary)

        self.buy = self.model.add_variables(self.players, self.gameweeks, name='buy', vartype=so.binary)
        self.sell = self.model.add_variables(self.players, self.gameweeks, name='sell', vartype=so.binary)

        self.triple = self.model.add_variables(self.players, self.gameweeks, name='3xc', vartype=so.binary)
        self.bboost = self.model.add_variables(self.gameweeks, name='bb', vartype=so.binary)
        self.freehit = self.model.add_variables(self.gameweeks, name='fh', vartype=so.binary)
        self.wildcard = self.model.add_variables(self.gameweeks, name='wc', vartype=so.binary)

        self.aux = self.model.add_variables(self.players, self.all_gameweeks, name='aux', vartype=so.binary)
        self.free_transfers = self.model.add_variables(
            np.arange(self.start - 1, self.start + self.period + 1), name='ft', vartype=so.integer, lb=0
        )
        self.hits = self.model.add_variables(self.all_gameweeks, name='hits', vartype=so.integer, lb=0)
        self.in_the_bank = self.model.add_variables(self.all_gameweeks, name='itb', vartype=so.continuous, lb=0)

        # Objective: maximize total expected points
        starter = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(self.starter[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        cap = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(self.captain[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        vice = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(vicecap_decay * self.vicecaptain[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        bench = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(
                so.expr_sum(decay_bench[o] * self.bench[p, w, o] for o in order) * self.data.loc[p, f'GW{w}']
                for p in self.players
            )
            for w in self.gameweeks
        )

        txc = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1)
            * so.expr_sum(2 * self.triple[p, w] * self.data.loc[p, f'GW{w}'] for p in self.players)
            for w in self.gameweeks
        )

        hits = so.expr_sum(
            (np.power(decay_gameweek, w - self.start) if objective_type == 'linear' else 1) * (hit_val * self.hits[w])
            for w in self.gameweeks
        )

        ftv = so.expr_sum(
            (np.power(decay_gameweek, w - self.start - 1) if objective_type == 'linear' else 1)
            * (ft_val * (self.free_transfers[w] - 1))
            for w in self.gameweeks[1:]
        )

        itbv = so.expr_sum(
            (np.power(decay_gameweek, w - self.start - 1) if objective_type == 'linear' else 1)
            * (itb_val * self.in_the_bank[w])
            for w in self.gameweeks
        )

        self.model.set_objective(
            -starter - cap - vice - bench - txc - ftv - itbv + hits, name='total_xp_obj', sense='N'
        )

        # Initial conditions: set team and FT depending on the team
        self.model.add_constraints((self.team[p, self.start - 1] == 1 for p in self.initial_team), name='initial_team')
        self.model.add_constraint(self.in_the_bank[self.start - 1] == self.bank, name='initial_itb')
        number_players_for_midterm = 13
        self.model.add_constraint(
            (so.expr_sum(self.team[p, self.start] for p in wc_team) >= number_players_for_midterm),
            name='initial_wc_team',
        )

        # Constraints
        # Chips
        # The chips must not be used more than once
        self.model.add_constraint(
            so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) <= int(not self.threexc_used),
            name='tc_once',
        )
        self.model.add_constraint(
            so.expr_sum(self.bboost[w] for w in self.gameweeks) <= int(not self.bboost_used), name='bb_once'
        )
        self.model.add_constraint(
            so.expr_sum(self.freehit[w] for w in self.gameweeks) <= int(not self.freehit_used), name='fh_once'
        )
        self.model.add_constraint(
            so.expr_sum(self.wildcard[w] for w in self.gameweeks) <= int(not self.wildcard_used), name='wc_once'
        )

        # The chips must not be used on the same GW
        self.model.add_constraints(
            (
                so.expr_sum(self.triple[p, w] for p in self.players)
                + self.bboost[w]
                + self.freehit[w]
                + self.wildcard[w]
                <= 1
                for w in self.gameweeks
            ),
            name='chip_once',
        )

        # The chips must be used on the selected GW
        self.model.add_constraint((self.wildcard[self.start] == 1), name='wildcard_gw')

        if bboost_gw + 1:
            self.model.add_constraint(self.bboost[self.start + bboost_gw] == 1, name='bboost_gw')
        else:
            self.model.add_constraint(so.expr_sum(self.bboost[w] for w in self.gameweeks) == 0, name='bboost_unused')

        if threexc_gw + 1:
            self.model.add_constraint(
                so.expr_sum(self.triple[p, self.start + threexc_gw] for p in self.players) == 1, name='triple_gw'
            )
        else:
            self.model.add_constraint(
                so.expr_sum(self.triple[p, w] for p in self.players for w in self.gameweeks) == 0, name='triple_unused'
            )

        if freehit_gw + 1:
            self.model.add_constraint(self.freehit[self.start + freehit_gw] == 1, name='freehit_gw')
        else:
            self.model.add_constraint(so.expr_sum(self.freehit[w] for w in self.gameweeks) == 0, name='freehit_unused')

        # Team
        # The number of players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'GK'] for p in self.players) == MAX_NUMBER_GOALKEEPERS
                for w in self.gameweeks
            ),
            name='gk_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'DF'] for p in self.players) == MAX_NUMBER_DEFENDERS
                for w in self.gameweeks
            ),
            name='def_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'MD'] for p in self.players) == MAX_NUMBER_MIDFIELDERS
                for w in self.gameweeks
            ),
            name='mid_limit',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, 'FW'] for p in self.players) == MAX_NUMBER_FORWARDS
                for w in self.gameweeks
            ),
            name='for_limit',
        )

        # The number of players from a team must exceed three
        self.model.add_constraints(
            (
                so.expr_sum(self.team[p, w] * self.data.loc[p, team_name] for p in self.players)
                <= MAX_NUMBER_PLAYERS_PER_TEAM
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='team_limit',
        )

        # The number of Freehit players must fit the requirements
        # 2 Gk, 5 Def, 5 Mid, 3 For
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'GK'] for p in self.players)
                == MAX_NUMBER_GOALKEEPERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='gk_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'DF'] for p in self.players)
                == MAX_NUMBER_DEFENDERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='def_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'MD'] for p in self.players)
                == MAX_NUMBER_MIDFIELDERS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='mid_limit_fh',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'FW'] for p in self.players)
                == MAX_NUMBER_FORWARDS * self.freehit[w]
                for w in self.gameweeks
            ),
            name='for_limit_fh',
        )

        # The number of Freehit players from a team must not exceed three
        self.model.add_constraints(
            (
                so.expr_sum(self.team_fh[p, w] * self.data.loc[p, team_name] for p in self.players)
                <= 3 * self.freehit[w]
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='team_limit_fh',
        )

        # Starters
        # The formation must be valid i.e. Minimum one goalkeeper,
        # 3 defenders, 2 midfielders and 1 striker on the lineup
        self.model.add_constraints(
            (so.expr_sum(self.starter[p, w] for p in self.players) == 11 + 4 * self.bboost[w] for w in self.gameweeks),
            name='11_starters',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'GK'] for p in self.players)
                == MIN_FORMATION['GK'] + self.bboost[w]
                for w in self.gameweeks
            ),
            name='gk_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'DF'] for p in self.players) >= MIN_FORMATION['DF']
                for w in self.gameweeks
            ),
            name='def_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'MD'] for p in self.players) >= MIN_FORMATION['MD']
                for w in self.gameweeks
            ),
            name='mid_min',
        )
        self.model.add_constraints(
            (
                so.expr_sum(self.starter[p, w] * self.data.loc[p, 'FW'] for p in self.players) >= MIN_FORMATION['FW']
                for w in self.gameweeks
            ),
            name='for_min',
        )

        # Linearization constraints to limit the Freehit Team
        self.model.add_constraints(
            (self.starter[p, w] <= self.team_fh[p, w] + self.aux[p, w] for p in self.players for w in self.gameweeks),
            name='4.24',
        )
        self.model.add_constraints(
            (self.aux[p, w] <= self.team[p, w] for p in self.players for w in self.gameweeks), name='4.25'
        )
        self.model.add_constraints(
            (self.aux[p, w] <= 1 - self.freehit[w] for p in self.players for w in self.gameweeks), name='4.26'
        )

        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team_fh[p, w] * self.data.loc[p, team_name] * (self.data.loc[p, 'DF'] + self.data.loc[p, 'GK'])
                    for p in self.players
                )
                <= def_stack_limit * self.freehit[w]
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='def_stack_limit_fh',
        )

        self.model.add_constraints(
            (
                so.expr_sum(
                    self.team[p, w] * self.data.loc[p, team_name] * (self.data.loc[p, 'DF'] + self.data.loc[p, 'GK'])
                    for p in self.players
                )
                <= def_stack_limit
                for team_name in self.team_names
                for w in self.gameweeks
            ),
            name='def_stack_limit',
        )

        # Captain
        # One captain (or one triple cap) must be picked
        self.model.add_constraints(
            (so.expr_sum(self.captain[p, w] + self.triple[p, w] for p in self.players) == 1 for w in self.gameweeks),
            name='one_captain',
        )
        # One vice captain must be picked
        self.model.add_constraints(
            (so.expr_sum(self.vicecaptain[p, w] for p in self.players) == 1 for w in self.gameweeks),
            name='one_vicecaptain',
        )
        # The captain, vice captain and triple captain must be starters and
        # must not be the same player
        self.model.add_constraints(
            (
                self.captain[p, w] + self.triple[p, w] + self.vicecaptain[p, w] <= self.starter[p, w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='cap_in_starters',
        )

        # Substitutions
        # The first substitute is a single goalkeeper
        self.model.add_constraints(
            (
                so.expr_sum(self.bench[p, w, 0] for p in self.players if self.data.loc[p, 'GK'] == 1) <= 1
                for w in self.gameweeks
            ),
            name='one_bench_gk',
        )
        # There must be a single substitute per bench spot
        self.model.add_constraints(
            (so.expr_sum(self.bench[p, w, o] for p in self.players) <= 1 for w in self.gameweeks for o in [1, 2, 3]),
            name='one_per_bench_spot',
        )

        # The players not started in the team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order)
                <= self.team[p, w] + 10000 * self.freehit[w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='bench_team',
        )
        # The players not started in the freehit team are benched
        self.model.add_constraints(
            (
                self.starter[p, w] + so.expr_sum(self.bench[p, w, o] for o in order)
                <= self.team_fh[p, w] + 10000 * (1 - self.freehit[w])
                for p in self.players
                for w in self.gameweeks
            ),
            name='bench_team_fh',
        )

        # Budget
        sold_amount = {
            w: so.expr_sum(self.sell[p, w] * self.data.loc[p, 'selling_price'] for p in self.players)
            for w in self.gameweeks
        }
        bought_amount = {
            w: so.expr_sum(self.buy[p, w] * self.data.loc[p, 'purchase_price'] for p in self.players)
            for w in self.gameweeks
        }
        # The cost of the squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w] == self.in_the_bank[w - 1] + sold_amount[w] - bought_amount[w]
                for w in self.gameweeks
            ),
            name='budget',
        )
        # The team must be the same as the previous GW plus/minus transfers
        self.model.add_constraints(
            (
                self.team[p, w - 1] + self.buy[p, w] - self.sell[p, w] == self.team[p, w]
                for p in self.players
                for w in self.gameweeks
            ),
            name='team_similarity',
        )
        # The player must not be sold and bought simultaneously
        self.model.add_constraints(
            (self.sell[p, w] + self.buy[p, w] <= 1 for p in self.players for w in self.gameweeks),
            name='single_buy_or_sell',
        )

        # The cost of the freehit squad must exceed the budget
        self.model.add_constraints(
            (
                self.in_the_bank[w - 1]
                + so.expr_sum(self.team[p, w - 1] * self.data.loc[p, 'selling_price'] for p in self.players)
                >= so.expr_sum(self.team_fh[p, w] * self.data.loc[p, 'purchase_price'] for p in self.players)
                for w in self.gameweeks
            ),
            name='budget_fh',
        )
        # On Freehit GW the number of transfers must be zero
        self.model.add_constraints(
            (so.expr_sum(self.sell[p, w] for p in self.players) <= 15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_sold_fh',
        )
        self.model.add_constraints(
            (so.expr_sum(self.buy[p, w] for p in self.players) <= 15 * (1 - self.freehit[w]) for w in self.gameweeks),
            name='zero_bought_fh',
        )

        # The number of players bought and sold is equal
        self.model.add_constraints(
            (
                so.expr_sum(self.buy[p, w] for p in self.players) == so.expr_sum(self.sell[p, w] for p in self.players)
                for w in self.gameweeks
            ),
            name='equal_transfers',
        )

        goalkeeper_cost = {
            w: so.expr_sum(
                self.team[p, w] * self.data.loc[p, 'purchase_price'] * self.data.loc[p, 'GK'] for p in self.players
            )
            for w in self.gameweeks
        }
        self.model.add_constraints(
            (goalkeeper_cost[w] <= goalkeeper_max_budget for w in self.gameweeks), name='goalkeeper_budget'
        )

        goalkeeper_cost_fh = {
            w: so.expr_sum(
                self.team_fh[p, w] * self.data.loc[p, 'purchase_price'] * self.data.loc[p, 'GK'] for p in self.players
            )
            for w in self.gameweeks
        }
        self.model.add_constraints(
            (goalkeeper_cost_fh[w] <= goalkeeper_max_budget for w in self.gameweeks), name='goalkeeper_budget'
        )

        # Transfers
        # The rolling transfer must be equal to the number of free
        # transfers not used (+ 1)
        self.model.add_constraints(
            (
                15 * self.wildcard[w]
                + self.free_transfers[w]
                - so.expr_sum(self.buy[p, w] for p in self.players)
                + 1
                + self.hits[w]
                >= self.free_transfers[w + 1]
                for w in self.gameweeks
            ),
            name='rolling_ft_rel',
        )
        # The hits value is zero only when the number of FT is 2
        self.model.add_constraints(
            (10000 * (2 - self.free_transfers[w + 1]) >= self.hits[w] for w in self.gameweeks), name='4.42'
        )
        # The minimum number of FT is 1
        self.model.add_constraints((self.free_transfers[w + 1] >= 1 for w in self.gameweeks), name='min_ft')
        # The maximum number of FT is 2 on regular GWs
        self.model.add_constraints(
            (self.free_transfers[w + 1] <= 2 - self.wildcard[w] - self.freehit[w] for w in self.gameweeks),
            name='max_ft',
        )

        # Solve
        self.model.export_mps(filename=f'tmp/{model_name}.mps')
        command = f'cbc tmp/{model_name}.mps solve solu ' + f'tmp/{model_name}_solution.txt'

        process = Popen(command, shell=True, stdout=DEVNULL)  # noqa: S602
        process.wait()

        # Reset variables for next passes
        for v in self.model.get_variables():
            v.set_value(0)

        with Path(f'tmp/{model_name}_solution.txt').open('r') as f:
            for line in f:
                if 'objective value' in line:
                    continue
                words = line.split()
                var = self.model.get_variable(words[1])
                var.set_value(float(words[2]))

        return pretty_print(
            self.data,
            self.start,
            self.period,
            self.team,
            self.team_fh,
            self.starter,
            self.bench,
            self.captain,
            self.vicecaptain,
            self.buy,
            self.sell,
            self.free_transfers,
            self.hits,
            self.in_the_bank,
            self.model.get_objective_value(),
            self.freehit,
            self.wildcard,
            self.bboost,
            self.triple,
            nb_suboptimal=model_name,
        )

    def solve(self, model_name: str, log: bool = False, i: int = 0, time_lim: int = 0) -> dict:
        self.model.export_mps(filename=f'tmp/{model_name}.mps')
        if time_lim == 0:
            command = f'cbc tmp/{model_name}.mps cost column solve solu ' + f'tmp/{model_name}_solution.txt'
            if log:
                os.system(command)  # noqa: S605
            else:
                process = Popen(command, shell=True, stdout=DEVNULL)  # noqa: S602
                process.wait()

        else:
            command = (
                f'cbc tmp/{model_name}.mps cost column ratio 1 solve solu ' + f'tmp/{model_name}_solution_feasible.txt'
            )
            if log:
                os.system(command)  # noqa: S605
            else:
                process = Popen(command, shell=True, stdout=DEVNULL)  # noqa: S602
                process.wait()

            command = (
                f'cbc tmp/{model_name}.mps mips tmp/{model_name}_solution_feasible.txt '
                f'cost column sec {time_lim} solve solu tmp/{model_name}_solution.txt'
            )
            if log:
                os.system(command)  # noqa: S605
            else:
                process = Popen(command, shell=True, stdout=DEVNULL)  # noqa: S602
                process.wait()

        # Reset variables for next passes
        for v in self.model.get_variables():
            v.set_value(0)

        with Path(f'tmp/{model_name}_solution.txt').open('r') as f:
            for line in f:
                if 'objective value' in line:
                    continue
                words = line.split()
                var = self.model.get_variable(words[1])
                var.set_value(float(words[2]))

        return pretty_print(
            self.data,
            self.start,
            self.period,
            self.team,
            self.team_fh,
            self.starter,
            self.bench,
            self.captain,
            self.vicecaptain,
            self.buy,
            self.sell,
            self.free_transfers,
            self.hits,
            self.in_the_bank,
            self.model.get_objective_value(),
            self.freehit,
            self.wildcard,
            self.bboost,
            self.triple,
            nb_suboptimal=i,
        )

    def suboptimals(self, model_name: str, iterations: int = 3, cutoff_search: str = 'first_transfer') -> dict:
        sa = {}

        for i in range(iterations):
            print(f'\n----- Solution {i + 1} -----')
            self.solve(model_name + f'_{i}', i=i)

            if i != iterations - 1:
                # Select the players that have been transfered in/out
                if cutoff_search == 'first_buy':
                    actions = so.expr_sum(
                        self.buy[p, self.start] for p in self.players if self.buy[p, self.start].get_value() > 0
                    )
                    gw_range = [self.start]
                elif cutoff_search == 'horizon_buy':
                    actions = so.expr_sum(
                        so.expr_sum(self.buy[p, w] for p in self.players if self.buy[p, w].get_value() > 0)
                        for w in self.gameweeks
                    )
                    gw_range = self.gameweeks
                elif cutoff_search == 'first_transfer':
                    actions = so.expr_sum(
                        self.buy[p, self.start] for p in self.players if self.buy[p, self.start].get_value() > 0
                    ) + so.expr_sum(
                        self.sell[p, self.start] for p in self.players if self.sell[p, self.start].get_value() > 0
                    )
                    gw_range = [self.start]
                elif cutoff_search == 'horizon_transfer':
                    actions = so.expr_sum(
                        so.expr_sum(self.buy[p, w] for p in self.players if self.buy[p, w].get_value() > 0)
                        for w in self.gameweeks
                    ) + so.expr_sum(
                        so.expr_sum(self.sell[p, w] for p in self.players if self.sell[p, w].get_value() > 0)
                        for w in self.gameweeks
                    )
                    gw_range = self.gameweeks

                if actions.get_value() != 0:
                    # This step forces one transfer to be unfeasible
                    # Note: the constraint is only applied to the activated
                    # transfers so the ones not activated are thus allowed.
                    self.model.add_constraint(actions <= actions.get_value() - 1, name=f'cutoff_{i}')
                else:
                    # Force one transfer in case of sub-optimal solution
                    # choosing to roll transfer
                    self.model.add_constraint(
                        so.expr_sum(so.expr_sum(self.sell[p, w] for p in self.players) for w in gw_range) >= 1,
                        name=f'cutoff_{i}',
                    )

            if self.freehit[self.start].get_value():
                sa[i] = [
                    [p for p in self.players if self.team_fh[p, self.start].get_value() > 0],
                    [],
                    self.model.get_objective_value(),
                ]
            else:
                sa[i] = [
                    [p for p in self.players if self.buy[p, self.start].get_value() > 0],
                    [p for p in self.players if self.sell[p, self.start].get_value() > 0],
                    self.model.get_objective_value(),
                ]

        return sa

    def sensitivity_analysis(self, repeats: int = 3, iterations: int = 3, parameters: dict = {'model_name': 'sa'}):  # noqa: ANN201, B006
        podium = pd.DataFrame(columns=list(np.arange(1, 4)))
        hashes = {}
        raw_data = self.data.copy()

        # Reproduce the optimization from scratch
        for r in range(repeats):
            self.build_model(parameters)

            print(f'\n----- Trial {r + 1} -----')

            # Apply random noise to the original prediction
            # (i.e not stacking noise)
            self.data = raw_data.copy()
            self.random_noise(None)
            # Find optimal solutions
            sa = self.suboptimals(f'sensitivity_analysis_{r}', iterations=iterations)

            yield 1

            # Store data
            for i, v in enumerate(sa.values()):
                transfer = hash((tuple(v[0]), tuple(v[1])))
                hashes[transfer] = v

                if transfer in podium.index:
                    podium.loc[transfer, i + 1] += 1

                else:
                    for pos in range(1, iterations + 1):
                        podium.loc[transfer, pos] = 0

                    podium.loc[transfer, i + 1] = 1

                num_cols = sum([podium.loc[transfer, r] for r in np.arange(1, iterations + 1)])
                podium.loc[transfer, f'EV_{int(num_cols)}'] = v[2]

        podium.fillna(0).to_csv('tmp/podium.csv')
        with Path('tmp/hashes.json').open('w') as outfile:
            json.dump(hashes, outfile)


if __name__ == '__main__':
    Path('tmp/').mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger: logging.Logger = logging.getLogger(__name__)

    with Path('info.json').open() as f:
        info = json.load(f)
        team_id = info['team-id']

    to = TeamOptimization(team_id=team_id, horizon=3, noise=False, premium=True)

    to.build_model(
        {
            'model_name': 'vanilla',
            'freehit_gw': -1,
            'wildcard_gw': -1,
            'bboost_gw': -1,
            'threexc_gw': -1,
            'objective_type': 'decay',
            'decay_gameweek': 0.85,
            'vicecap_decay': 0.1,
            'decay_bench': [0.03, 0.21, 0.06, 0.002],
            'ft_val': 1.1,
            'itb_val': 0.008,
            'hit_val': 6,
            'goalkeeper_max_budget': 10,
            'def_stack_limit': 2,
        }
    )

    to.differential_model(nb_differentials=3, threshold=10, target='Top_100K')

    to.automated_chips_model(
        {
            'objective_type': 'decay',
            'decay_gameweek': 0.9,
            'vicecap_decay': 0.1,
            'decay_bench': [0.03, 0.21, 0.06, 0.002],
            'ft_val': 1.5,
            'itb_val': 0.008,
            'hit_val': 6,
        }
    )

    to.advanced_wildcard(
        {
            'freehit_gw': -1,
            'bboost_gw': -1,
            'threexc_gw': -1,
            'objective_type': 'decay',
            'decay_gameweek': 0.9,
            'vicecap_decay': 0.1,
            'decay_bench': [0.03, 0.21, 0.06, 0.002],
            'ft_val': 1.5,
            'itb_val': 0.008,
            'hit_val': 6,
        }
    )

    to.biased_model(
        love={'buy': {}, 'start': {}, 'team': {}, 'cap': {}},
        hate={'sell': {}, 'team': {}, 'bench': {}},
        hit_limit={'max': {}, 'eq': {}, 'min': {}},
        two_ft_gw=[],
    )

    to.solve(model_name='vanilla', log=True, time_lim=0)

    to.suboptimals(model_name='vanilla', iterations=3, cutoff_search='first_transfer')

    for _ in to.sensitivity_analysis(repeats=10, iterations=5):
        pass
