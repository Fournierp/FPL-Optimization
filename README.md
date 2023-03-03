# Fantasy-Premier-League 

## Introduction

Fantasy Premier League (FPL) is a popular online game where players can create their own virtual teams of real-life Premier League footballers and compete with others based on the players' real-life performances in matches. The game allows participants to select and manage a squad of 15 players. The aim of the game is to score as many points as possible by earning points for goals, assists, clean sheets and other performance-based criteria, and making strategic decisions around transfers, formations and captaincy.

The idea ixed integer linear programming (MILP) can be useful in Fantasy Premier League for optimizing team selection and transfer decisions. MILP can be used to optimize transfer decisions by taking into account the potential benefits and costs associated with each transfer (like player performance history, injury status, fixture difficulty, and cost).

## Installation

Firstly, ensure that you have installed python and pip. Then build you virtualenv to avoid version conflicts with other projects.

```{bash}
sudo apt install python3-virtualenv
virtualenv -p python3.10 venv3
```

Then install the required libraries listed in the requirements.txt

```{bash}
pip install -r requirements.txt
```

## Usage Example

**Note:** Prior to running this, one should have downloaded the gameweek prediction data and saved it in folder: *data/fpl_review/SEASON/gameweek/GAMEWEEK*.

```{bash}
python3 optimization/team_optimization.py
```

An alternative is to use the Google Colab version [here](https://colab.research.google.com/drive/1izz48YWGfsK0VBnDLlki6gDVvbvHr25I?usp=sharing).

## Acknowledgements

[FPL](https://fantasy.premierleague.com/) - Official data on player ownership, chips used etc.

[FPL Review](https://fplreview.com/) - FPL Predictions

[Forecast-Based Optimization Model for Fantasy Premier League](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2577003) - Linear Optimization model

[sertalpbilal/FPL-Optimization-Tools](https://github.com/sertalpbilal/FPL-Optimization-Tools) - Tutorials and recipes to use optimization for winning Fantasy Premier League

[livefpl.net](https://www.livefpl.net/elite) - Best 1000 Managers of All Time

[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) - Historical FPL data

## Contribute

To build on this tool, please fork it and make pull requests. Or simply send me some suggestions !

## Authors

* **Paul Fournier** - *Initial work* - [Fournierp](https://github.com/Fournierp)
