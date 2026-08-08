"""
This file contains the configuration manager for the lineup generator.
This involves anything to do with the simulations, constants, or default parameters for 
machine learning models. 
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for game simulation parameters"""
    default_simulations: int = 1000
    default_innings: int = 9
    min_plate_appearances: int = 150
    max_lineup_size: int = 9
    min_lineup_size: int = 9


@dataclass
class MLConfig:
    """Configuration for machine learning models"""
    n_estimators: int = 100
    max_depth: int = 5
    random_state: int = 42
    test_size: float = 0.2
    n_lineups_for_training: int = 10000
    n_simulations_per_lineup: int = 100


@dataclass
class BotConfig:
    """Configuration for Discord bot"""
    command_prefix: str = '!'
    max_message_length: int = 2000
    timeout_seconds: int = 30


class ConfigManager:
    """Centralized configuration management for the lineup generator"""
    
    def __init__(self):
        self.simulation = SimulationConfig()
        self.ml = MLConfig()
        self.bot = BotConfig()
        
        # MLB Team abbreviations
        self.mlb_teams = [
            'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CWS', 'CIN', 'CLE',
            'COL', 'DET', 'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL',
            'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SD', 'SF',
            'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSN'
        ]
        
        # Required columns for player data
        self.required_player_columns = [
            'Name', 'PA', '1B', '2B', '3B', 'HR', 'BB', 'HBP', 'SO', 'GDP',
            'wOBA', 'OBP', 'SLG', 'xwOBA', 'xBA', 'xSLG', 'ISO'
        ]
        
        # Run expectancy matrix from the 2025 season 
        # Source: Fangraphs 
        self.base_matrix = [
            [0.48, 0.25, 0.10],  # Bases empty
            [0.87, 0.48, 0.21],  # Runner on 1st
            [1.12, 0.67, 0.31],  # Runner on 2nd
            [1.38, 0.86, 0.32],  # Runners on 1st & 2nd
            [1.55, 0.96, 0.42],  # Runner on 3rd
            [1.78, 1.31, 0.48],  # Runners on 1st & 3rd
            [2.04, 1.41, 0.67],  # Runners on 2nd & 3rd
            [2.69, 1.61, 0.96]   # Bases loaded
        ]
        
        # Positions of each runner for each base
        self.base_states = [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)
        ]
        
        # Fallback if pulling data fails 
        self.default_player_probs = {
            'single': 0.15, 'double': 0.05, 'triple': 0.01, 'hr': 0.03,
            'walk': 0.09, 'hbp': 0.01, 'strikeout': 0.22, 'sac_fly': 0.015,
            'groundout': 0.25, 'flyout': 0.15, 'out': 0.105
        }
    
    def get_team_abbreviation(self, team_name: str) -> str:
        """Converts the team to abbreviation"""
        team_mapping = {
            'diamondbacks': 'ARI', 'braves': 'ATL', 'orioles': 'BAL', 'red sox': 'BOS',
            'cubs': 'CHC', 'white sox': 'CWS', 'reds': 'CIN', 'guardians': 'CLE',
            'rockies': 'COL', 'tigers': 'DET', 'astros': 'HOU', 'royals': 'KC',
            'angels': 'LAA', 'dodgers': 'LAD', 'marlins': 'MIA', 'brewers': 'MIL',
            'twins': 'MIN', 'mets': 'NYM', 'yankees': 'NYY', 'athletics': 'OAK',
            'phillies': 'PHI', 'pirates': 'PIT', 'padres': 'SDP', 'giants': 'SFG',
            'mariners': 'SEA', 'cardinals': 'STL', 'rays': 'TBR', 'rangers': 'TEX',
            'blue jays': 'TOR', 'nationals': 'WSN'
        }
        return team_mapping.get(team_name.lower(), team_name.upper())
    
    def validate_team(self, team: str) -> bool:
        """Validate if team abbreviation is valid"""
        return team.upper() in self.mlb_teams
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """Get simulation parameters as dictionary"""
        return {
            'default_simulations': self.simulation.default_simulations,
            'default_innings': self.simulation.default_innings,
            'min_plate_appearances': self.simulation.min_plate_appearances,
            'max_lineup_size': self.simulation.max_lineup_size,
            'min_lineup_size': self.simulation.min_lineup_size
        }
    
    def get_ml_params(self) -> Dict[str, Any]:
        """Get ML parameters as dictionary"""
        return {
            'n_estimators': self.ml.n_estimators,
            'max_depth': self.ml.max_depth,
            'random_state': self.ml.random_state,
            'test_size': self.ml.test_size,
            'n_lineups_for_training': self.ml.n_lineups_for_training,
            'n_simulations_per_lineup': self.ml.n_simulations_per_lineup
        }
    
    def get_bot_params(self) -> Dict[str, Any]:
        """Get bot parameters as dictionary"""
        return {
            'command_prefix': self.bot.command_prefix,
            'max_message_length': self.bot.max_message_length,
            'timeout_seconds': self.bot.timeout_seconds
        }
    
    def update_simulation_config(self, **kwargs):
        """Update simulation configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.simulation, key):
                setattr(self.simulation, key, value)
    
    def update_ml_config(self, **kwargs):
        """Update ML configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.ml, key):
                setattr(self.ml, key, value)
    
    def update_bot_config(self, **kwargs):
        """Update bot configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.bot, key):
                setattr(self.bot, key, value)
