"""
Player Model for Lineup Generator
Represents individual player data and statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class PlayerStats:
    """Container for player statistical data"""
    name: str
    team: str
    plate_appearances: int
    singles: int
    doubles: int
    triples: int
    home_runs: int
    walks: int
    hit_by_pitch: int
    strikeouts: int
    ground_into_double_play: int
    woba: float
    obp: float
    slg: float
    xwoba: float
    xba: float
    xslg: float
    iso: float


class PlayerModel:
    """Represents a baseball player with their statistics and capabilities"""
    
    def __init__(self, player_data: pd.Series, config_manager=None):
        """
        Initialize player model from pandas Series
        
        Args:
            player_data: Pandas Series containing player statistics
            config_manager: ConfigManager instance for default values
        """
        self.name = player_data.get('Name', 'Unknown')
        self.team = player_data.get('Team', player_data.get('Tm', 'Unknown'))
        self.pa = max(player_data.get('PA', 1), 1)
        
        # Basic hitting statistics
        self.singles = player_data.get('1B', 0)
        self.doubles = player_data.get('2B', 0)
        self.triples = player_data.get('3B', 0)
        self.home_runs = player_data.get('HR', 0)
        self.walks = player_data.get('BB', 0)
        self.hit_by_pitch = player_data.get('HBP', 0)
        self.strikeouts = player_data.get('SO', 0)
        self.ground_into_double_play = player_data.get('GDP', 0)
        
        # Advanced metrics
        self.woba = player_data.get('wOBA', 0.0)
        self.obp = player_data.get('OBP', 0.0)
        self.slg = player_data.get('SLG', 0.0)
        self.xwoba = player_data.get('xwOBA', 0.0)
        self.xba = player_data.get('xBA', 0.0)
        self.xslg = player_data.get('xSLG', 0.0)
        self.iso = player_data.get('ISO', 0.0)
        
        # Additional stats for enhanced simulation
        self.ground_balls = player_data.get('GB', self.pa * 0.4)
        self.fly_balls = player_data.get('FB', self.pa * 0.4)
        self.stolen_bases = player_data.get('SB', 0)
        
        # Store raw data for reference
        self.raw_data = player_data
        
        # Calculate derived statistics
        self._calculate_derived_stats()
    
    def _calculate_derived_stats(self):
        """Calculate derived statistics from basic stats"""
        # Rate statistics
        self.single_rate = self.singles / self.pa
        self.double_rate = self.doubles / self.pa
        self.triple_rate = self.triples / self.pa
        self.hr_rate = self.home_runs / self.pa
        self.walk_rate = self.walks / self.pa
        self.hbp_rate = self.hit_by_pitch / self.pa
        self.strikeout_rate = self.strikeouts / self.pa
        self.gidp_rate = self.ground_into_double_play / self.pa
        
        # Contact and power metrics
        self.contact_rate = 1 - self.strikeout_rate
        self.power_score = self.slg - self.obp
        self.speed_score = self.stolen_bases / self.pa
        
        # Situational hitting (estimated)
        self.clutch_factor = min(1.2, max(0.8, self.woba / 0.32))  # Normalize around league average
    
    def get_outcome_probabilities(self) -> Dict[str, float]:
        """
        Calculate probability of different batting outcomes
        
        Returns:
            Dictionary mapping outcomes to probabilities
        """
        # Calculate basic probabilities
        single_prob = self.single_rate
        double_prob = self.double_rate
        triple_prob = self.triple_rate
        hr_prob = self.hr_rate
        walk_prob = self.walk_rate
        hbp_prob = self.hbp_rate
        strikeout_prob = self.strikeout_rate
        
        # Estimate sac fly probability (not directly available in standard stats)
        sac_fly_prob = min(0.02, max(0.005, self.contact_rate * 0.02))
        
        # Estimate other outcomes
        groundout_prob = (self.ground_balls / self.pa) * 0.6  # ~60% of GB become outs
        flyout_prob = (self.fly_balls / self.pa) * 0.7  # ~70% of FB become outs
        
        # Calculate remaining out probability
        total_positive = (single_prob + double_prob + triple_prob + hr_prob + 
                         walk_prob + hbp_prob + strikeout_prob + sac_fly_prob + 
                         groundout_prob + flyout_prob)
        remaining_out_prob = max(0, 1 - total_positive)
        
        probs = {
            'single': single_prob,
            'double': double_prob,
            'triple': triple_prob,
            'hr': hr_prob,
            'walk': walk_prob,
            'hbp': hbp_prob,
            'strikeout': strikeout_prob,
            'sac_fly': sac_fly_prob,
            'groundout': groundout_prob,
            'flyout': flyout_prob,
            'out': remaining_out_prob
        }
        
        # Use expected stats for small sample sizes
        if self.pa < 50:
            probs['single'] = 0.7 * probs['single'] + 0.3 * self.xba
            probs['double'] = 0.7 * probs['double'] + 0.3 * max(0, (self.xslg - self.xba) / 3)
            probs['hr'] = 0.7 * probs['hr'] + 0.3 * max(0, (self.xslg - self.xba) / 6)
        
        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            # Fallback to league average
            probs = {
                'single': 0.15, 'double': 0.05, 'triple': 0.01, 'hr': 0.03,
                'walk': 0.09, 'hbp': 0.01, 'strikeout': 0.22, 'sac_fly': 0.015,
                'groundout': 0.25, 'flyout': 0.15, 'out': 0.105
            }
        
        return probs
    
    def get_position_suitability(self, position: int) -> Dict[str, float]:
        """
        Calculate how suitable this player is for a specific lineup position
        
        Args:
            position: Lineup position (1-9)
            
        Returns:
            Dictionary of suitability metrics
        """
        suitability = {}
        
        if position == 1:  # Leadoff
            suitability['obp_weight'] = 0.4
            suitability['speed_weight'] = 0.3
            suitability['contact_weight'] = 0.3
            suitability['score'] = (self.obp * 0.4 + self.speed_score * 0.3 + self.contact_rate * 0.3)
            
        elif position == 2:  # Two-hole
            suitability['obp_weight'] = 0.3
            suitability['contact_weight'] = 0.4
            suitability['power_weight'] = 0.3
            suitability['score'] = (self.obp * 0.3 + self.contact_rate * 0.4 + self.power_score * 0.3)
            
        elif position in [3, 4]:  # Cleanup hitters
            suitability['power_weight'] = 0.5
            suitability['woba_weight'] = 0.3
            suitability['clutch_weight'] = 0.2
            suitability['score'] = (self.power_score * 0.5 + self.woba * 0.3 + self.clutch_factor * 0.2)
            
        elif position in [5, 6]:  # Middle of order
            suitability['woba_weight'] = 0.4
            suitability['power_weight'] = 0.3
            suitability['contact_weight'] = 0.3
            suitability['score'] = (self.woba * 0.4 + self.power_score * 0.3 + self.contact_rate * 0.3)
            
        else:  # Bottom of order
            suitability['contact_weight'] = 0.4
            suitability['obp_weight'] = 0.3
            suitability['defense_weight'] = 0.3  # Placeholder for defensive metrics
            suitability['score'] = (self.contact_rate * 0.4 + self.obp * 0.3 + 0.5 * 0.3)  # Assume average defense
        
        return suitability
    
    def get_feature_vector(self) -> Dict[str, float]:
        """
        Get feature vector for machine learning models
        
        Returns:
            Dictionary of features for ML models
        """
        return {
            'wOBA': self.woba,
            'OBP': self.obp,
            'SLG': self.slg,
            'xwOBA': self.xwoba,
            'xBA': self.xba,
            'xSLG': self.xslg,
            'ISO': self.iso,
            'BB_rate': self.walk_rate,
            'K_rate': self.strikeout_rate,
            'HR_rate': self.hr_rate,
            'contact_rate': self.contact_rate,
            'power_score': self.power_score,
            'speed_score': self.speed_score,
            'clutch_factor': self.clutch_factor,
            'PA': self.pa
        }
    
    def simulate_at_bat(self) -> str:
        """
        Simulate a single at-bat for this player
        
        Returns:
            String representing the outcome
        """
        probs = self.get_outcome_probabilities()
        outcomes = list(probs.keys())
        probabilities = list(probs.values())
        return np.random.choice(outcomes, p=probabilities)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for this player"""
        return {
            'name': self.name,
            'team': self.team,
            'PA': self.pa,
            'wOBA': self.woba,
            'OBP': self.obp,
            'SLG': self.slg,
            'xwOBA': self.xwoba,
            'contact_rate': self.contact_rate,
            'power_score': self.power_score,
            'speed_score': self.speed_score
        }
    
    def __str__(self) -> str:
        """String representation of the player"""
        return f"{self.name} ({self.team}) - wOBA: {self.woba:.3f}, OBP: {self.obp:.3f}, SLG: {self.slg:.3f}"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"PlayerModel(name='{self.name}', team='{self.team}', wOBA={self.woba:.3f}, PA={self.pa})"
