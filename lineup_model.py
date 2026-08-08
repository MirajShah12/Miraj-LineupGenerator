"""
Lineup Model for Lineup Generator
Represents a baseball lineup with validation and optimization capabilities
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from player_model import PlayerModel


@dataclass
class LineupResult:
    """Container for lineup optimization results"""
    lineup: List[str]
    expected_runs: float
    confidence_interval: Tuple[float, float]
    simulation_count: int
    optimization_method: str


class LineupModel:
    """Represents a baseball lineup with validation and analysis capabilities"""
    
    def __init__(self, players: List[PlayerModel], config_manager=None):
        """
        Initialize lineup model
        
        Args:
            players: List of PlayerModel objects
            config_manager: ConfigManager instance for validation
        """
        self.players = players
        self.config_manager = config_manager
        self.lineup_order = []
        self.expected_runs = 0.0
        self.simulation_results = []
        
        # Validate lineup
        self._validate_lineup()
    
    def _validate_lineup(self):
        """Validate that the lineup meets requirements"""
        if len(self.players) < 9:
            raise ValueError(f"Lineup must have at least 9 players, got {len(self.players)}")
        
        if len(self.players) > 9:
            # Take the first 9 players if more than 9 provided
            self.players = self.players[:9]
        
        # Check for duplicate players
        player_names = [player.name for player in self.players]
        if len(set(player_names)) != len(player_names):
            raise ValueError("Duplicate players found in lineup")
        
        # Initialize lineup order as the order provided
        self.lineup_order = [player.name for player in self.players]
    
    def get_player_by_name(self, name: str) -> Optional[PlayerModel]:
        """Get player by name"""
        for player in self.players:
            if player.name == name:
                return player
        return None
    
    def get_player_by_position(self, position: int) -> Optional[PlayerModel]:
        """Get player by lineup position (1-9)"""
        if 1 <= position <= 9:
            player_name = self.lineup_order[position - 1]
            return self.get_player_by_name(player_name)
        return None
    
    def set_lineup_order(self, new_order: List[str]):
        """
        Set the batting order
        
        Args:
            new_order: List of player names in desired batting order
        """
        if len(new_order) != 9:
            raise ValueError("Lineup must have exactly 9 players")
        
        # Validate all players exist
        player_names = [player.name for player in self.players]
        for name in new_order:
            if name not in player_names:
                raise ValueError(f"Player '{name}' not found in lineup")
        
        self.lineup_order = new_order
    
    def optimize_lineup_order(self, method: str = 'traditional') -> LineupResult:
        """
        Optimize the batting order using different strategies
        
        Args:
            method: Optimization method ('traditional', 'woba', 'ml')
            
        Returns:
            LineupResult with optimized lineup
        """
        if method == 'traditional':
            return self._optimize_traditional()
        elif method == 'woba':
            return self._optimize_woba()
        elif method == 'ml':
            return self._optimize_ml()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_traditional(self) -> LineupResult:
        """Traditional lineup optimization strategy"""
        # Sort players by traditional lineup roles
        df_players = [player for player in self.players]
        
        # Find leadoff hitter (highest OBP)
        leadoff = max(df_players, key=lambda p: p.obp)
        
        # Find two-hole hitter (highest xwOBA)
        two_hole = max([p for p in df_players if p != leadoff], key=lambda p: p.xwoba)
        
        # Find cleanup hitters (highest wOBA)
        power_hitters = sorted([p for p in df_players if p not in [leadoff, two_hole]], 
                              key=lambda p: p.woba, reverse=True)[:2]
        
        # Find sluggers (highest wOBA + xSLG)
        remaining = [p for p in df_players if p not in [leadoff, two_hole] + power_hitters]
        sluggers = sorted(remaining, key=lambda p: p.woba + p.xslg, reverse=True)[:2]
        
        # Remaining players
        rest = [p for p in df_players if p not in [leadoff, two_hole] + power_hitters + sluggers]
        
        # Build optimized lineup
        optimized_order = [leadoff.name, two_hole.name] + [p.name for p in power_hitters] + \
                         [p.name for p in sluggers] + [p.name for p in rest]
        
        # Ensure we have exactly 9 players
        optimized_order = optimized_order[:9]
        
        # Update lineup order
        self.set_lineup_order(optimized_order)
        
        return LineupResult(
            lineup=optimized_order,
            expected_runs=0.0,  # Will be calculated by simulation
            confidence_interval=(0.0, 0.0),
            simulation_count=0,
            optimization_method='traditional'
        )
    
    def _optimize_woba(self) -> LineupResult:
        """Optimize lineup based on wOBA"""
        # Sort players by wOBA (descending)
        sorted_players = sorted(self.players, key=lambda p: p.woba, reverse=True)
        
        # Create lineup with best wOBA players in key positions
        optimized_order = []
        
        # Position 1: Best OBP among top wOBA players
        top_woba = sorted_players[:5]
        leadoff = max(top_woba, key=lambda p: p.obp)
        optimized_order.append(leadoff.name)
        
        # Position 2: Best xwOBA among remaining
        remaining = [p for p in sorted_players if p != leadoff]
        two_hole = max(remaining[:5], key=lambda p: p.xwoba)
        optimized_order.append(two_hole.name)
        
        # Positions 3-4: Next best wOBA players
        remaining = [p for p in remaining if p != two_hole]
        cleanup = remaining[:2]
        optimized_order.extend([p.name for p in cleanup])
        
        # Positions 5-9: Remaining players in wOBA order
        remaining = [p for p in remaining if p not in cleanup]
        optimized_order.extend([p.name for p in remaining])
        
        # Ensure we have exactly 9 players
        optimized_order = optimized_order[:9]
        
        self.set_lineup_order(optimized_order)
        
        return LineupResult(
            lineup=optimized_order,
            expected_runs=0.0,
            confidence_interval=(0.0, 0.0),
            simulation_count=0,
            optimization_method='woba'
        )
    
    def _optimize_ml(self) -> LineupResult:
        """Optimize lineup using machine learning approach"""
        # This would integrate with the LineupAnalysis class
        # For now, use a simplified approach based on position suitability
        optimized_order = []
        used_players = set()
        
        # For each position, find the best available player
        for position in range(1, 10):
            best_player = None
            best_score = -1
            
            for player in self.players:
                if player.name not in used_players:
                    suitability = player.get_position_suitability(position)
                    score = suitability['score']
                    
                    if score > best_score:
                        best_score = score
                        best_player = player
            
            if best_player:
                optimized_order.append(best_player.name)
                used_players.add(best_player.name)
        
        self.set_lineup_order(optimized_order)
        
        return LineupResult(
            lineup=optimized_order,
            expected_runs=0.0,
            confidence_interval=(0.0, 0.0),
            simulation_count=0,
            optimization_method='ml'
        )
    
    def get_lineup_features(self) -> Dict[str, Any]:
        """Get feature vector for the entire lineup"""
        features = {}
        
        # Individual position features
        for i, player_name in enumerate(self.lineup_order):
            player = self.get_player_by_name(player_name)
            if player:
                player_features = player.get_feature_vector()
                for key, value in player_features.items():
                    features[f'pos_{i+1}_{key}'] = value
        
        # Lineup-level features
        features['lineup_wOBA_avg'] = np.mean([p.woba for p in self.players])
        features['lineup_OBP_avg'] = np.mean([p.obp for p in self.players])
        features['lineup_SLG_avg'] = np.mean([p.slg for p in self.players])
        features['lineup_power_avg'] = np.mean([p.power_score for p in self.players])
        features['lineup_contact_avg'] = np.mean([p.contact_rate for p in self.players])
        features['lineup_speed_avg'] = np.mean([p.speed_score for p in self.players])
        
        # Balance metrics
        woba_values = [p.woba for p in self.players]
        features['lineup_wOBA_std'] = np.std(woba_values)
        features['lineup_balance_score'] = 1 - (np.std(woba_values) / np.mean(woba_values))
        
        return features
    
    def get_lineup_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the lineup"""
        return {
            'lineup_order': self.lineup_order,
            'total_wOBA': sum(p.woba for p in self.players),
            'avg_wOBA': np.mean([p.woba for p in self.players]),
            'avg_OBP': np.mean([p.obp for p in self.players]),
            'avg_SLG': np.mean([p.slg for p in self.players]),
            'total_power': sum(p.power_score for p in self.players),
            'avg_contact_rate': np.mean([p.contact_rate for p in self.players]),
            'expected_runs': self.expected_runs,
            'simulation_count': len(self.simulation_results)
        }
    
    def add_simulation_result(self, runs: float):
        """Add a simulation result"""
        self.simulation_results.append(runs)
        self.expected_runs = np.mean(self.simulation_results)
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for expected runs"""
        if len(self.simulation_results) < 2:
            return (0.0, 0.0)
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(self.simulation_results, lower_percentile)
        upper_bound = np.percentile(self.simulation_results, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def __str__(self) -> str:
        """String representation of the lineup"""
        lineup_str = "Lineup:\n"
        for i, player_name in enumerate(self.lineup_order):
            player = self.get_player_by_name(player_name)
            if player:
                lineup_str += f"{i+1}. {player.name} - wOBA: {player.woba:.3f}\n"
        return lineup_str
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"LineupModel(players={len(self.players)}, expected_runs={self.expected_runs:.2f})"
