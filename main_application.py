"""
Main Application for Lineup Generator
Orchestrates all components and provides high-level interface
"""

import os
import sys
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

from config_manager import ConfigManager
from data_generator import DataGenerator
from game_generator import GameGenerator
from lineup_analysis import LineupAnalysis
from discord_bot import DiscordBot
from player_model import PlayerModel
from lineup_model import LineupModel


@dataclass
class ApplicationConfig:
    """Configuration for the main application"""
    discord_token: Optional[str] = None
    default_season: int = 2024
    enable_discord_bot: bool = False
    enable_ml_training: bool = True
    cache_data: bool = True
    log_level: str = "INFO"


class LineupGeneratorApp:
    """Main application class that orchestrates all components"""
    
    def __init__(self, app_config: ApplicationConfig = None):
        """
        Initialize the lineup generator application
        
        Args:
            app_config: Application configuration
        """
        self.app_config = app_config or ApplicationConfig()
        
        # Initialize core components
        self.config_manager = ConfigManager()
        self.data_generator = DataGenerator(self.config_manager)
        self.game_generator = GameGenerator(self.config_manager)
        self.lineup_analysis = LineupAnalysis(self.config_manager, self.game_generator)
        
        # Initialize Discord bot if enabled
        self.discord_bot = None
        if self.app_config.enable_discord_bot and self.app_config.discord_token:
            self.discord_bot = DiscordBot(
                self.config_manager,
                self.data_generator,
                self.game_generator,
                self.lineup_analysis
            )
        
        # Application state
        self.is_initialized = False
        self.current_season = self.app_config.default_season
        
    def initialize(self):
        """Initialize the application"""
        print("Initializing Lineup Generator Application...")
        
        try:
            # Validate configuration
            self._validate_config()
            
            # Initialize ML models if enabled
            if self.app_config.enable_ml_training:
                print("Preparing ML models...")
                self._prepare_ml_models()
            
            self.is_initialized = True
            print("Application initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing application: {e}")
            raise
    
    def _validate_config(self):
        """Validate application configuration"""
        if self.app_config.enable_discord_bot and not self.app_config.discord_token:
            raise ValueError("Discord token required when Discord bot is enabled")
        
        if self.app_config.default_season < 2020 or self.app_config.default_season > 2025:
            raise ValueError("Default season must be between 2020 and 2025")
    
    def _prepare_ml_models(self):
        """Prepare ML models with sample data"""
        try:
            # Get sample data for ML training
            print("Getting sample data for ML model training...")
            players = self.data_generator.get_top_players(50, self.current_season)
            
            if len(players) >= 9:
                # Generate training data
                training_data = self.lineup_analysis.generate_training_data(players, 1000, 100)
                
                # Train models
                self.lineup_analysis.train_position_models(training_data)
                
                print("ML models trained successfully!")
            else:
                print("Not enough players for ML training, using test data...")
                # Use test data
                test_players = self.data_generator.create_test_data()
                training_data = self.lineup_analysis.generate_training_data(test_players, 500, 50)
                self.lineup_analysis.train_position_models(training_data)
                
        except Exception as e:
            print(f"Warning: Could not prepare ML models: {e}")
            print("Application will continue without ML models")
    
    def generate_lineup(self, team: str, season: int = None, method: str = 'traditional') -> Dict[str, Any]:
        """
        Generate optimal lineup for a team
        
        Args:
            team: Team abbreviation
            season: Season year
            method: Optimization method
            
        Returns:
            Dictionary with lineup results
        """
        if not self.is_initialized:
            self.initialize()
        
        if season is None:
            season = self.current_season
        
        print(f"Generating {method} lineup for {team} ({season})...")
        
        try:
            # Get team roster
            players = self.data_generator.get_team_roster(team, season)
            
            if len(players) < 9:
                raise ValueError(f"Not enough players for {team}. Found {len(players)} players.")
            
            # Create lineup
            lineup = LineupModel(players, self.config_manager)
            
            # Optimize lineup
            if method in ['traditional', 'woba', 'ml']:
                result = lineup.optimize_lineup_order(method)
                lineup.set_lineup_order(result.lineup)
            else:
                # Use game generator for simulation-based optimization
                result = self.game_generator.optimize_lineup_order(players, method)
                lineup.set_lineup_order(result.lineup)
            
            # Evaluate lineup
            evaluation = self.game_generator.evaluate_lineup(lineup, 1000)
            
            return {
                'team': team,
                'season': season,
                'method': method,
                'lineup': lineup.lineup_order,
                'evaluation': evaluation,
                'summary': lineup.get_lineup_summary()
            }
            
        except Exception as e:
            print(f"Error generating lineup: {e}")
            raise
    
    def compare_lineup_methods(self, team: str, season: int = None) -> Dict[str, Any]:
        """
        Compare different lineup optimization methods
        
        Args:
            team: Team abbreviation
            season: Season year
            
        Returns:
            Dictionary with comparison results
        """
        if not self.is_initialized:
            self.initialize()
        
        if season is None:
            season = self.current_season
        
        print(f"Comparing lineup methods for {team} ({season})...")
        
        try:
            # Get team roster
            players = self.data_generator.get_team_roster(team, season)
            
            if len(players) < 9:
                raise ValueError(f"Not enough players for {team}. Found {len(players)} players.")
            
            # Test different methods
            methods = ['traditional', 'woba', 'ml']
            lineups = []
            
            for method in methods:
                lineup = LineupModel(players, self.config_manager)
                result = lineup.optimize_lineup_order(method)
                lineup.set_lineup_order(result.lineup)
                lineups.append(lineup)
            
            # Compare lineups
            comparison = self.game_generator.compare_lineups(lineups, 1000)
            
            return {
                'team': team,
                'season': season,
                'comparison': comparison,
                'methods': methods
            }
            
        except Exception as e:
            print(f"Error comparing lineups: {e}")
            raise
    
    def analyze_team(self, team: str, season: int = None) -> Dict[str, Any]:
        """
        Perform comprehensive team analysis
        
        Args:
            team: Team abbreviation
            season: Season year
            
        Returns:
            Dictionary with analysis results
        """
        if not self.is_initialized:
            self.initialize()
        
        if season is None:
            season = self.current_season
        
        print(f"Performing comprehensive analysis for {team} ({season})...")
        
        try:
            # Get team roster
            players = self.data_generator.get_team_roster(team, season)
            
            if len(players) < 9:
                raise ValueError(f"Not enough players for {team}. Found {len(players)} players.")
            
            # Generate training data if not available
            if self.lineup_analysis.training_data is None:
                print("Generating training data...")
                self.lineup_analysis.generate_training_data(players, 1000, 100)
            
            # Train models if not already trained
            if not self.lineup_analysis.ml_models:
                print("Training ML models...")
                self.lineup_analysis.train_position_models()
            
            # Analyze position importance
            print("Analyzing position importance...")
            importance = self.lineup_analysis.analyze_position_importance()
            
            # Optimize using ML
            print("Optimizing lineup with ML...")
            optimized_lineup = self.lineup_analysis.optimize_lineup_ml(players, 'genetic')
            
            # Evaluate optimized lineup
            evaluation = self.game_generator.evaluate_lineup(optimized_lineup, 1000)
            
            # Get league averages for comparison
            league_averages = self.data_generator.get_league_averages(season)
            
            return {
                'team': team,
                'season': season,
                'players': [player.get_summary_stats() for player in players],
                'optimized_lineup': optimized_lineup.lineup_order,
                'evaluation': evaluation,
                'position_importance': {pos: df.head(5).to_dict('records') for pos, df in importance.items()},
                'league_averages': league_averages,
                'analysis_summary': self.lineup_analysis.get_analysis_summary()
            }
            
        except Exception as e:
            print(f"Error analyzing team: {e}")
            raise
    
    def get_team_roster(self, team: str, season: int = None) -> List[Dict[str, Any]]:
        """
        Get team roster with player statistics
        
        Args:
            team: Team abbreviation
            season: Season year
            
        Returns:
            List of player statistics
        """
        if not self.is_initialized:
            self.initialize()
        
        if season is None:
            season = self.current_season
        
        try:
            players = self.data_generator.get_team_roster(team, season)
            return [player.get_summary_stats() for player in players]
            
        except Exception as e:
            print(f"Error getting team roster: {e}")
            raise
    
    def run_discord_bot(self):
        """Run the Discord bot"""
        if not self.discord_bot:
            raise ValueError("Discord bot not enabled or token not provided")
        
        if not self.is_initialized:
            self.initialize()
        
        print("Starting Discord bot...")
        try:
            self.discord_bot.run(self.app_config.discord_token)
        except KeyboardInterrupt:
            print("Stopping Discord bot...")
            self.discord_bot.stop()
        except Exception as e:
            print(f"Error running Discord bot: {e}")
            raise
    
    def export_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Export results to CSV file
        
        Args:
            results: Results dictionary
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            team = results.get('team', 'unknown')
            season = results.get('season', self.current_season)
            filename = f"{team}_{season}_lineup_results.csv"
        
        # Convert results to DataFrame
        if 'lineup' in results:
            lineup_data = []
            for i, player_name in enumerate(results['lineup']):
                lineup_data.append({
                    'position': i + 1,
                    'player': player_name,
                    'method': results.get('method', 'unknown')
                })
            
            df = pd.DataFrame(lineup_data)
            df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
        
        return filename
    
    def get_status(self) -> Dict[str, Any]:
        """Get application status"""
        return {
            'is_initialized': self.is_initialized,
            'current_season': self.current_season,
            'discord_bot_enabled': self.discord_bot is not None,
            'discord_bot_running': self.discord_bot.is_running if self.discord_bot else False,
            'ml_models_trained': len(self.lineup_analysis.ml_models),
            'training_data_size': len(self.lineup_analysis.training_data) if self.lineup_analysis.training_data is not None else 0,
            'config': {
                'default_season': self.app_config.default_season,
                'enable_ml_training': self.app_config.enable_ml_training,
                'cache_data': self.app_config.cache_data
            }
        }
    
    def update_config(self, **kwargs):
        """Update application configuration"""
        for key, value in kwargs.items():
            if hasattr(self.app_config, key):
                setattr(self.app_config, key, value)
                print(f"Updated {key} to {value}")
    
    def clear_cache(self):
        """Clear all cached data"""
        self.data_generator.clear_cache()
        self.lineup_analysis.training_data = None
        self.lineup_analysis.ml_models.clear()
        self.lineup_analysis.scalers.clear()
        print("Cache cleared")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Lineup Generator Application')
    parser.add_argument('--team', required=True, help='Team abbreviation')
    parser.add_argument('--season', type=int, default=2024, help='Season year')
    parser.add_argument('--method', default='traditional', help='Optimization method')
    parser.add_argument('--discord-token', help='Discord bot token')
    parser.add_argument('--run-bot', action='store_true', help='Run Discord bot')
    parser.add_argument('--compare', action='store_true', help='Compare methods')
    parser.add_argument('--analyze', action='store_true', help='Perform ML analysis')
    
    args = parser.parse_args()
    
    # Create application config
    app_config = ApplicationConfig(
        discord_token=args.discord_token,
        enable_discord_bot=bool(args.discord_token),
        default_season=args.season
    )
    
    # Initialize application
    app = LineupGeneratorApp(app_config)
    app.initialize()
    
    try:
        if args.run_bot:
            app.run_discord_bot()
        elif args.compare:
            results = app.compare_lineup_methods(args.team, args.season)
            print(f"Comparison results for {args.team}:")
            print(results)
        elif args.analyze:
            results = app.analyze_team(args.team, args.season)
            print(f"Analysis results for {args.team}:")
            print(results)
        else:
            results = app.generate_lineup(args.team, args.season, args.method)
            print(f"Lineup for {args.team}:")
            print(f"Method: {results['method']}")
            print(f"Expected Runs: {results['evaluation']['expected_runs']:.2f}")
            print("Lineup:")
            for i, player in enumerate(results['lineup']):
                print(f"{i+1}. {player}")
    
    except KeyboardInterrupt:
        print("Application stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
