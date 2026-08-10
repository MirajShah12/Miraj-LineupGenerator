"""
Generates the data for machine learning testing.
This is supposed to use the game generator and aggregate players to their teams
to create a dataset for machine learning testing.
"""

import pandas as pd
import numpy as np
import pybaseball as pyb
from typing import Dict, List, Optional, Tuple, Any
import warnings
from config_manager import ConfigManager
from player_model import PlayerModel

warnings.filterwarnings('ignore')


class DataGenerator:
    """Gathers all the data for game generation"""
    
    def __init__(self, config_manager: ConfigManager):

        self.config_manager = config_manager
        self.cached_data = {}
        self.last_fetch_season = None
    
    def get_all_players_with_teams(self, season: int = 2025) -> pd.DataFrame:
        """ Gets all the data from each player on each team for a given season """
        
        #caching data for easier access
        cache_key = f"all_players_{season}"
        
        if cache_key in self.cached_data and self.last_fetch_season == season:
            print(f"Using cached data for {season} season")
            return self.cached_data[cache_key]
        
        try:
            print(f"Fetching MLB data for {season} season...")
            df = pyb.batting_stats(season, qual=self.config_manager.simulation.min_plate_appearances)
            
            if df.empty:
                print(f"No data found for {season} season")
                return pd.DataFrame()
            
            # Cache the data
            self.cached_data[cache_key] = df
            self.last_fetch_season = season
            
            print(f"Successfully fetched {len(df)} players for {season} season")
            return df
            
        except Exception as e:
            print(f"Error retrieving batting stats for {season}: {e}")
            return pd.DataFrame()
    
    def filter_by_team(self, df: pd.DataFrame, team_abbr: str) -> pd.DataFrame:
        """ Finds each team in our list of teams and loads them for testing """
        if df.empty:
            return pd.DataFrame()
        
        # Find team column
        team_cols = [col for col in df.columns if 'team' in col.lower() or 'tm' in col.lower()]
        
        #Safeguard for failure to pull from pybaseball
        if not team_cols:
            print("No team column found in DataFrame")
            return pd.DataFrame()
        
        team_col = team_cols[0]
        filtered_df = df[df[team_col] == team_abbr.upper()].copy()
        
        if filtered_df.empty:
            print(f"No players found for team {team_abbr}")
        

        return filtered_df
    
    def aggregate_team_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Puts all the players on their teams with their abbreviation building the new dataframe
        Allows us to map each player to each other 
        """
        team_rosters = {}
        
        for team in self.config_manager.mlb_teams:
            team_df = self.filter_by_team(df, team)
            if not team_df.empty:
                team_rosters[team] = team_df
                print(f"Found {len(team_df)} players for {team}")
        
        return team_rosters
    
    def get_team_roster(self, team_abbr: str, season: int = 2025) -> List[PlayerModel]:
        """
        Get roster for a specific team as PlayerModel objects
        
        Args:
            team_abbr: Team abbreviation
            season: MLB season year
            
        Returns:
            List of PlayerModel objects
        """
        # Validate team
        if not self.config_manager.validate_team(team_abbr):
            raise ValueError(f"Invalid team abbreviation: {team_abbr}")
        
        # Get all players data
        all_players_df = self.get_all_players_with_teams(season)
        
        if all_players_df.empty:
            raise ValueError(f"No data available for {season} season")
        
        # Filter by team
        team_df = self.filter_by_team(all_players_df, team_abbr)
        
        if team_df.empty:
            raise ValueError(f"No players found for team {team_abbr}")
        
        # Convert to PlayerModel objects
        players = []
        for _, row in team_df.iterrows():
            try:
                player = PlayerModel(row, self.config_manager)
                players.append(player)
            except Exception as e:
                print(f"Error creating PlayerModel for {row.get('Name', 'Unknown')}: {e}")
                continue
        
        # Sort by wOBA (descending)
        players.sort(key=lambda p: p.woba, reverse=True)
        
        return players
    
    def get_top_players(self, n_players: int = 50, season: int = 2025) -> List[PlayerModel]:
        """
        Get top N players across all teams
        
        Args:
            n_players: Number of top players to return
            season: MLB season year
            
        Returns:
            List of top PlayerModel objects
        """
        all_players_df = self.get_all_players_with_teams(season)
        
        if all_players_df.empty:
            return []
        
        # Sort by wOBA and take top N
        top_players_df = all_players_df.nlargest(n_players, 'wOBA')
        
        players = []
        for _, row in top_players_df.iterrows():
            try:
                player = PlayerModel(row, self.config_manager)
                players.append(player)
            except Exception as e:
                print(f"Error creating PlayerModel for {row.get('Name', 'Unknown')}: {e}")
                continue
        
        return players
    
    def create_test_data(self, team_name: str = "Test Team") -> List[PlayerModel]:
        """
        Create test data for development and testing
        
        Args:
            team_name: Name for the test team
            
        Returns:
            List of PlayerModel objects with test data
        """
        test_data = {
            'Name': ['Gleyber Torres', 'Juan Soto', 'Aaron Judge', 'Austin Wells', 'Giancarlo Stanton',
                     'Jazz Chisholm Jr.', 'Anthony Rizzo', 'Anthony Volpe', 'Alex Verdugo'],
            'Team': [team_name] * 9,
            'PA': [650, 680, 700, 450, 450, 520, 350, 600, 590],
            '1B': [80, 95, 90, 45, 45, 65, 45, 80, 70],
            '2B': [26, 31, 36, 18, 20, 21, 12, 27, 28],
            '3B': [0, 4, 1, 1, 0, 4, 0, 7, 1],
            'HR': [15, 41, 58, 13, 27, 24, 8, 12, 13],
            'BB': [65, 129, 133, 47, 38, 45, 27, 42, 40],
            'HBP': [4, 4, 9, 3, 3, 3, 4, 3, 5],
            'SO': [130, 119, 171, 95, 140, 160, 65, 150, 90],
            'GDP': [8, 5, 8, 4, 12, 6, 5, 9, 9],
            'wOBA': [0.313, 0.421, 0.476, 0.315, 0.330, 0.325, 0.301, 0.289, 0.284],
            'OBP': [0.330, 0.419, 0.458, 0.322, 0.298, 0.324, 0.301, 0.293, 0.291],
            'SLG': [0.378, 0.569, 0.701, 0.395, 0.475, 0.436, 0.335, 0.364, 0.356],
            'xwOBA': [0.310, 0.420, 0.470, 0.310, 0.335, 0.320, 0.295, 0.285, 0.280],
            'xBA': [0.250, 0.285, 0.310, 0.240, 0.245, 0.240, 0.235, 0.240, 0.245],
            'xSLG': [0.370, 0.560, 0.690, 0.390, 0.480, 0.430, 0.330, 0.360, 0.350],
            'ISO': [0.128, 0.250, 0.343, 0.173, 0.227, 0.186, 0.134, 0.121, 0.115]
        }
        
        test_df = pd.DataFrame(test_data)
        players = []
        
        for _, row in test_df.iterrows():
            try:
                player = PlayerModel(row, self.config_manager)
                players.append(player)
            except Exception as e:
                print(f"Error creating test PlayerModel: {e}")
                continue
        
        return players
    
    def validate_player_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that player data contains required columns
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        missing_columns = []
        
        for col in self.config_manager.required_player_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        is_valid = len(missing_columns) == 0
        return is_valid, missing_columns
    
    def clean_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare player data for analysis
        
        Args:
            df: Raw player data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Remove duplicates based on Name
        df_clean = df.drop_duplicates(subset='Name', keep='first')
        
        # Filter out players with insufficient plate appearances
        min_pa = self.config_manager.simulation.min_plate_appearances
        df_clean = df_clean[df_clean['PA'] >= min_pa]
        
        # Fill missing values with reasonable defaults
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['wOBA', 'OBP', 'SLG', 'xwOBA', 'xBA', 'xSLG']:
                # Use league average for advanced metrics
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                # Use 0 for counting stats
                df_clean[col] = df_clean[col].fillna(0)
        
        return df_clean
    
    def get_league_averages(self, season: int = 2025) -> Dict[str, float]:
        """
        get league average for general testing 
        """
        all_players_df = self.get_all_players_with_teams(season)
        
        if all_players_df.empty:
            return {}
        
        # Calculate league averages
        league_averages = {
            'wOBA': all_players_df['wOBA'].mean(),
            'OBP': all_players_df['OBP'].mean(),
            'SLG': all_players_df['SLG'].mean(),
            'xwOBA': all_players_df['xwOBA'].mean(),
            'xBA': all_players_df['xBA'].mean(),
            'xSLG': all_players_df['xSLG'].mean(),
            'ISO': all_players_df['ISO'].mean(),
            'BB_rate': (all_players_df['BB'] / all_players_df['PA']).mean(),
            'K_rate': (all_players_df['SO'] / all_players_df['PA']).mean(),
            'HR_rate': (all_players_df['HR'] / all_players_df['PA']).mean()
        }
        
        return league_averages
    
    def export_team_data(self, team_abbr: str, season: int = 2025, filename: str = None) -> str:
        """ Converts data to csv """
        team_df = self.filter_by_team(self.get_all_players_with_teams(season), team_abbr)
        
        if team_df.empty:
            raise ValueError(f"No data found for team {team_abbr}")
        
        if filename is None:
            filename = f"{team_abbr}_{season}_roster.csv"
        
        team_df.to_csv(filename, index=False)
        print(f"Exported {len(team_df)} players to {filename}")
        
        return filename
    
    def clear_cache(self):
        """Clear cached data"""
        self.cached_data.clear()
        self.last_fetch_season = None
        print("Data cache cleared")
