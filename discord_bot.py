"""
Discord Bot for Lineup Generator
Handles Discord bot functionality and user interactions
"""

import discord
from discord.ext import commands
import asyncio
import pandas as pd
import nest_asyncio
from typing import Dict, List, Optional, Any
import traceback

from config_manager import ConfigManager
from data_generator import DataGenerator
from game_generator import GameGenerator
from lineup_analysis import LineupAnalysis
from lineup_model import LineupModel

# Apply nest_asyncio to handle event loops
nest_asyncio.apply()


class DiscordBot:
    """Discord bot for lineup generation and analysis"""
    
    def __init__(self, config_manager: ConfigManager, data_generator: DataGenerator, 
                 game_generator: GameGenerator, lineup_analysis: LineupAnalysis):
        """
        Initialize Discord bot
        
        Args:
            config_manager: ConfigManager instance
            data_generator: DataGenerator instance
            game_generator: GameGenerator instance
            lineup_analysis: LineupAnalysis instance
        """
        self.config_manager = config_manager
        self.data_generator = data_generator
        self.game_generator = game_generator
        self.lineup_analysis = lineup_analysis
        
        # Bot setup
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(
            command_prefix=self.config_manager.bot.command_prefix,
            intents=intents
        )
        
        # Bot state
        self.is_running = False
        self.current_season = 2024
        
        # Setup commands
        self._setup_commands()
    
    def _setup_commands(self):
        """Setup Discord bot commands"""
        
        @self.bot.event
        async def on_ready():
            print(f"Bot logged in as {self.bot.user.name}")
            self.is_running = True
        
        @self.bot.command(name='lineup')
        async def lineup_command(ctx, team: str, season: int = None, method: str = 'traditional'):
            """
            Generate optimal lineup for a team
            
            Usage: !lineup <team> [season] [method]
            Example: !lineup NYY 2024 traditional
            """
            try:
                if season is None:
                    season = self.current_season
                
                # Validate team
                if not self.config_manager.validate_team(team):
                    valid_teams = ', '.join(self.config_manager.mlb_teams)
                    await ctx.send(f"Invalid team abbreviation. Valid teams: {valid_teams}")
                    return
                
                # Validate method
                valid_methods = ['traditional', 'woba', 'ml', 'brute_force', 'genetic']
                if method not in valid_methods:
                    await ctx.send(f"Invalid method. Valid methods: {', '.join(valid_methods)}")
                    return
                
                await ctx.send(f"Generating {method} lineup for {team} ({season})...")
                
                # Get team roster
                players = self.data_generator.get_team_roster(team, season)
                
                if len(players) < 9:
                    await ctx.send(f"Not enough players found for {team}. Found {len(players)} players.")
                    return
                
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
                
                # Format response
                response = self._format_lineup_response(team, season, method, lineup, evaluation)
                
                # Split response if too long
                await self._send_long_message(ctx, response)
                
            except Exception as e:
                await ctx.send(f"Error generating lineup: {str(e)}")
                print(f"Error in lineup command: {e}")
                traceback.print_exc()
        
        @self.bot.command(name='compare')
        async def compare_command(ctx, team: str, season: int = None):
            """
            Compare different lineup optimization methods
            
            Usage: !compare <team> [season]
            Example: !compare NYY 2024
            """
            try:
                if season is None:
                    season = self.current_season
                
                if not self.config_manager.validate_team(team):
                    valid_teams = ', '.join(self.config_manager.mlb_teams)
                    await ctx.send(f"Invalid team abbreviation. Valid teams: {valid_teams}")
                    return
                
                await ctx.send(f"Comparing lineup methods for {team} ({season})...")
                
                # Get team roster
                players = self.data_generator.get_team_roster(team, season)
                
                if len(players) < 9:
                    await ctx.send(f"Not enough players found for {team}. Found {len(players)} players.")
                    return
                
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
                
                # Format response
                response = self._format_comparison_response(team, season, comparison)
                await self._send_long_message(ctx, response)
                
            except Exception as e:
                await ctx.send(f"Error comparing lineups: {str(e)}")
                print(f"Error in compare command: {e}")
                traceback.print_exc()
        
        @self.bot.command(name='analyze')
        async def analyze_command(ctx, team: str, season: int = None):
            """
            Perform detailed lineup analysis with ML models
            
            Usage: !analyze <team> [season]
            Example: !analyze NYY 2024
            """
            try:
                if season is None:
                    season = self.current_season
                
                if not self.config_manager.validate_team(team):
                    valid_teams = ', '.join(self.config_manager.mlb_teams)
                    await ctx.send(f"Invalid team abbreviation. Valid teams: {valid_teams}")
                    return
                
                await ctx.send(f"Performing ML analysis for {team} ({season})...")
                
                # Get team roster
                players = self.data_generator.get_team_roster(team, season)
                
                if len(players) < 9:
                    await ctx.send(f"Not enough players found for {team}. Found {len(players)} players.")
                    return
                
                # Generate training data if not available
                if self.lineup_analysis.training_data is None:
                    await ctx.send("Generating training data for ML models...")
                    self.lineup_analysis.generate_training_data(players, 1000, 100)
                
                # Train models if not already trained
                if not self.lineup_analysis.ml_models:
                    await ctx.send("Training ML models...")
                    self.lineup_analysis.train_position_models()
                
                # Analyze position importance
                await ctx.send("Analyzing position importance...")
                importance = self.lineup_analysis.analyze_position_importance()
                
                # Optimize using ML
                optimized_lineup = self.lineup_analysis.optimize_lineup_ml(players, 'genetic')
                
                # Evaluate optimized lineup
                evaluation = self.game_generator.evaluate_lineup(optimized_lineup, 1000)
                
                # Format response
                response = self._format_analysis_response(team, season, optimized_lineup, evaluation, importance)
                await self._send_long_message(ctx, response)
                
            except Exception as e:
                await ctx.send(f"Error in analysis: {str(e)}")
                print(f"Error in analyze command: {e}")
                traceback.print_exc()
        
        @self.bot.command(name='players')
        async def players_command(ctx, team: str, season: int = None):
            """
            Show team roster with player statistics
            
            Usage: !players <team> [season]
            Example: !players NYY 2024
            """
            try:
                if season is None:
                    season = self.current_season
                
                if not self.config_manager.validate_team(team):
                    valid_teams = ', '.join(self.config_manager.mlb_teams)
                    await ctx.send(f"Invalid team abbreviation. Valid teams: {valid_teams}")
                    return
                
                # Get team roster
                players = self.data_generator.get_team_roster(team, season)
                
                if not players:
                    await ctx.send(f"No players found for {team} in {season}")
                    return
                
                # Format response
                response = self._format_players_response(team, season, players)
                await self._send_long_message(ctx, response)
                
            except Exception as e:
                await ctx.send(f"Error getting players: {str(e)}")
                print(f"Error in players command: {e}")
                traceback.print_exc()
        
        @self.bot.command(name='help')
        async def help_command(ctx):
            """Show available commands"""
            help_text = """
**Available Commands:**

`!lineup <team> [season] [method]` - Generate optimal lineup
- team: Team abbreviation (e.g., NYY, LAD)
- season: Year (default: 2024)
- method: traditional, woba, ml, brute_force, genetic (default: traditional)

`!compare <team> [season]` - Compare different lineup methods

`!analyze <team> [season]` - Perform detailed ML analysis

`!players <team> [season]` - Show team roster

`!help` - Show this help message

**Examples:**
- `!lineup NYY 2024 traditional`
- `!compare LAD`
- `!analyze ATL 2024`
- `!players SF`
            """
            await ctx.send(help_text)
        
        @self.bot.command(name='status')
        async def status_command(ctx):
            """Show bot status and configuration"""
            status = f"""
**Bot Status:**
- Running: {self.is_running}
- Current Season: {self.current_season}
- Commands Available: {len(self.bot.commands)}
- ML Models Trained: {len(self.lineup_analysis.ml_models)}
- Training Data Size: {len(self.lineup_analysis.training_data) if self.lineup_analysis.training_data is not None else 0}

**Configuration:**
- Default Simulations: {self.config_manager.simulation.default_simulations}
- Min Plate Appearances: {self.config_manager.simulation.min_plate_appearances}
- ML Training Lineups: {self.config_manager.ml.n_lineups_for_training}
            """
            await ctx.send(status)
    
    def _format_lineup_response(self, team: str, season: int, method: str, 
                              lineup: LineupModel, evaluation: Dict[str, Any]) -> str:
        """Format lineup response for Discord"""
        response = f"**{team} Optimal Lineup ({season}) - {method.title()} Method**\n\n"
        
        # Add lineup order
        response += "**Lineup Order:**\n"
        for i, player_name in enumerate(lineup.lineup_order):
            player = lineup.get_player_by_name(player_name)
            if player:
                response += f"{i+1}. {player.name} - wOBA: {player.woba:.3f}\n"
        
        # Add evaluation results
        response += f"\n**Performance:**\n"
        response += f"Expected Runs: {evaluation['expected_runs']:.2f}\n"
        response += f"Confidence Interval: {evaluation['confidence_interval'][0]:.2f} - {evaluation['confidence_interval'][1]:.2f}\n"
        response += f"Simulations: {evaluation['simulation_count']}\n"
        
        # Add lineup summary
        summary = lineup.get_lineup_summary()
        response += f"\n**Lineup Summary:**\n"
        response += f"Average wOBA: {summary['avg_wOBA']:.3f}\n"
        response += f"Average OBP: {summary['avg_OBP']:.3f}\n"
        response += f"Average SLG: {summary['avg_SLG']:.3f}\n"
        
        return response
    
    def _format_comparison_response(self, team: str, season: int, 
                                   comparison: Dict[str, Any]) -> str:
        """Format comparison response for Discord"""
        response = f"**{team} Lineup Comparison ({season})**\n\n"
        
        if not comparison['lineups']:
            return response + "No lineups to compare."
        
        # Add comparison summary
        summary = comparison['comparison_summary']
        response += f"**Comparison Summary:**\n"
        response += f"Best Method: {comparison['best_lineup']['lineup']} ({comparison['best_lineup']['expected_runs']:.2f} runs)\n"
        response += f"Worst Method: {comparison['worst_lineup']['lineup']} ({comparison['worst_lineup']['expected_runs']:.2f} runs)\n"
        response += f"Runs Difference: {summary['runs_spread']:.2f}\n\n"
        
        # Add detailed results
        response += "**Detailed Results:**\n"
        for i, lineup_result in enumerate(comparison['lineups']):
            method = ['Traditional', 'wOBA', 'ML'][i] if i < 3 else f'Method {i+1}'
            response += f"{method}: {lineup_result['expected_runs']:.2f} runs\n"
        
        return response
    
    def _format_analysis_response(self, team: str, season: int, lineup: LineupModel,
                                 evaluation: Dict[str, Any], importance: Dict[int, pd.DataFrame]) -> str:
        """Format analysis response for Discord"""
        response = f"**{team} ML Analysis ({season})**\n\n"
        
        # Add optimized lineup
        response += "**ML-Optimized Lineup:**\n"
        for i, player_name in enumerate(lineup.lineup_order):
            player = lineup.get_player_by_name(player_name)
            if player:
                response += f"{i+1}. {player.name} - wOBA: {player.woba:.3f}\n"
        
        # Add performance
        response += f"\n**Performance:**\n"
        response += f"Expected Runs: {evaluation['expected_runs']:.2f}\n"
        response += f"Confidence Interval: {evaluation['confidence_interval'][0]:.2f} - {evaluation['confidence_interval'][1]:.2f}\n"
        
        # Add top position insights
        response += f"\n**Key Position Insights:**\n"
        for pos in [1, 2, 3, 4]:  # Show top 4 positions
            if pos in importance:
                top_feature = importance[pos].iloc[0]
                response += f"Position {pos}: {top_feature['feature']} (importance: {top_feature['importance']:.3f})\n"
        
        return response
    
    def _format_players_response(self, team: str, season: int, players: List[PlayerModel]) -> str:
        """Format players response for Discord"""
        response = f"**{team} Roster ({season})**\n\n"
        
        # Sort players by wOBA
        sorted_players = sorted(players, key=lambda p: p.woba, reverse=True)
        
        response += "**Top Players by wOBA:**\n"
        for i, player in enumerate(sorted_players[:10]):  # Show top 10
            response += f"{i+1}. {player.name} - wOBA: {player.woba:.3f}, OBP: {player.obp:.3f}, SLG: {player.slg:.3f}\n"
        
        if len(sorted_players) > 10:
            response += f"\n... and {len(sorted_players) - 10} more players"
        
        return response
    
    async def _send_long_message(self, ctx, message: str):
        """Send long message by splitting if necessary"""
        max_length = self.config_manager.bot.max_message_length
        
        if len(message) <= max_length:
            await ctx.send(message)
        else:
            # Split message into chunks
            chunks = []
            current_chunk = ""
            
            for line in message.split('\n'):
                if len(current_chunk + line + '\n') > max_length:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = line + '\n'
                    else:
                        # Single line is too long, split it
                        chunks.append(line[:max_length])
                        current_chunk = line[max_length:] + '\n'
                else:
                    current_chunk += line + '\n'
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Send chunks
            for chunk in chunks:
                await ctx.send(chunk)
                await asyncio.sleep(0.5)  # Small delay between messages
    
    def run(self, token: str):
        """
        Run the Discord bot
        
        Args:
            token: Discord bot token
        """
        try:
            self.bot.run(token)
        except Exception as e:
            print(f"Error running bot: {e}")
            traceback.print_exc()
    
    def stop(self):
        """Stop the Discord bot"""
        if self.is_running:
            self.bot.close()
            self.is_running = False
            print("Bot stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get bot status information"""
        return {
            'is_running': self.is_running,
            'current_season': self.current_season,
            'commands_count': len(self.bot.commands),
            'ml_models_trained': len(self.lineup_analysis.ml_models),
            'training_data_size': len(self.lineup_analysis.training_data) if self.lineup_analysis.training_data is not None else 0
        }
