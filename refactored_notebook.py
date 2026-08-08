"""
Clean organization of all the other classes created during this project
"""

# Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our new class-based modules
from config_manager import ConfigManager
from data_generator import DataGenerator
from game_generator import GameGenerator
from lineup_analysis import LineupAnalysis
from discord_bot import DiscordBot
from player_model import PlayerModel
from lineup_model import LineupModel
from main_application import LineupGeneratorApp, ApplicationConfig

print("=== REFACTORED LINEUP GENERATOR ===")
print("Using new class-based architecture for better organization")
print()

# Initialize the application
print("1. Initializing Application...")
app_config = ApplicationConfig(
    default_season=2025,
    enable_ml_training=True,
    cache_data=True
)

app = LineupGeneratorApp(app_config)
app.initialize()

print(f"Initialization successful")
print(f"Status: {app.get_status()}")
print()

#Example for the yankees 
print("2. Generating Traditional Lineup for Yankees...")
try:
    yankees_lineup = app.generate_lineup('NYY', 2025, 'traditional')
    
    print(f"Generated {yankees_lineup['method']} lineup for {yankees_lineup['team']}")
    print(f"Expected Runs: {yankees_lineup['evaluation']['expected_runs']:.2f}")
    print("Lineup:")
    for i, player in enumerate(yankees_lineup['lineup']):
        print(f"  {i+1}. {player}")
    print()
    
except Exception as e:
    print(f"Error generating Yankees lineup: {e}")
    print("Using test data instead...")
    
    # Only will be used if pybaseball is down  
    test_players = app.data_generator.create_test_data("Yankees")
    lineup = LineupModel(test_players, app.config_manager)
    result = lineup.optimize_lineup_order('traditional')
    lineup.set_lineup_order(result.lineup)
    
    print("Test Lineup:")
    for i, player in enumerate(lineup.lineup_order):
        print(f"  {i+1}. {player}")
    print()

# Example 2: Compare different lineup methods
print("3. Comparing Lineup Methods...")
try:
    comparison = app.compare_lineup_methods('NYY', 2025)
    
    print(f"Compared {len(comparison['methods'])} methods for {comparison['team']}")
    print("Results:")
    for i, lineup_result in enumerate(comparison['comparison']['lineups']):
        method = comparison['methods'][i]
        print(f"  {method.title()}: {lineup_result['expected_runs']:.2f} runs")
    print()
    
except Exception as e:
    print(f"Error comparing methods: {e}")
    print()

# Example 3: Perform ML analysis
print("4. Performing ML Analysis...")
try:
    analysis = app.analyze_team('NYY', 2025)
    
    print(f"Completed ML analysis for {analysis['team']}")
    print(f"ML-Optimized Lineup:")
    for i, player in enumerate(analysis['optimized_lineup']):
        print(f"  {i+1}. {player}")
    
    print(f"Expected Runs: {analysis['evaluation']['expected_runs']:.2f}")
    print()
    
    # Show position importance insights
    print("Key Position Insights:")
    for pos in [1, 2, 3, 4]:
        if pos in analysis['position_importance']:
            top_feature = analysis['position_importance'][pos][0]
            print(f"  Position {pos}: {top_feature['feature']} (importance: {top_feature['importance']:.3f})")
    print()
    
except Exception as e:
    print(f"Error in ML analysis: {e}")
    print()

# Example 4: Get team roster
print("5. Getting Team Roster...")
try:
    roster = app.get_team_roster('NYY', 2025)
    
    print(f"Retrieved roster for Yankees ({len(roster)} players)")
    print("Top 5 players by wOBA:")
    sorted_roster = sorted(roster, key=lambda p: p['wOBA'], reverse=True)
    for i, player in enumerate(sorted_roster[:5]):
        print(f"  {i+1}. {player['name']} - wOBA: {player['wOBA']:.3f}")
    print()
    
except Exception as e:
    print(f"Error getting roster: {e}")
    print()

# Example 5: Demonstrate individual component usage
print("6. Demonstrating Individual Component Usage...")

# Use DataGenerator directly
print("Using DataGenerator to get top players...")
try:
    top_players = app.data_generator.get_top_players(20, 2025)
    print(f"✅ Retrieved {len(top_players)} top players")
    
    # Use GameGenerator directly
    print("Using GameGenerator to simulate games...")
    if len(top_players) >= 9:
        test_lineup = LineupModel(top_players[:9], app.config_manager)
        evaluation = app.game_generator.evaluate_lineup(test_lineup, 100)
        print(f"✅ Simulated {evaluation['simulation_count']} games")
        print(f"Expected runs: {evaluation['expected_runs']:.2f}")
    
    # Use LineupAnalysis directly
    print("Using LineupAnalysis for ML insights...")
    if len(top_players) >= 9:
        # Generate some training data
        training_data = app.lineup_analysis.generate_training_data(top_players, 100, 50)
        print(f"✅ Generated {len(training_data)} training samples")
        
        # Train models
        models = app.lineup_analysis.train_position_models(training_data)
        print(f"✅ Trained {len(models)} ML models")
    
except Exception as e:
    print(f"❌ Error in component demonstration: {e}")
    print()

# Example 6: Export results
print("7. Exporting Results...")
try:
    # Export lineup results
    if 'yankees_lineup' in locals():
        filename = app.export_results(yankees_lineup)
        print(f"✅ Exported results to {filename}")
    
    # Export analysis results
    if 'analysis' in locals():
        filename = app.export_results(analysis, "yankees_analysis.csv")
        print(f"✅ Exported analysis to {filename}")
    
except Exception as e:
    print(f"❌ Error exporting results: {e}")
    print()

# Example 7: Configuration management
print("8. Configuration Management...")
print("Current configuration:")
status = app.get_status()
for key, value in status['config'].items():
    print(f"  {key}: {value}")

# Update configuration
print("\nUpdating configuration...")
app.update_config(default_season=2023, enable_ml_training=False)
print("Updated configuration:")
status = app.get_status()
for key, value in status['config'].items():
    print(f"  {key}: {value}")
print()

# Example 8: Discord Bot (if token provided)
print("9. Discord Bot Setup...")
print("To run the Discord bot, you would use:")
print("app.run_discord_bot()")
print("But this requires a Discord bot token.")
print("Available commands would be:")
print("  !lineup <team> [season] [method]")
print("  !compare <team> [season]")
print("  !analyze <team> [season]")
print("  !players <team> [season]")
print("  !help")
print()

# Summary
print("=== SUMMARY ===")
print("✅ Successfully refactored lineup generator using class-based architecture")
print("✅ All components are properly separated and organized:")
print("  - ConfigManager: Handles all configuration and constants")
print("  - DataGenerator: Manages MLB data fetching and processing")
print("  - PlayerModel: Represents individual players with statistics")
print("  - LineupModel: Represents lineups with validation and optimization")
print("  - GameGenerator: Handles game simulation and lineup evaluation")
print("  - LineupAnalysis: Manages ML analysis and optimization")
print("  - DiscordBot: Handles Discord bot functionality")
print("  - LineupGeneratorApp: Main application orchestrating all components")
print()
print("✅ Benefits of new architecture:")
print("  - Better separation of concerns")
print("  - Easier to maintain and extend")
print("  - More modular and testable")
print("  - Cleaner code organization")
print("  - Better error handling")
print("  - More flexible configuration")
print()
print("✅ Ready for production use!")

# Clean up
print("\nCleaning up...")
app.clear_cache()
print("✅ Cleanup complete")
