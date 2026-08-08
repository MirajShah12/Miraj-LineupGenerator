# Lineup Generator - Refactored Class-Based Architecture

A comprehensive baseball lineup optimization system with machine learning analysis, game simulation, and Discord bot integration.

## 🏗️ Architecture Overview

The system has been refactored into a clean, modular class-based architecture with the following components:

### Core Classes

- **`ConfigManager`** - Centralized configuration and constants management
- **`DataGenerator`** - MLB data fetching, processing, and team management
- **`PlayerModel`** - Individual player representation with statistics and capabilities
- **`LineupModel`** - Lineup representation with validation and optimization strategies
- **`GameGenerator`** - Baseball game simulation and lineup evaluation
- **`LineupAnalysis`** - Machine learning analysis and optimization
- **`DiscordBot`** - Discord bot functionality and user interactions
- **`LineupGeneratorApp`** - Main application orchestrating all components

## 🚀 Features

### Lineup Optimization Methods
- **Traditional** - Classic lineup construction based on OBP, wOBA, and power
- **wOBA-based** - Optimization using weighted On-Base Average
- **ML-based** - Machine learning optimization using Random Forest and Deep Learning
- **Brute Force** - Testing all possible lineup permutations
- **Genetic Algorithm** - Evolutionary optimization approach
- **Simulated Annealing** - Probabilistic optimization method

### Game Simulation
- Comprehensive baseball outcome modeling (singles, doubles, triples, home runs, walks, strikeouts, etc.)
- Advanced situational hitting (sacrifice flies, double plays, runner advancement)
- Run expectancy matrix integration
- Multiple simulation methods for statistical accuracy

### Machine Learning Analysis
- Position-specific feature importance analysis
- Random Forest regression models for each lineup position
- Deep Learning models with TensorFlow/Keras
- Training data generation with random lineup sampling
- Feature engineering for lineup optimization

### Discord Bot Integration
- Real-time lineup generation commands
- Team comparison functionality
- ML analysis commands
- Player roster queries
- Comprehensive help system

## 📦 Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Usage

### Basic Usage

```python
from main_application import LineupGeneratorApp, ApplicationConfig

# Initialize application
app_config = ApplicationConfig(default_season=2024)
app = LineupGeneratorApp(app_config)
app.initialize()

# Generate lineup
results = app.generate_lineup('NYY', 2024, 'traditional')
print(f"Expected Runs: {results['evaluation']['expected_runs']:.2f}")
```

### Command Line Usage

```bash
# Generate lineup
python main_application.py --team NYY --season 2024 --method traditional

# Compare methods
python main_application.py --team NYY --compare

# Run Discord bot
python main_application.py --team NYY --discord-token YOUR_TOKEN --run-bot
```

### Discord Bot Commands

- `!lineup <team> [season] [method]` - Generate optimal lineup
- `!compare <team> [season]` - Compare different methods
- `!analyze <team> [season]` - Perform ML analysis
- `!players <team> [season]` - Show team roster
- `!help` - Show available commands

## 🏗️ Class Structure

### ConfigManager
```python
from config_manager import ConfigManager

config = ConfigManager()
config.update_simulation_config(default_simulations=2000)
config.validate_team('NYY')  # True
```

### DataGenerator
```python
from data_generator import DataGenerator

data_gen = DataGenerator(config)
players = data_gen.get_team_roster('NYY', 2024)
top_players = data_gen.get_top_players(50, 2024)
```

### PlayerModel
```python
from player_model import PlayerModel

player = PlayerModel(player_data, config)
probs = player.get_outcome_probabilities()
suitability = player.get_position_suitability(1)  # Leadoff suitability
```

### LineupModel
```python
from lineup_model import LineupModel

lineup = LineupModel(players, config)
result = lineup.optimize_lineup_order('traditional')
features = lineup.get_lineup_features()
```

### GameGenerator
```python
from game_generator import GameGenerator

game_gen = GameGenerator(config)
evaluation = game_gen.evaluate_lineup(lineup, 1000)
comparison = game_gen.compare_lineups([lineup1, lineup2], 1000)
```

### LineupAnalysis
```python
from lineup_analysis import LineupAnalysis

analysis = LineupAnalysis(config, game_gen)
training_data = analysis.generate_training_data(players, 1000, 100)
models = analysis.train_position_models(training_data)
importance = analysis.analyze_position_importance()
```

## 🔍 Key Improvements

### 1. Separation of Concerns
- Each class has a single, well-defined responsibility
- Clear interfaces between components
- Easy to test and maintain individual components

### 2. Configuration Management
- Centralized configuration with type safety
- Easy to update parameters without code changes
- Validation and default value handling

### 3. Error Handling
- Comprehensive error handling throughout
- Graceful fallbacks for data issues
- Clear error messages and logging

### 4. Extensibility
- Easy to add new optimization methods
- Modular ML model architecture
- Pluggable data sources

### 5. Performance
- Efficient data caching
- Parallel processing where appropriate
- Optimized simulation algorithms

## 📊 Example Output

```
=== REFACTORED LINEUP GENERATOR ===
Using new class-based architecture for better organization

1. Initializing Application...
✅ Application initialized successfully!

2. Generating Traditional Lineup for Yankees...
✅ Generated traditional lineup for NYY
Expected Runs: 5.74
Lineup:
  1. Aaron Judge - wOBA: 0.463
  2. Giancarlo Stanton - wOBA: 0.395
  3. DJ LeMahieu - wOBA: 0.320
  ...

3. Comparing Lineup Methods...
✅ Compared 3 methods for NYY
Results:
  Traditional: 5.74 runs
  Woba: 5.68 runs
  Ml: 5.82 runs
```

## 🧪 Testing

The modular architecture makes testing much easier:

```python
# Test individual components
def test_player_model():
    player = PlayerModel(test_data, config)
    assert player.woba > 0
    assert len(player.get_outcome_probabilities()) > 0

def test_lineup_validation():
    lineup = LineupModel(players, config)
    assert len(lineup.lineup_order) == 9
    assert lineup.validate_lineup() is True
```

## 🔮 Future Enhancements

- **Pitcher Analysis** - Add pitcher vs. batter matchup analysis
- **Defensive Metrics** - Incorporate defensive statistics
- **Injury Analysis** - Factor in player health and availability
- **Weather Conditions** - Account for ballpark and weather effects
- **Advanced ML** - Implement more sophisticated ML models
- **Real-time Updates** - Live data integration for current games

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! The modular architecture makes it easy to:
- Add new optimization methods
- Implement additional ML models
- Extend the Discord bot functionality
- Improve simulation accuracy
- Add new data sources

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team.