"""
Game Generator for Lineup Generator
Handles baseball game simulation and lineup evaluation
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor
import warnings

from config_manager import ConfigManager
from player_model import PlayerModel
from lineup_model import LineupModel, LineupResult

warnings.filterwarnings('ignore')


class GameGenerator:
    """Handles baseball game simulation and lineup evaluation"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize game generator
        
        Args:
            config_manager: ConfigManager instance for configuration
        """
        self.config_manager = config_manager
        self.base_matrix = np.array(config_manager.base_matrix)
        self.base_states = config_manager.base_states
    
    def simulate_at_bat(self, player: PlayerModel) -> str:
        """
        Simulate a single at-bat for a player
        
        Args:
            player: PlayerModel object
            
        Returns:
            String representing the outcome
        """
        return player.simulate_at_bat()
    
    def update_game_state(self, bases: List[int], outs: int, outcome: str, player: PlayerModel) -> Tuple[List[int], int, int]:
        """
        Update game state based on batting outcome
        
        Args:
            bases: Current base state [1st, 2nd, 3rd]
            outs: Current number of outs
            outcome: Batting outcome
            player: PlayerModel object
            
        Returns:
            Tuple of (new_bases, new_outs, runs_scored)
        """
        runs = 0
        new_bases = bases.copy()
        new_outs = outs
        
        if outcome == 'hr':
            # Home run - all runners score plus batter
            runs += sum(bases) + 1
            new_bases = [0, 0, 0]
            
        elif outcome == 'single':
            # Single - runner on 3rd scores; runner on 2nd scores (60% or 2 outs) or to 3rd; runner on 1st to 2nd (or 3rd with 2 outs)
            # Runner on 3rd
            if bases[2]:
                runs += 1
                new_bases[2] = 0
            # Runner on 2nd
            if bases[1]:
                if outs == 2 or np.random.random() < 0.60:
                    runs += 1
                    new_bases[1] = 0
                else:
                    new_bases[2] = 1
                    new_bases[1] = 0
            # Runner on 1st
            if bases[0]:
                if outs == 2 and new_bases[2] == 0 and np.random.random() < 0.40:
                    new_bases[2] = 1
                    new_bases[0] = 0
                else:
                    new_bases[1] = 1
                    new_bases[0] = 0
            # Batter to 1st
            new_bases[0] = 1
            
        elif outcome == 'double':
            # Double - runners on 2nd and 3rd score; runner on 1st scores (40% or 2 outs) or to 3rd
            if bases[2]:
                runs += 1
                new_bases[2] = 0
            if bases[1]:
                runs += 1
                new_bases[1] = 0
            if bases[0]:
                if outs == 2 or np.random.random() < 0.40:
                    runs += 1
                    new_bases[0] = 0
                else:
                    new_bases[2] = 1
                    new_bases[0] = 0
            new_bases[1] = 1
            
        elif outcome == 'triple':
            # Triple - all runners score, batter to 3rd
            runs += sum(bases)
            new_bases = [0, 0, 1]
            
        elif outcome in ['walk', 'hbp']:
            # Walk or HBP - forced advancement only!
            if bases[0] and bases[1] and bases[2]: # Bases loaded
                runs += 1
                new_bases = [1, 1, 1]
            elif bases[0] and bases[1]: # 1st & 2nd occupied
                new_bases = [1, 1, 1]
            elif bases[0] and bases[2]: # 1st & 3rd occupied
                new_bases = [1, 1, 1]
            elif bases[0]: # 1st occupied
                new_bases = [1, 1, 0]
            elif bases[1] and bases[2]: # 2nd & 3rd occupied
                new_bases = [1, 1, 1]
            elif bases[1]: # 2nd occupied only
                new_bases = [1, 1, 0]
            elif bases[2]: # 3rd occupied only
                new_bases = [1, 0, 1]
            else: # Bases empty
                new_bases = [1, 0, 0]
            
        elif outcome == 'sac_fly':
            # Sacrifice fly - runner on 3rd scores, batter out
            new_outs += 1
            if outs < 2:
                if bases[2]:
                    runs += 1
                    new_bases[2] = 0
                if bases[1] and not new_bases[2] and np.random.random() < 0.25:
                    new_bases[2] = 1
                    new_bases[1] = 0
                
        elif outcome == 'strikeout':
            new_outs += 1
            
        elif outcome == 'groundout':
            # Groundout - potential for double play
            new_outs += 1
            gdp_prob = player.gidp_rate
            # Double play if runner on 1st and less than 2 outs
            if bases[0] and outs < 2 and gdp_prob > np.random.random():
                new_outs = min(outs + 2, 3)
                new_bases[0] = 0
                # Runner on 3rd scores if 0 outs on double play
                if bases[2] and outs == 0:
                    runs += 1
                    new_bases[2] = 0
                if bases[1]:
                    new_bases[2] = 1
                    new_bases[1] = 0
            else:
                # Non-double play groundout (productive out / fielder choice)
                if bases[2] and outs < 2 and np.random.random() < 0.50:
                    runs += 1
                    new_bases[2] = 0
                if bases[1] and not new_bases[2] and np.random.random() < 0.35:
                    new_bases[2] = 1
                    new_bases[1] = 0
                if bases[0] and not new_bases[1]:
                    new_bases[1] = 1
                    new_bases[0] = 0
            
        elif outcome == 'flyout':
            # Flyout - runner on 3rd can tag up
            new_outs += 1
            if bases[2] and outs < 2 and np.random.random() < 0.70:
                runs += 1
                new_bases[2] = 0
                    
        elif outcome == 'out':
            # Regular out 
            new_outs += 1
            gdp_prob = player.gidp_rate
            if bases[0] and outs < 2 and gdp_prob > np.random.random():
                new_outs = min(outs + 2, 3)
                new_bases[0] = 0
                if bases[2] and outs == 0:
                    runs += 1
                    new_bases[2] = 0
                if bases[1]:
                    new_bases[2] = 1
                    new_bases[1] = 0
        
        return new_bases, new_outs, runs
    
    def simulate_inning(self, lineup: LineupModel, batter_idx: int) -> Tuple[int, int]:
        """
        Simulate a single inning
        
        Args:
            lineup: LineupModel object
            batter_idx: Starting batter index
            
        Returns:
            Tuple of (runs_scored, next_batter_idx)
        """
        outs = 0
        bases = [0, 0, 0]
        runs = 0
        
        while outs < 3:
            player_name = lineup.lineup_order[batter_idx % len(lineup.lineup_order)]
            player = lineup.get_player_by_name(player_name)
            
            if player is None:
                print(f"Warning: Player {player_name} not found in lineup")
                outs += 1
                batter_idx += 1
                continue
            
            outcome = self.simulate_at_bat(player)
            bases, outs, new_runs = self.update_game_state(bases, outs, outcome, player)
            runs += new_runs
            batter_idx += 1
        
        return runs, batter_idx
    
    def get_re24_value(self, bases: List[int], outs: int) -> float:
        """Returns RE24 run expectancy for current base state and outs"""
        if outs >= 3:
            return 0.0
        base_tuple = (int(bases[0]), int(bases[1]), int(bases[2]))
        try:
            state_idx = self.base_states.index(base_tuple)
            return float(self.base_matrix[state_idx][outs])
        except (ValueError, IndexError):
            return 0.0

    def simulate_game_pbp(self, lineup: LineupModel, game_id: str = "Game_0001", innings: int = 9) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Simulate a complete game and return total runs + play-by-play (PBP) events for RNN/LSTM sequence modeling
        """
        total_runs = 0
        pbp_events = []
        batter_idx = 0

        for inning in range(1, innings + 1):
            outs = 0
            bases = [0, 0, 0]
            prev_player = None
            prev_outcome = "start_inning"

            while outs < 3:
                l_pos = (batter_idx % len(lineup.lineup_order)) + 1
                p_name = lineup.lineup_order[l_pos - 1]
                player = lineup.get_player_by_name(p_name)

                if player is None:
                    outs += 1
                    batter_idx += 1
                    continue

                re_start = self.get_re24_value(bases, outs)
                pre_bases = (int(bases[0]), int(bases[1]), int(bases[2]))
                pre_outs = outs

                outcome = self.simulate_at_bat(player)
                new_bases, new_outs, runs = self.update_game_state(bases, outs, outcome, player)
                total_runs += runs
                re_end = self.get_re24_value(new_bases, new_outs)

                run_value_re24 = (re_end - re_start) + runs

                event = {
                    'Game ID': game_id,
                    'Inning': inning,
                    'Batter': player.name,
                    'Lineup Position': l_pos,
                    'Pre-AB Bases': pre_bases,
                    'Pre-AB Outs': pre_outs,
                    'on_1b': pre_bases[0],
                    'on_2b': pre_bases[1],
                    'on_3b': pre_bases[2],
                    'RE_start': re_start,
                    'Post-AB Bases': (int(new_bases[0]), int(new_bases[1]), int(new_bases[2])),
                    'Post-AB Outs': new_outs,
                    'RE_end': re_end,
                    'Runs_Scored': runs,
                    'Event Outcome': outcome,
                    'Run_Value_RE24': run_value_re24,
                    'prev_batter_wOBA': prev_player.woba if prev_player else 0.320,
                    'prev_batter_OBP': prev_player.obp if prev_player else 0.320,
                    'prev_batter_SLG': prev_player.slg if prev_player else 0.400,
                    'prev_outcome': prev_outcome,
                    'batter_wOBA': player.woba,
                    'batter_OBP': player.obp,
                    'batter_SLG': player.slg,
                    'batter_xwOBA': player.xwoba,
                    'batter_xBA': player.xba,
                    'batter_xSLG': player.xslg,
                    'batter_ISO': player.iso,
                    'batter_BB_rate': player.walk_rate,
                    'batter_K_rate': player.strikeout_rate,
                    'batter_HR_rate': player.hr_rate,
                    'batter_contact_rate': player.contact_rate
                }
                pbp_events.append(event)

                prev_player = player
                prev_outcome = outcome
                bases = new_bases
                outs = new_outs
                batter_idx += 1

        return total_runs, pbp_events

    def generate_pbp_dataset(self, player_pool: List[PlayerModel], n_games: int = 500) -> pd.DataFrame:
        """
        Generate play-by-play (PBP) dataset across random lineups for LSTM/RNN sequence training
        """
        print(f"Generating {n_games} games for PBP dataset...")
        all_events = []

        for g_idx in range(n_games):
            if (g_idx + 1) % max(1, n_games // 5) == 0 or g_idx == 0:
                print(f"  Simulated game {g_idx + 1}/{n_games}")

            game_id = f"Game_{g_idx+1:04d}"
            sub_players = list(np.random.choice(player_pool, size=9, replace=False))
            lineup = LineupModel(sub_players, self.config_manager)

            _, g_events = self.simulate_game_pbp(lineup, game_id=game_id)
            all_events.extend(g_events)

        df_pbp = pd.DataFrame(all_events)
        print(f"Generated PBP dataset with {len(df_pbp)} events.")
        return df_pbp

    def simulate_game(self, lineup: LineupModel, innings: int = None) -> int:
        """
        Simulate a complete game
        
        Args:
            lineup: LineupModel object
            innings: Number of innings to simulate
            
        Returns:
            Total runs scored
        """
        if innings is None:
            innings = self.config_manager.simulation.default_innings
        
        total_runs = 0
        batter_idx = 0
        
        for _ in range(innings):
            runs, batter_idx = self.simulate_inning(lineup, batter_idx)
            total_runs += runs
        
        return total_runs
    
    def simulate_multiple_games(self, lineup: LineupModel, n_games: int = None) -> List[int]:
        """
        Simulate multiple games with the same lineup
        
        Args:
            lineup: LineupModel object
            n_games: Number of games to simulate
            
        Returns:
            List of runs scored in each game
        """
        if n_games is None:
            n_games = self.config_manager.simulation.default_simulations
        
        game_results = []
        
        for _ in range(n_games):
            runs = self.simulate_game(lineup)
            game_results.append(runs)
            lineup.add_simulation_result(runs)
        
        return game_results
    
    def evaluate_lineup(self, lineup: LineupModel, n_simulations: int = None) -> Dict[str, Any]:
        """
        Evaluate a lineup with comprehensive statistics
        
        Args:
            lineup: LineupModel object
            n_simulations: Number of simulations to run
            
        Returns:
            Dictionary of evaluation results
        """
        if n_simulations is None:
            n_simulations = self.config_manager.simulation.default_simulations
        
        # Clear previous results
        lineup.simulation_results.clear()
        
        # Run simulations
        game_results = self.simulate_multiple_games(lineup, n_simulations)
        
        # Calculate statistics
        mean_runs = np.mean(game_results)
        std_runs = np.std(game_results)
        min_runs = np.min(game_results)
        max_runs = np.max(game_results)
        
        # Calculate confidence interval
        confidence_interval = lineup.get_confidence_interval()
        
        # Calculate percentiles
        percentiles = {
            '25th': np.percentile(game_results, 25),
            '50th': np.percentile(game_results, 50),
            '75th': np.percentile(game_results, 75),
            '90th': np.percentile(game_results, 90),
            '95th': np.percentile(game_results, 95)
        }
        
        return {
            'lineup': lineup.lineup_order,
            'expected_runs': mean_runs,
            'std_runs': std_runs,
            'min_runs': min_runs,
            'max_runs': max_runs,
            'confidence_interval': confidence_interval,
            'percentiles': percentiles,
            'simulation_count': n_simulations,
            'game_results': game_results
        }
    
    def compare_lineups(self, lineups: List[LineupModel], n_simulations: int = None) -> Dict[str, Any]:
        """
        Compare multiple lineups
        
        Args:
            lineups: List of LineupModel objects
            n_simulations: Number of simulations per lineup
            
        Returns:
            Dictionary of comparison results
        """
        if n_simulations is None:
            n_simulations = self.config_manager.simulation.default_simulations
        
        results = []
        
        for i, lineup in enumerate(lineups):
            print(f"Evaluating lineup {i+1}/{len(lineups)}...")
            evaluation = self.evaluate_lineup(lineup, n_simulations)
            results.append(evaluation)
        
        # Sort by expected runs
        results.sort(key=lambda x: x['expected_runs'], reverse=True)
        
        return {
            'lineups': results,
            'best_lineup': results[0] if results else None,
            'worst_lineup': results[-1] if results else None,
            'comparison_summary': {
                'best_runs': results[0]['expected_runs'] if results else 0,
                'worst_runs': results[-1]['expected_runs'] if results else 0,
                'runs_spread': results[0]['expected_runs'] - results[-1]['expected_runs'] if results else 0
            }
        }
    
    def optimize_lineup_order(self, players: List[PlayerModel], method: str = 'brute_force', 
                            n_simulations: int = None) -> LineupResult:
        """
        Optimize lineup order using different methods
        
        Args:
            players: List of PlayerModel objects
            method: Optimization method ('brute_force', 'genetic', 'ml')
            n_simulations: Number of simulations per lineup
            
        Returns:
            LineupResult with optimized lineup
        """
        if n_simulations is None:
            n_simulations = self.config_manager.simulation.default_simulations
        
        if method == 'brute_force':
            return self._optimize_brute_force(players, n_simulations)
        elif method == 'genetic':
            return self._optimize_genetic(players, n_simulations)
        elif method == 'ml':
            return self._optimize_ml(players, n_simulations)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _optimize_brute_force(self, players: List[PlayerModel], n_simulations: int) -> LineupResult:
        """Brute force optimization by testing all permutations"""
        if len(players) != 9:
            raise ValueError("Brute force optimization requires exactly 9 players")
        
        best_lineup = None
        best_runs = 0
        results = []
        
        # Test all permutations (9! = 362,880)
        print("Testing all lineup permutations...")
        
        for i, perm in enumerate(permutations(players)):
            if i % 10000 == 0:
                print(f"Tested {i}/{362880} permutations")
            
            # Create lineup with this permutation
            lineup = LineupModel(list(perm), self.config_manager)
            
            # Evaluate lineup
            evaluation = self.evaluate_lineup(lineup, n_simulations)
            runs = evaluation['expected_runs']
            
            results.append((list(perm), runs))
            
            if runs > best_runs:
                best_runs = runs
                best_lineup = list(perm)
        
        # Sort results
        results.sort(key=lambda x: x[1], reverse=True)
        
        return LineupResult(
            lineup=[p.name for p in best_lineup],
            expected_runs=best_runs,
            confidence_interval=(0, 0),  # Would need more detailed analysis
            simulation_count=n_simulations,
            optimization_method='brute_force'
        )
    
    def _optimize_genetic(self, players: List[PlayerModel], n_simulations: int) -> LineupResult:
        """Genetic algorithm optimization (simplified version)"""
        # This is a simplified genetic algorithm implementation
        # In practice, you'd want a more sophisticated approach
        
        population_size = 50
        generations = 20
        mutation_rate = 0.1
        
        # Initialize population
        population = []
        for _ in range(population_size):
            lineup_players = players[:9] if len(players) >= 9 else players + [players[0]] * (9 - len(players))
            np.random.shuffle(lineup_players)
            population.append(lineup_players)
        
        best_lineup = None
        best_runs = 0
        
        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")
            
            # Evaluate population
            fitness_scores = []
            for lineup_players in population:
                lineup = LineupModel(lineup_players, self.config_manager)
                evaluation = self.evaluate_lineup(lineup, n_simulations // 10)  # Reduced simulations for speed
                runs = evaluation['expected_runs']
                fitness_scores.append(runs)
                
                if runs > best_runs:
                    best_runs = runs
                    best_lineup = lineup_players
            
            # Selection and reproduction (simplified)
            # Sort by fitness
            sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            
            # Keep top 50% and generate new offspring
            new_population = []
            for i in range(population_size // 2):
                new_population.append(sorted_population[i][0])
            
            # Generate offspring through crossover and mutation
            for i in range(population_size // 2, population_size):
                parent1 = sorted_population[i % (population_size // 2)][0]
                parent2 = sorted_population[(i + 1) % (population_size // 2)][0]
                
                # Simple crossover
                child = parent1[:5] + parent2[5:]
                
                # Mutation
                if np.random.random() < mutation_rate:
                    # Swap two random players
                    idx1, idx2 = np.random.choice(len(child), 2, replace=False)
                    child[idx1], child[idx2] = child[idx2], child[idx1]
                
                new_population.append(child)
            
            population = new_population
        
        return LineupResult(
            lineup=[p.name for p in best_lineup],
            expected_runs=best_runs,
            confidence_interval=(0, 0),
            simulation_count=n_simulations,
            optimization_method='genetic'
        )
    
    def _optimize_ml(self, players: List[PlayerModel], n_simulations: int) -> LineupResult:
        """ML-based optimization (placeholder for integration with LineupAnalysis)"""
        # This would integrate with the LineupAnalysis class
        # For now, use a simple heuristic approach
        
        # Sort players by wOBA
        sorted_players = sorted(players, key=lambda p: p.woba, reverse=True)
        
        # Create lineup using ML-inspired heuristics
        lineup_players = []
        
        # Position 1: Best OBP among top players
        top_players = sorted_players[:5]
        leadoff = max(top_players, key=lambda p: p.obp)
        lineup_players.append(leadoff)
        
        # Position 2: Best xwOBA among remaining
        remaining = [p for p in sorted_players if p != leadoff]
        two_hole = max(remaining[:5], key=lambda p: p.xwoba)
        lineup_players.append(two_hole)
        
        # Positions 3-4: Next best wOBA players
        remaining = [p for p in remaining if p != two_hole]
        lineup_players.extend(remaining[:2])
        
        # Positions 5-9: Remaining players
        remaining = [p for p in remaining if p not in lineup_players]
        lineup_players.extend(remaining)
        
        # Ensure we have exactly 9 players
        lineup_players = lineup_players[:9]
        
        # Create and evaluate lineup
        lineup = LineupModel(lineup_players, self.config_manager)
        evaluation = self.evaluate_lineup(lineup, n_simulations)
        
        return LineupResult(
            lineup=[p.name for p in lineup_players],
            expected_runs=evaluation['expected_runs'],
            confidence_interval=evaluation['confidence_interval'],
            simulation_count=n_simulations,
            optimization_method='ml'
        )
