"""
Lineup Analysis for Lineup Generator
Handles machine learning analysis, optimization, and statistical modeling
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings

from config_manager import ConfigManager
from player_model import PlayerModel
from lineup_model import LineupModel
from game_generator import GameGenerator

warnings.filterwarnings('ignore')


BENCHMARK_YANKEES_2024 = [
    'Gleyber Torres', 'Juan Soto', 'Aaron Judge', 'Austin Wells',
    'Giancarlo Stanton', 'Jazz Chisholm Jr.', 'Anthony Rizzo',
    'Anthony Volpe', 'Alex Verdugo'
]

class LineupAnalysis:
    """Handles machine learning analysis and optimization for lineups"""
    
    def __init__(self, config_manager: ConfigManager, game_generator: GameGenerator):
        """
        Initialize lineup analysis
        
        Args:
            config_manager: ConfigManager instance
            game_generator: GameGenerator instance for simulations
        """
        self.config_manager = config_manager
        self.game_generator = game_generator
        self.ml_models = {}
        self.scalers = {}
        self.training_data = None
        self.pbp_data = None
        self.feature_importance = {}
        
    def generate_training_data(self, players: List[PlayerModel], n_lineups: int = None, 
                             n_simulations: int = None) -> pd.DataFrame:
        """
        Generate training data for ML models by creating random lineups
        
        Args:
            players: List of PlayerModel objects
            n_lineups: Number of random lineups to generate
            n_simulations: Number of simulations per lineup
            
        Returns:
            DataFrame with lineup features and results
        """
        if n_lineups is None:
            n_lineups = self.config_manager.ml.n_lineups_for_training
        if n_simulations is None:
            n_simulations = self.config_manager.ml.n_simulations_per_lineup
        
        print(f"Generating {n_lineups} random lineups for ML training...")
        
        if len(players) < 9:
            raise ValueError(f"Need at least 9 players, got {len(players)}")
        
        lineup_data = []
        
        for i in range(n_lineups):
            if i % 100 == 0:
                print(f"Generated {i}/{n_lineups} lineups")
            
            # Create random lineup
            lineup_players = np.random.choice(players, size=9, replace=False).tolist()
            lineup = LineupModel(lineup_players, self.config_manager)
            
            # Simulate games with this lineup
            game_results = self.game_generator.simulate_multiple_games(lineup, n_simulations)
            avg_runs = np.mean(game_results)
            std_runs = np.std(game_results)
            
            # Get lineup features
            features = lineup.get_lineup_features()
            
            # Create one row per lineup with relative position features
            lineup_row = {
                'lineup_id': i,
                'total_runs': avg_runs,
                'runs_std': std_runs,
                **features
            }
            
            # Engineer relative position features (position stat minus team average)
            for pos in range(1, 10):
                lineup_row[f'pos_{pos}_wOBA_rel'] = lineup_row[f'pos_{pos}_wOBA'] - lineup_row['lineup_wOBA_avg']
                lineup_row[f'pos_{pos}_OBP_rel'] = lineup_row[f'pos_{pos}_OBP'] - lineup_row['lineup_OBP_avg']
                lineup_row[f'pos_{pos}_SLG_rel'] = lineup_row[f'pos_{pos}_SLG'] - lineup_row['lineup_SLG_avg']
                lineup_row[f'pos_{pos}_ISO_rel'] = lineup_row[f'pos_{pos}_ISO'] - lineup_row['lineup_power_avg']
                
            lineup_data.append(lineup_row)
        
        self.training_data = pd.DataFrame(lineup_data)
        print(f"Generated training data with {len(self.training_data)} lineups")
        
        return self.training_data
    
    def train_position_models(self, training_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Train ML models: an overall lineup model predicting total_runs from all 9 position features,
        plus position-specific analysis models.
        
        Args:
            training_data: Training data DataFrame (uses self.training_data if None)
            
        Returns:
            Dictionary of trained models
        """
        if training_data is None:
            training_data = self.training_data
        
        if training_data is None:
            raise ValueError("No training data available. Call generate_training_data first.")
        
        print("Training ML models for lineup optimization...")
        
        # Build list of all position features across all 9 positions (excluding invariant lineup-level averages)
        all_features = []
        for pos in range(1, 10):
            all_features.extend([
                f'pos_{pos}_wOBA', f'pos_{pos}_OBP', f'pos_{pos}_SLG', 
                f'pos_{pos}_xwOBA', f'pos_{pos}_xBA', f'pos_{pos}_xSLG', 
                f'pos_{pos}_ISO', f'pos_{pos}_BB_rate', f'pos_{pos}_K_rate', 
                f'pos_{pos}_HR_rate', f'pos_{pos}_contact_rate',
                f'pos_{pos}_wOBA_rel', f'pos_{pos}_OBP_rel', f'pos_{pos}_SLG_rel', f'pos_{pos}_ISO_rel'
            ])
        full_feature_cols = [col for col in all_features if col in training_data.columns]
        
        X = training_data[full_feature_cols]
        y = training_data['total_runs']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config_manager.ml.test_size, 
            random_state=self.config_manager.ml.random_state
        )
        
        rf = RandomForestRegressor(
            n_estimators=self.config_manager.ml.n_estimators,
            max_depth=self.config_manager.ml.max_depth,
            random_state=self.config_manager.ml.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        importance_df = pd.DataFrame({
            'feature': full_feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.ml_models['overall'] = {
            'model': rf,
            'scaler': scaler,
            'features': full_feature_cols,
            'mse': mse,
            'r2': r2,
            'importance': importance_df
        }
        self.scalers['overall'] = scaler
        
        print(f"  Overall Model R2 Score: {r2:.4f}, MSE: {mse:.4f}")
        print("  Top 5 overall features:")
        for _, row in importance_df.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
            
        # Train position-specific models for analysis
        for pos in range(1, 10):
            pos_features = [f'pos_{pos}_wOBA', f'pos_{pos}_OBP', f'pos_{pos}_SLG', 
                           f'pos_{pos}_xwOBA', f'pos_{pos}_xBA', f'pos_{pos}_xSLG', 
                           f'pos_{pos}_ISO', f'pos_{pos}_BB_rate', f'pos_{pos}_K_rate', 
                           f'pos_{pos}_HR_rate', f'pos_{pos}_contact_rate']
            pos_features = [c for c in pos_features if c in training_data.columns]
            
            X_pos = training_data[pos_features]
            scaler_pos = StandardScaler()
            X_pos_scaled = scaler_pos.fit_transform(X_pos)
            
            rf_pos = RandomForestRegressor(
                n_estimators=50, max_depth=5,
                random_state=self.config_manager.ml.random_state, n_jobs=-1
            )
            rf_pos.fit(X_pos_scaled, y)
            
            self.ml_models[pos] = {
                'model': rf_pos,
                'scaler': scaler_pos,
                'features': pos_features,
                'importance': pd.DataFrame({
                    'feature': pos_features,
                    'importance': rf_pos.feature_importances_
                }).sort_values('importance', ascending=False)
            }
        
        return self.ml_models

    def analyze_position_importance(self, training_data: pd.DataFrame = None) -> Dict[int, pd.DataFrame]:
        """
        Analyze feature importance for each lineup position
        
        Args:
            training_data: Training data DataFrame
            
        Returns:
            Dictionary of feature importance by position
        """
        if training_data is None:
            training_data = self.training_data
        
        if training_data is None:
            raise ValueError("No training data available.")
        
        if 'overall' not in self.ml_models:
            self.train_position_models(training_data)
            
        position_analysis = {}
        for pos in range(1, 10):
            if pos in self.ml_models and 'importance' in self.ml_models[pos]:
                position_analysis[pos] = self.ml_models[pos]['importance']
                
        self.feature_importance = position_analysis
        return position_analysis
    
    def predict_lineup_performance(self, lineup: LineupModel) -> Dict[str, Any]:
        """
        Predict lineup performance using trained overall ML model
        
        Args:
            lineup: LineupModel object
            
        Returns:
            Dictionary of predictions and confidence metrics
        """
        if 'overall' not in self.ml_models:
            raise ValueError("No trained overall model available. Call train_position_models first.")
        
        model_info = self.ml_models['overall']
        features_dict = lineup.get_lineup_features()
        
        X = np.array([[features_dict.get(f, 0.0) for f in model_info['features']]])
        X_scaled = model_info['scaler'].transform(X)
        
        pred = model_info['model'].predict(X_scaled)[0]
        
        # Estimate tree prediction variance for confidence interval
        tree_preds = [tree.predict(X_scaled)[0] for tree in model_info['model'].estimators_]
        pred_std = np.std(tree_preds)
        
        return {
            'overall_prediction': float(pred),
            'confidence_interval': (float(pred - 1.96 * pred_std), float(pred + 1.96 * pred_std)),
            'pred_std': float(pred_std)
        }
    
    def optimize_lineup_ml(self, players: List[PlayerModel], method: str = 'full_search') -> LineupModel:
        """
        Optimize lineup using ML model by evaluating player permutations
        
        Args:
            players: List of PlayerModel objects (exactly 9 players for full_search, or top 9 selected)
            method: Optimization method ('full_search', 'genetic', 'simulated_annealing')
            
        Returns:
            Optimized LineupModel
        """
        if len(players) < 9:
            raise ValueError("Need at least 9 players for optimization")
        
        # If overall model is not yet trained, train it
        if 'overall' not in self.ml_models:
            print("Overall ML model not trained yet. Generating training data and training model...")
            self.generate_training_data(players, n_lineups=2000, n_simulations=50)
            self.train_position_models()
            
        target_players = players[:9]
        
        from itertools import permutations
        
        best_lineup_players = list(target_players)
        best_pred_runs = -np.inf
        
        print("Evaluating lineup permutations using ML model...")
        model_info = self.ml_models['overall']
        model = model_info['model']
        scaler = model_info['scaler']
        feature_cols = model_info['features']
        
        # Pre-build feature vectors for all permutations
        perm_list = list(permutations(target_players))
        X_rows = []
        for perm in perm_list:
            temp_lineup = LineupModel(list(perm), self.config_manager)
            f_dict = temp_lineup.get_lineup_features()
            for pos in range(1, 10):
                f_dict[f'pos_{pos}_wOBA_rel'] = f_dict[f'pos_{pos}_wOBA'] - f_dict['lineup_wOBA_avg']
                f_dict[f'pos_{pos}_OBP_rel'] = f_dict[f'pos_{pos}_OBP'] - f_dict['lineup_OBP_avg']
                f_dict[f'pos_{pos}_SLG_rel'] = f_dict[f'pos_{pos}_SLG'] - f_dict['lineup_SLG_avg']
                f_dict[f'pos_{pos}_ISO_rel'] = f_dict[f'pos_{pos}_ISO'] - f_dict['lineup_power_avg']
            X_rows.append([f_dict.get(c, 0.0) for c in feature_cols])
            
        X_mat = np.array(X_rows)
        X_mat_scaled = scaler.transform(X_mat)
        
        preds = model.predict(X_mat_scaled)
        
        # Select top K candidates from ML predictions for simulation refinement
        top_k = min(50, len(preds))
        top_indices = np.argsort(preds)[-top_k:]
        
        best_lineup_players = list(perm_list[top_indices[-1]])
        best_sim_runs = -np.inf
        
        # Refine top candidate lineups using Monte Carlo simulation with paired seed control
        if self.game_generator:
            print(f"Refining top {top_k} ML candidate lineups + heuristic seeds via paired Monte Carlo simulation...")
            
            # Seed candidate pool with traditional and wOBA heuristic orders
            trad_lineup = LineupModel(target_players, self.config_manager)._optimize_traditional()
            woba_lineup = LineupModel(target_players, self.config_manager)._optimize_woba()
            
            base_lineup = LineupModel(target_players, self.config_manager)
            candidate_lineups = [list(perm_list[idx]) for idx in top_indices]
            
            # Ensure traditional and wOBA orders are in candidates
            for seed_m in [trad_lineup, woba_lineup]:
                p_objs = [base_lineup.get_player_by_name(n) for n in seed_m.lineup]
                if all(p is not None for p in p_objs):
                    candidate_lineups.append(p_objs)
            
            # Reference lineup for paired seed comparison
            ref_lineup_model = LineupModel([base_lineup.get_player_by_name(n) for n in trad_lineup.lineup], self.config_manager)
            
            N_REF_GAMES = 2000
            best_delta_runs = -np.inf
            best_lineup_players = list(target_players)
            
            for cand_players in candidate_lineups:
                cand_model = LineupModel(cand_players, self.config_manager)
                
                # Paired game simulation
                delta_runs_list = []
                for g_idx in range(N_REF_GAMES):
                    g_seed = 5000 + g_idx
                    
                    np.random.seed(g_seed)
                    cand_r = self.game_generator.simulate_game(cand_model)
                    
                    np.random.seed(g_seed)
                    ref_r = self.game_generator.simulate_game(ref_lineup_model)
                    
                    delta_runs_list.append(cand_r - ref_r)
                
                mean_delta = np.mean(delta_runs_list)
                if mean_delta > best_delta_runs:
                    best_delta_runs = mean_delta
                    best_lineup_players = cand_players
                    
            print(f"ML+Paired Simulation Optimization Complete! Best Delta Runs vs Traditional: {best_delta_runs:+.4f}")
        else:
            best_pred_runs = preds[top_indices[-1]]
            print(f"ML Optimization Complete! Best Predicted Runs: {best_pred_runs:.3f}")
        
        optimized_lineup = LineupModel(best_lineup_players, self.config_manager)
        optimized_lineup.set_lineup_order([p.name for p in best_lineup_players])
        return optimized_lineup

    def train_lstm_model(self, df_pbp: pd.DataFrame = None, max_seq_len: int = 15) -> Dict[str, Any]:
        """
        Train a sequential Neural Network (LSTM / RNN) on play-by-play (PBP) inning sequences.
        Models situation context (outs, bases, RE_start) and previous batter outcome impact.
        """
        if df_pbp is None:
            if self.pbp_data is None:
                raise ValueError("No PBP data available. Call game_generator.generate_pbp_dataset first.")
            df_pbp = self.pbp_data
        else:
            self.pbp_data = df_pbp

        print("Preparing PBP sequence data for LSTM/RNN training...")
        feature_cols = [
            'batter_OBP', 'batter_SLG', 'batter_ISO', 'batter_BB_rate', 'batter_contact_rate',
            'batter_xwOBA', 'batter_wOBA', 'batter_xBA', 'batter_HR_rate', 'batter_K_rate',
            'prev_batter_wOBA', 'prev_batter_OBP', 'prev_batter_SLG',
            'on_1b', 'on_2b', 'on_3b', 'Pre-AB Outs'
        ]
        feature_cols = [c for c in feature_cols if c in df_pbp.columns]

        sequences = []
        targets = []

        grouped = df_pbp.groupby(['Game ID', 'Inning'])
        for _, group in grouped:
            seq = group[feature_cols].values
            target = group['Run_Value_RE24'].values
            sequences.append(seq)
            targets.append(target)

        N_samples = len(sequences)
        X_padded = np.full((N_samples, max_seq_len, len(feature_cols)), -99.0, dtype=np.float32)
        y_padded = np.full((N_samples, max_seq_len, 1), -99.0, dtype=np.float32)

        for i, (seq, trg) in enumerate(zip(sequences, targets)):
            length = min(len(seq), max_seq_len)
            X_padded[i, :length, :] = seq[:length]
            y_padded[i, :length, 0] = trg[:length]

        scaler = StandardScaler()
        X_flat = X_padded.reshape(-1, len(feature_cols))
        valid_mask = (X_flat[:, 0] != -99.0)
        scaler.fit(X_flat[valid_mask])

        X_scaled = X_padded.copy()
        for i in range(N_samples):
            for t in range(max_seq_len):
                if X_scaled[i, t, 0] != -99.0:
                    X_scaled[i, t, :] = scaler.transform(X_scaled[i, t, :].reshape(1, -1))[0]

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_padded, test_size=0.2, random_state=42)

        print("Building Sequential LSTM Model...")
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            model = keras.Sequential([
                layers.Masking(mask_value=-99.0, input_shape=(max_seq_len, len(feature_cols))),
                layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
                layers.TimeDistributed(layers.Dense(32, activation='relu')),
                layers.TimeDistributed(layers.Dense(1, activation='linear'))
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, callbacks=[early_stopping], verbose=0)
            print("TF/Keras LSTM Model Trained Successfully!")
        except Exception as e:
            print(f"TensorFlow/Keras not available ({e}). Using Scikit-Learn Sequential Feature Regressor fallback...")
            from sklearn.neural_network import MLPRegressor
            flat_X_tr = X_train.reshape(X_train.shape[0], -1)
            flat_y_tr = np.mean(y_train, axis=1).squeeze()
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
            model.fit(flat_X_tr, flat_y_tr)

        self.ml_models['lstm'] = {
            'model': model,
            'scaler': scaler,
            'features': feature_cols,
            'max_seq_len': max_seq_len
        }
        return self.ml_models['lstm']

    def optimize_lineup_lstm(self, players: List[PlayerModel]) -> LineupModel:
        """
        Optimize 9-hitter lineup using sequential LSTM predictions benchmarking against 2024 Yankees starting lineup.
        """
        target_p = players[:9]
        from itertools import permutations

        if 'lstm' not in self.ml_models:
            print("LSTM model not trained. Generating PBP dataset and training LSTM model...")
            pbp_df = self.game_generator.generate_pbp_dataset(target_p, n_games=200)
            self.train_lstm_model(pbp_df)

        lstm_info = self.ml_models['lstm']
        model = lstm_info['model']
        scaler = lstm_info['scaler']
        feature_cols = lstm_info['features']
        max_len = lstm_info.get('max_seq_len', 15)

        perm_list = list(permutations(target_p))
        best_cand = list(target_p)
        best_score = -np.inf

        print("Evaluating lineup sequence permutations via LSTM Model...")
        scores = []
        for perm in perm_list:
            # Build 9-batter sequence feature matrix
            seq_mat = np.zeros((1, max_len, len(feature_cols)), dtype=np.float32)
            prev_p = None
            for idx, p in enumerate(perm):
                if idx >= max_len: break
                row_dict = {
                    'batter_OBP': p.obp, 'batter_SLG': p.slg, 'batter_ISO': p.iso,
                    'batter_BB_rate': p.walk_rate, 'batter_contact_rate': p.contact_rate,
                    'batter_xwOBA': p.xwoba, 'batter_wOBA': p.woba, 'batter_xBA': p.xba,
                    'batter_HR_rate': p.hr_rate, 'batter_K_rate': p.strikeout_rate,
                    'prev_batter_wOBA': prev_p.woba if prev_p else 0.320,
                    'prev_batter_OBP': prev_p.obp if prev_p else 0.320,
                    'prev_batter_SLG': prev_p.slg if prev_p else 0.400,
                    'on_1b': 0, 'on_2b': 0, 'on_3b': 0, 'Pre-AB Outs': 0
                }
                vec = np.array([row_dict.get(c, 0.0) for c in feature_cols]).reshape(1, -1)
                vec_s = scaler.transform(vec)[0]
                seq_mat[0, idx, :] = vec_s
                prev_p = p

            if hasattr(model, 'predict'):
                pred = model.predict(seq_mat, verbose=0) if hasattr(model, 'compile') else model.predict(seq_mat.reshape(1, -1))
                score = float(np.sum(pred))
            else:
                score = 0.0

            scores.append(score)

        best_idx = int(np.argmax(scores))
        best_cand = list(perm_list[best_idx])
        print(f"LSTM Sequential Optimization Complete! Best Predicted Sequence Score: {scores[best_idx]:.4f}")

        res_lineup = LineupModel(best_cand, self.config_manager)
        res_lineup.set_lineup_order([p.name for p in best_cand])
        return res_lineup
    
    def _optimize_simulated_annealing(self, players: List[PlayerModel]) -> LineupModel:
        """Simulated annealing optimization"""
        # Initial lineup
        current_lineup = players[:9] if len(players) >= 9 else players + [players[0]] * (9 - len(players))
        np.random.shuffle(current_lineup)
        
        current_lineup_model = LineupModel(current_lineup, self.config_manager)
        
        # Evaluate initial lineup
        if self.ml_models:
            prediction = self.predict_lineup_performance(current_lineup_model)
            current_score = prediction['overall_prediction']
        else:
            evaluation = self.game_generator.evaluate_lineup(current_lineup_model, 100)
            current_score = evaluation['expected_runs']
        
        best_lineup = current_lineup.copy()
        best_score = current_score
        
        # Simulated annealing parameters
        initial_temp = 1.0
        final_temp = 0.01
        cooling_rate = 0.95
        max_iterations = 1000
        
        temp = initial_temp
        
        for iteration in range(max_iterations):
            # Generate neighbor (swap two random players)
            neighbor_lineup = current_lineup.copy()
            idx1, idx2 = np.random.choice(len(neighbor_lineup), 2, replace=False)
            neighbor_lineup[idx1], neighbor_lineup[idx2] = neighbor_lineup[idx2], neighbor_lineup[idx1]
            
            neighbor_lineup_model = LineupModel(neighbor_lineup, self.config_manager)
            
            # Evaluate neighbor
            if self.ml_models:
                prediction = self.predict_lineup_performance(neighbor_lineup_model)
                neighbor_score = prediction['overall_prediction']
            else:
                evaluation = self.game_generator.evaluate_lineup(neighbor_lineup_model, 100)
                neighbor_score = evaluation['expected_runs']
            
            # Accept or reject neighbor
            if neighbor_score > current_score or np.random.random() < np.exp((neighbor_score - current_score) / temp):
                current_lineup = neighbor_lineup
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_lineup = current_lineup.copy()
            
            # Cool down
            temp *= cooling_rate
            
            if temp < final_temp:
                break
        
        return LineupModel(best_lineup, self.config_manager)
    
    def plot_feature_importance(self, position: int = None, top_n: int = 10):
        """
        Plot feature importance for a specific position or all positions
        
        Args:
            position: Specific position to plot (1-9), or None for all positions
            top_n: Number of top features to show
        """
        if not self.feature_importance:
            print("No feature importance data available. Call analyze_position_importance first.")
            return
        
        if position is not None:
            if position not in self.feature_importance:
                print(f"No data available for position {position}")
                return
            
            importance_df = self.feature_importance[position]
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance for Position {position}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        
        else:
            # Plot all positions
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            fig.suptitle('Feature Importance by Lineup Position', fontsize=16)
            
            for pos in range(1, 10):
                row = (pos - 1) // 3
                col = (pos - 1) % 3
                
                if pos in self.feature_importance:
                    importance_df = self.feature_importance[pos]
                    top_features = importance_df.head(top_n)
                    
                    axes[row, col].barh(range(len(top_features)), top_features['importance'])
                    axes[row, col].set_yticks(range(len(top_features)))
                    axes[row, col].set_yticklabels(top_features['feature'], fontsize=8)
                    axes[row, col].set_xlabel('Importance')
                    axes[row, col].set_title(f'Position {pos}')
                    axes[row, col].invert_yaxis()
            
            plt.tight_layout()
            plt.show()
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis results"""
        summary = {
            'training_data_size': len(self.training_data) if self.training_data is not None else 0,
            'models_trained': len(self.ml_models),
            'feature_importance_available': len(self.feature_importance) > 0,
            'config': {
                'n_lineups_training': self.config_manager.ml.n_lineups_for_training,
                'n_simulations_per_lineup': self.config_manager.ml.n_simulations_per_lineup,
                'n_estimators': self.config_manager.ml.n_estimators,
                'max_depth': self.config_manager.ml.max_depth
            }
        }
        
        if self.ml_models:
            summary['model_performance'] = {}
            for pos, model_info in self.ml_models.items():
                summary['model_performance'][pos] = {
                    'r2': model_info.get('r2', 0),
                    'mse': model_info.get('mse', 0)
                }
        
        return summary
