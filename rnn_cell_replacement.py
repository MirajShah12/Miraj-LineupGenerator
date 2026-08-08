# RNN MODEL FOR LINEUP OPTIMIZATION WITH GRADIENT IMPORTANCE AND POSITIONAL SENSITIVITY ANALYSIS
print("=== RNN LINEUP OPTIMIZATION ===")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prepare_rnn_data(df_lineups):
    """Prepare lineup data as sequences for RNN"""
    print("Preparing RNN sequence data...")
    
    feature_cols = ['wOBA', 'OBP', 'SLG', 'xwOBA', 'xBA', 'xSLG', 'ISO', 'BB_rate', 'K_rate', 'HR_rate', 'contact_rate']
    
    sequences = []
    targets = []
    unique_lineups = df_lineups['lineup_id'].unique()
    
    for lineup_id in unique_lineups:
        lineup_data = df_lineups[df_lineups['lineup_id'] == lineup_id].sort_values('position')
        sequence = []
        for pos in range(1, 10):
            pos_data = lineup_data[lineup_data['position'] == pos].iloc[0]
            features = [pos_data[col] for col in feature_cols]
            sequence.append(features)
        sequences.append(sequence)
        targets.append(lineup_data['avg_runs'].iloc[0])
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Created sequences: {X.shape}")
    return X, y, feature_cols

def create_rnn_model(sequence_length, n_features, rnn_units=128):
    """Create an RNN model to predict runs from lineup sequence"""
    model = keras.Sequential([
        layers.LSTM(rnn_units, return_sequences=True, input_shape=(sequence_length, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(rnn_units // 2, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(rnn_units // 4),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def get_gradient_importance(model, X_sample):
    """Calculate gradient-based feature importance"""
    print("Computing gradient-based importance...")
    
    X_tensor = tf.constant(X_sample, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor, training=False)
    
    gradients = tape.gradient(predictions, X_tensor)
    importance = np.mean(np.abs(gradients.numpy()), axis=0)
    
    sequence_length, n_features = importance.shape
    feature_names = ['wOBA', 'OBP', 'SLG', 'xwOBA', 'xBA', 'xSLG', 'ISO', 'BB_rate', 'K_rate', 'HR_rate', 'contact_rate']
    
    importance_dict = {}
    for pos in range(sequence_length):
        importance_dict[pos + 1] = {}
        for feat_idx, feat_name in enumerate(feature_names):
            importance_dict[pos + 1][feat_name] = importance[pos, feat_idx]
    
    return importance_dict

def positional_sensitivity_analysis(model, X_sample):
    """Perform positional sensitivity analysis"""
    print("Performing positional sensitivity analysis...")
    
    baseline_pred = model.predict(X_sample, verbose=0)
    sensitivity_dict = {}
    
    for pos in range(9):
        X_perturbed = X_sample.copy()
        perturbation = np.random.normal(0, 0.1, size=(X_sample.shape[1], X_sample.shape[2]))
        X_perturbed[:, pos, :] += perturbation
        perturbed_pred = model.predict(X_perturbed, verbose=0)
        sensitivity = np.mean(np.abs(perturbed_pred - baseline_pred))
        
        sensitivity_dict[pos + 1] = {
            'mean_change': sensitivity,
            'baseline_avg': np.mean(baseline_pred),
            'perturbed_avg': np.mean(perturbed_pred)
        }
    
    return sensitivity_dict

def train_rnn_model(X, y, validation_split=0.2, epochs=100, batch_size=32):
    """Train the RNN model"""
    print("Training RNN model...")
    
    n_samples, seq_len, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)
    
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=validation_split, random_state=42
    )
    
    model = create_rnn_model(seq_len, n_features)
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    test_loss, test_mae, test_mape = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test MAPE: {test_mape:.4f}%")
    
    return model, scaler, target_scaler, history, X_test, y_test

def plot_training_history(history):
    """Plot RNN training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution
print("Testing RNN approach with lineup data...")

try:
    if 'lineup_df_comprehensive' in locals() and lineup_df_comprehensive is not None:
        test_data = lineup_df_comprehensive
        print(f"Using comprehensive dataset: {len(test_data['lineup_id'].unique())} lineups")
    elif 'lineup_df_yankees' in locals() and lineup_df_yankees is not None:
        test_data = lineup_df_yankees
        print(f"Using Yankees dataset: {len(test_data['lineup_id'].unique())} lineups")
    else:
        print("No lineup data available. Please run previous cells.")
        test_data = None
    
    if test_data is not None:
        X, y, feature_names = prepare_rnn_data(test_data)
        model, scaler, target_scaler, history, X_test, y_test = train_rnn_model(X, y, epochs=100)
        plot_training_history(history)
        
        n_sample = min(100, len(X_test))
        sample_idx = np.random.choice(len(X_test), n_sample, replace=False)
        X_sample = X_test[sample_idx]
        
        importance_dict = get_gradient_importance(model, X_sample)
        
        print("\n=== GRADIENT-BASED IMPORTANCE ===")
        print("Top 3 most important features by position:")
        for pos in range(1, 4):
            sorted_features = sorted(importance_dict[pos].items(), key=lambda x: x[1], reverse=True)
            print(f"\nPosition {pos}:")
            for feat, importance in sorted_features[:3]:
                print(f"  {feat}: {importance:.6f}")
        
        sensitivity = positional_sensitivity_analysis(model, X_sample)
        
        print("\n=== POSITIONAL SENSITIVITY ANALYSIS ===")
        print("How sensitive is output to changes at each position:")
        for pos in range(1, 10):
            sens = sensitivity[pos]
            print(f"Position {pos}: Mean change = {sens['mean_change']:.4f}")
        
        print("\nRNN MODEL SUCCESSFULLY TRAINED!")
        print("Key features:")
        print("- Processes lineup as sequence to capture ordering")
        print("- LSTM layers learn long-range dependencies")
        print("- Gradient importance shows feature impact")
        print("- Positional sensitivity shows position impact")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

