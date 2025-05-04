import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def split_and_select_features(data, target_cols, test_size=0.2, random_state=42):
    # Load dataset
    X = data.drop(target_cols, axis=1)
    y = data[target_cols]  # y is now a DataFrame with multiple target columns
    
    # Split into training/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Save the reduced datasets
    pd.DataFrame(X_train).to_csv(r"F:\University\Uni Stuff (semester 11)\Thesis\code\data\training\X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(r"F:\University\Uni Stuff (semester 11)\Thesis\code\data\testing\X_test.csv", index=False)
    y_train.to_csv(r"F:\University\Uni Stuff (semester 11)\Thesis\code\data\training\y_train.csv", index=False)
    y_test.to_csv(r"F:\University\Uni Stuff (semester 11)\Thesis\code\data\testing\y_test.csv", index=False)
    
    return X_train, X_test, y_train, y_test

def prepare_model_inputs_all_features(train_df, test_df, target_columns):
    # Splits the DataFrames into X and y and reshapes the features for the LSTM and MLP branches.
    
    # Parameters:
    #     train_df (pd.DataFrame): DataFrame containing training features and target(s).
    #     test_df (pd.DataFrame): DataFrame containing testing features and target(s).
    #     target_columns (list): List of column names corresponding to targets.
    #     lstm_feature_cols (list): List of column names to use for the LSTM branch.
    #     mlp_feature_cols (list): List of column names to use for the MLP branch.
        
    # Returns:
    #     X_train_lstm, X_train_mlp, y_train, X_test_lstm, X_test_mlp, y_test

    # Inversing scaling for target_columns
    scaled_training_values = train_df[['Crime_Location', 'Allegation']].values
    scaled_testing_values = test_df[['Crime_Location', 'Allegation']].values
    minmax_scaler = joblib.load(r"F:\University\Uni Stuff (semester 11)\Thesis\code\data\processed\MinMaxScaler.joblib")
    standard_scaler = joblib.load(r"F:\University\Uni Stuff (semester 11)\Thesis\code\data\processed\pre_pca_scaler.joblib")
    
    # Training targets
    scaled_train_targets = train_df[target_columns].values
    minmax_scaled_train = standard_scaler.inverse_transform(scaled_train_targets)
    original_train_targets = minmax_scaler.inverse_transform(minmax_scaled_train)
    y_train = np.round(original_train_targets).astype(int)  # Recover label-encoded integers
    
    # Testing targets
    scaled_test_targets = test_df[target_columns].values
    minmax_scaled_test = standard_scaler.inverse_transform(scaled_test_targets)
    original_test_targets = minmax_scaler.inverse_transform(minmax_scaled_test)
    y_test = np.round(original_test_targets).astype(int)
    
    # Step 2: Split features and targets
    X_train = train_df.drop(columns=target_columns)
    X_test = test_df.drop(columns=target_columns)
    
    # Step 3: Reshape for LSTM (3D) and MLP (2D)
    # LSTM shape: (samples, timesteps=1, features)
    X_train_lstm = X_train.values.reshape(-1, 1, X_train.shape[1])
    X_test_lstm = X_test.values.reshape(-1, 1, X_test.shape[1])
    
    # MLP shape: (samples, features)
    X_train_mlp = X_train.values
    X_test_mlp = X_test.values
    
    return X_train_lstm, X_train_mlp, y_train, X_test_lstm, X_test_mlp, y_test
