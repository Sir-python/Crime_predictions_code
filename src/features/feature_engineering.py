import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from visualizations.PCA_plot import plot_pca_variance
from data.load_data import *
import joblib
from sklearn.preprocessing import MinMaxScaler
from data.split_data import split_and_select_features

def spatial_aggregation(df):
    #    Spatial Aggregation:
    #    Group the data by the preserved raw Crime_Location to calculate aggregate measures.
    if 'Crime_Location_raw' in df.columns:
        spatial_agg = df.groupby('Crime_Location_raw').agg({
            'Age': 'mean',
            'crime_hour': 'mean',
            'crime_hour_sin': 'mean',
            'crime_hour_cos': 'mean',
            'PS/SD_encoded': 'first',         # Assuming PS/SD encoding is constant per location
            'Allegation_encoded': 'first',      # Likewise for Allegation encoding
            'crime_dayofweek': 'first'
        }).rename(columns={
            'Age': 'avg_age',
            'crime_hour': 'avg_crime_hour'
        }).reset_index()

        # Merge the aggregated spatial features back into the original DataFrame based on location
        df = df.merge(spatial_agg, left_on='Crime_Location_raw', right_on='Crime_Location_raw', 
                  how='left', suffixes=('', '_agg'))

    # Creating mapping dataframe for post clustering operations
    mapping_df = df[['Crime_Location_raw']].copy()
    mapping_df.index.name = 'index'
    mapping_df.to_csv(r"F:\University\Uni Stuff (semester 11)\Thesis\code\data\processed\crime_location_mapping.csv", index=True)  # Save with index

    df.drop(columns=['Crime_Location_raw'], inplace=True)
    return df

def transform_cyclical_features(df):
    # Encode cyclical features for day-of-week crime occurrences."""
    df['crime_dayofweek_sin'] = np.sin(2 * np.pi * df['crime_dayofweek'] / 7)
    df['crime_dayofweek_cos'] = np.cos(2 * np.pi * df['crime_dayofweek'] / 7)
    return df
    
def scale_features(df, numeric_features):
    # Standardizes numerical features using StandardScaler
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df, scaler

def choose_pca_components(df_scaled):
    # Determine optimal PCA components preserving 95% variance
    pca = PCA()
    pca.fit(df_scaled)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the number of components that explain at least 95% variance
    optimal_components = np.argmax(explained_variance >= 0.95) + 1
    print(f"Optimal number of components for 95% variance: {optimal_components}")

    # Plot the PCA explained variance
    plot_pca_variance(explained_variance)   

    return optimal_components

def create_interaction_features(df):
    # Generate new interaction features based on crime data
    df['hour_PS_SD_interaction'] = df['crime_hour'] * df['PS/SD_encoded']
    df['crime_proportion'] = df['Crime_Location'] / df['Crime_Location'].sum()
    return df

def apply_pca(df, numeric_features, optimal_components, scaler):
    # Apply PCA to reduce dimensionality and replace original features
    df_scaled = scaler.fit_transform(df[numeric_features])
    pca = PCA(n_components=optimal_components)
    df_pca = pca.fit_transform(df_scaled)

    # Convert PCA output into DataFrame
    pca_column_names = [f'PCA_{i+1}' for i in range(optimal_components)]
    df_pca = pd.DataFrame(df_pca, columns=pca_column_names, index=df.index)

    # Drop original numerical features and replace with PCA-transformed features
    df.drop(columns=numeric_features, inplace=True)
    df = pd.concat([df, df_pca], axis=1)

    return df

def FE_pipeline(df):
    df = spatial_aggregation(df)
    df = create_interaction_features(df)
    df = transform_cyclical_features(df)
    
    # Selecting Numeric Features
    numeric_features = df.select_dtypes(include=['float64', 'int64', 'bool']).columns.tolist()

    # Scaling Features
    df, scaler = scale_features(df, numeric_features)

    # Determine PCA Components
    optimal_components = choose_pca_components(df[numeric_features])

    # Apply PCA
    df = apply_pca(df, numeric_features, optimal_components, scaler)
    save_data(df, "FE_data_for_clustering.csv")

def generate_temporal_features(df):
    # Clean and format date/time strings
    # df['Crime_date'] = df['Crime_date'].astype(str).str.strip()
    # df['Crime_time'] = df['Crime_time'].astype(str).str.strip()

    # Extract datetime features
    df['Year'] = df['Crime_datetime'].dt.year
    df['Month'] = df['Crime_datetime'].dt.month
    df['Day'] = df['Crime_datetime'].dt.day
    df['Hour'] = df['Crime_datetime'].dt.hour
    df['DayOfWeek'] = df['Crime_datetime'].dt.dayofweek

    # Drop original columns
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)  # 1 if Sat/Sun, else 0
    df['Quarter'] = df['Crime_datetime'].dt.quarter  # Quarter of the year
    df['IsNightCrime'] = ((df['Hour'] >= 20) | (df['Hour'] <= 5)).astype(int)

    return df

def rolling_features(df):
    # Sort the DataFrame
    df = df.sort_values(by=['Crime_Location', 'Crime_datetime'])
    
    # Create a dummy column for counting
    df['_dummy'] = 1
    
    # Compute rolling counts directly on the 'Crime_datetime' column
    crime_count_7d = df.groupby('Crime_Location', group_keys=False).apply(
        lambda g: g.rolling('7D', on='Crime_datetime')['_dummy'].sum()
    )
    
    # Assign the computed rolling counts to new columns
    df['Crime_Count_Last_7_Days'] = crime_count_7d
    
    # Cleanup
    df.drop(['_dummy', 'Crime_date', 'Crime_time', 'Crime_datetime'], axis=1, inplace=True)
    df.fillna(0, inplace=True)
    return df

def hotspot_indicator(df):
    crime_hotspot_threshold = df['Crime_Location'].value_counts().quantile(0.75)  # Top 25% locations
    df['Is_Crime_Hotspot'] = df['Crime_Location'].map(
        lambda x: 1 if df['Crime_Location'].value_counts()[x] >= crime_hotspot_threshold else 0
    )

    df.fillna(0, inplace=True)
    return df

def data_normalization(df, scaler=None, save_path=r"F:\University\Uni Stuff (semester 11)\Thesis\code\data\processed\normalization_scaler.joblib"):
    numeric_cols = df.select_dtypes(include=['number']).columns  # Select numeric features only
    # cols = [col for col in df.columns if col not in ['Crime_Location', 'Allegation']]

    if scaler is None:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])  # Fit & transform training data
        if save_path:
            joblib.dump(scaler, save_path)  # Save the scaler for future use
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])  # Use the pre-trained scaler on new data

    joblib.dump(scaler,r"F:\University\Uni Stuff (semester 11)\Thesis\code\data\processed\MinMaxScaler.joblib")
    return df, scaler

def time_series_pipeline(df):
    temporal_df = generate_temporal_features(df)
    temporal_rolling_features_df = rolling_features(temporal_df)
    time_series_df = hotspot_indicator(temporal_rolling_features_df)
    # normalized_df, scaler = data_normalization(non_normalized_df)
    
    return time_series_df

def find_optimal_components(X, variance_threshold=0.95):
    # Finds the optimal number of PCA components to retain a specified variance threshold.
    
    # Parameters:
    #     X (np.ndarray or pd.DataFrame): Input features.
    #     variance_threshold (float): Desired cumulative explained variance (e.g., 0.95).
    
    # Returns:
    #     optimal_components (int): Number of components to retain.
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA to find explained variance
    pca = PCA().fit(X_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the number of components to retain the desired variance
    optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"Optimal number of components for 95% variance: {optimal_components}")

    # Plot the PCA explained variance
    plot_pca_variance(cumulative_variance)
    return optimal_components

def prepare_hybrid_model_data(df, target_columns, variance_threshold=0.95):
    # Prepares data for the hybrid model by finding optimal PCA components, splitting data,
    # and applying PCA to the training and testing sets.
    
    # Parameters:
    #     df (pd.DataFrame): Preprocessed DataFrame.
    #     target_columns (list): Names of target columns (e.g., ['Crime_Location', 'Allegation']).
    #     variance_threshold (float): Desired cumulative explained variance for PCA.
    
    # Returns:
    #     train_df (pd.DataFrame): Training data with PCA-transformed features and targets.
    #     test_df (pd.DataFrame): Testing data with PCA-transformed features and targets.
    #     pca (PCA): Fitted PCA model.

    # Step 1: Find optimal number of PCA components
    X = df.drop(columns=target_columns)
    optimal_components = find_optimal_components(X, variance_threshold)
    print(f"Optimal PCA Components: {optimal_components}")
    
    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_and_select_features(df, target_columns)
    
    # Step 3: Standardize and apply PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=optimal_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Step 4: Create DataFrames for PCA-transformed features
    pc_columns = [f"PC{i+1}" for i in range(optimal_components)]
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=pc_columns, index=X_train.index)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pc_columns, index=X_test.index)
    
    # Step 5: Combine with unscaled targets
    train_df = pd.concat([X_train_pca_df.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_pca_df.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    # Save scaler and PCA model
    joblib.dump(scaler, "standard_scaler.joblib")
    joblib.dump(pca, "pca_model.joblib")
    
    return train_df, test_df, pca