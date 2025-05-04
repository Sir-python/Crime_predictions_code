import pandas as pd
import numpy as np
from load_data import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data_clustering(df):
    # 1. Remove duplicates and drop irrelevant columns
    df = df.drop_duplicates()
    df = df.drop(columns=["SL"])
    
    # 2. Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 3. Handle missing values:
    df = df.dropna(subset=categorical_cols)
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # 4. Process date and time:
    #    - Extract the hour from Crime_time and create cyclical features.
    #    - Extract year, month, day and day-of-week from Crime_date.
    df['Crime_date'] = pd.to_datetime(df['Crime_date'])
    df['Crime_time'] = pd.to_datetime(df['Crime_time'])
    df['crime_hour'] = pd.to_datetime(df["Crime_time"], format='%H:%M:%S', errors='coerce').dt.hour
    df['Crime_year'] = df['Crime_date'].dt.year
    df['Crime_month'] = df['Crime_date'].dt.month
    df['Crime_day'] = df['Crime_date'].dt.day
    df['crime_dayofweek'] = pd.to_datetime(df["Crime_date"], errors='coerce').dt.dayofweek  # Monday=0, Sunday=6

    # Cyclical transformation for hour
    df['crime_hour_sin'] = np.sin(2 * np.pi * df['crime_hour'] / 24)
    df['crime_hour_cos'] = np.cos(2 * np.pi * df['crime_hour'] / 24)
    
    #Checking data types
    print(df.dtypes)

    # Drop original date/time columns (now that features have been extracted)
    df = df.drop(columns=['Crime_date', 'Crime_time'])
    
    # 5. Preserve the original Crime_Location for later spatial aggregation
    df['Crime_Location_raw'] = df['Crime_Location']
    
    # 6. Encode categorical features:
    #    One-hot encode for gender (assumed to have a low number of categories)
    df = pd.get_dummies(df, columns=['gender'])
    
    # Frequency encode location, PS/SD, and Allegation (these features help represent crime density)
    df['PS/SD_encoded'] = df.groupby('PS/SD')['PS/SD'].transform('count')
    df['Crime_Location'] = df.groupby('Crime_Location')['Crime_Location'].transform('count')
    df['Allegation_encoded'] = df.groupby('Allegation')['Allegation'].transform('count')
    df = df.drop(columns=['PS/SD', 'Allegation'])
    
    # 7. Scale only the non-temporal numerical features so that the temporal features remain in raw form.
    numerical_to_scale = ['Age', 'PS/SD_encoded', 'Allegation_encoded']
    scaler = StandardScaler()
    df[numerical_to_scale] = scaler.fit_transform(df[numerical_to_scale])
    
    return df

def hybrid_model_preprocessing(df, encoders_filepath=r"F:\University\Uni Stuff (semester 11)\Thesis\code\data\processed\label_encoders.pkl"):
    # Drop duplicates and identifier column
    df['Crime_datetime'] = pd.to_datetime(
        df['Crime_date'] + ' ' + df['Crime_time'],
        format='%d/%m/%Y %I:%M:%S %p',  # Matches "mm/dd/YYYY HH:MM:SS AM/PM"
        errors='coerce'
    )
    df = df.sort_values(by='Crime_datetime')
    df = df.drop(columns=["SL"])

    # Handle missing values (unchanged)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['gender'] = df['gender'].fillna('Unknown')
    df['PS/SD'] = df['PS/SD'].fillna('Unknown')
    df['Crime_Location'] = df['Crime_Location'].fillna('Unknown')
    df['Allegation'] = df['Allegation'].fillna('Unknown')

    # Label encoding (unchanged)
    label_encoders = {}
    categorical_cols = ['gender', 'PS/SD', 'Crime_Location', 'Allegation']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Save the label encoders to a file for later use
    joblib.dump(label_encoders, encoders_filepath)
    print(f"Label encoders saved to {encoders_filepath}")

    return df, label_encoders

def prepare_data(train_df, test_df, target_columns=['Crime_Location', 'Allegation']):

    # Split features and targets
    X_train = train_df.drop(target_columns, axis=1)
    y_train = train_df[target_columns]

    X_test = test_df.drop(target_columns, axis=1)
    y_test = test_df[target_columns]

    # Handle LSTM data (assuming you have a time-series component)
    timesteps = 10  # Example timesteps
    features = X_train.shape[1]  # Number of features
    X_train_lstm = np.array([X_train.iloc[i:i + timesteps].values for i in range(len(X_train) - timesteps + 1)])
    X_test_lstm = np.array([X_test.iloc[i:i + timesteps].values for i in range(len(X_test) - timesteps + 1)])

    # Handle MLP data (static/spatial features)
    X_train_mlp = X_train.iloc[timesteps - 1:].values  # Align MLP data with LSTM
    X_test_mlp = X_test.iloc[timesteps - 1:].values

    # Scale MLP data
    scaler = StandardScaler()
    X_train_mlp = scaler.fit_transform(X_train_mlp)
    X_test_mlp = scaler.transform(X_test_mlp)

    # Prepare target data
    y_train_location = y_train['Crime_Location'].iloc[timesteps - 1:].values
    y_test_location = y_test['Crime_Location'].iloc[timesteps - 1:].values

    y_train_crime = y_train['Allegation'].iloc[timesteps - 1:].values
    y_test_crime = y_test['Allegation'].iloc[timesteps - 1:].values

    return (X_train_lstm, X_train_mlp, y_train_location, y_train_crime,
            X_test_lstm, X_test_mlp, y_test_location, y_test_crime)