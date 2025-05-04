from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from visualizations.plot_results import pairplot_k_means
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from data.process_data import prepare_data
from tensorflow.keras.regularizers import l2 # type: ignore
from models.evaluate_model import evaluate_model
from sklearn.utils.class_weight import compute_class_weight

def k_means_model(df, k=4):
    # Build and fit the final K-Means model with k=4
    optimal_k = k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df)

    # Assign cluster labels to your DataFrame
    df['cluster'] = cluster_labels

    # Step 3: Evaluate the clusters
    print("Cluster centers (in scaled space):")
    print(kmeans.cluster_centers_)

    sil_score = silhouette_score(df, cluster_labels)
    print(f"Silhouette Score: {sil_score:.2f}")

    return df

def build_hybrid_model(lstm_input_shape, mlp_input_shape, location_output_dim=9, crime_output_dim=31):
    # Builds a hybrid model to predict crime locations and allegations.
    # LSTM Branch (temporal features)
    lstm_input = Input(shape=lstm_input_shape, name='lstm_input')
    x = LSTM(30, return_sequences=True)(lstm_input)
    x = LSTM(32)(x)
    x = Dropout(0.5)(x)

    # MLP Branch (static/spatial features)
    mlp_input = Input(shape=mlp_input_shape, name='mlp_input')
    y = Dense(64, activation='relu')(mlp_input)
    y = Dense(32, activation='relu')(y)
    y = Dense(30, activation='relu')(y)
    y = Dropout(0.3)(y)

    # Concatenate branches
    combined = Concatenate()([x, y])
    combined = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(combined)

    # Output 1: Crime Location Prediction (10 classes)
    location_output = Dense(location_output_dim, activation='softmax', name='location_output')(combined)

    # Output 2: Allegation Prediction (31 classes)
    crime_output = Dense(crime_output_dim, activation='softmax', name='crime_output')(combined)

    # Define the model
    model = Model(inputs=[lstm_input, mlp_input], outputs=[location_output, crime_output])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss={
            'location_output': 'sparse_categorical_crossentropy',
            'crime_output': 'sparse_categorical_crossentropy',
        },
        metrics={
            'location_output': [tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_accuracy')],
            'crime_output': [tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_accuracy')],
        }
    )
    return model

def hybrid_model_pipeline(train_df, test_df, target_columns=['Crime_Location', 'Allegation']):
    # Prepare inputs using the new prepare_data function.
    (X_train_lstm, X_train_mlp, y_train_location, y_train_crime,
     X_test_lstm, X_test_mlp, y_test_location, y_test_crime) = prepare_data(train_df, test_df, target_columns)

    # Verify the shapes of the inputs and targets.
    print(f"LSTM Input Shape: {X_train_lstm.shape}")
    print(f"MLP Input Shape: {X_train_mlp.shape}")
    print(f"Target Shapes: {y_train_location.shape}, {y_train_crime.shape}")

    # Define input shapes for the model.
    lstm_input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
    mlp_input_shape = (X_train_mlp.shape[1],)

    # Determine the number of classes for each target.
    num_location_classes = len(np.unique(y_train_location))
    num_crime_classes = len(np.unique(y_train_crime))
    print(f"Number of location classes: {num_location_classes}")
    print(f"Number of crime classes: {num_crime_classes}")

    # Build the hybrid model.
    model = build_hybrid_model(lstm_input_shape, mlp_input_shape)
    model.summary()

    # Calculate class weights for location and crime
    location_class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_location),
        y=y_train_location
    )
    crime_class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_crime),
        y=y_train_crime
    )

    # Create class weight dictionaries
    location_class_weight_dict = dict(enumerate(location_class_weights))
    crime_class_weight_dict = dict(enumerate(crime_class_weights))

    # Train the model with class weights for both outputs
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # print(f"Value of location_class_weight_dict: {location_class_weight_dict}\n")
    # print(f"Value of crime_class_weight_dict: {crime_class_weight_dict}\n")

    # print(f"Keys of location_class_weight_dict: {location_class_weight_dict.keys}\n")
    # print(f"Keys of crime_class_weight_dict: {crime_class_weight_dict.keys}\n")
    # print(y_train_location.dtype)
    # print(y_train_crime.dtype)
    print("Model output names:", model.output_names)
    history = model.fit(
        [X_train_lstm, X_train_mlp],
        {'location_output': y_train_location, 'crime_output': y_train_crime},
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Evaluate the model and summarize its performance.
    evaluate_model(
        model,
        X_test_lstm,
        X_test_mlp,
        y_test_location,
        y_test_crime,
        class_names_location=["Zone {}".format(i) for i in range(10)],
        class_names_crime=["Crime {}".format(i) for i in range(31)],
        history=history,
        save_path=r"F:\University\Uni Stuff (semester 11)\Thesis\code\src\models\hybrid_crime_prediction_model.keras"
    )