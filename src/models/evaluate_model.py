# Pre-requisites
from visualizations.plot_results import hotspots_per_cluster_plot
import pandas as pd
import joblib
from data.load_data import *
from feature_engineering import choose_pca_components, apply_pca
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical # type: ignore
from visualizations.LSTM_MLP_results import plot_diagnostics

# K-means cluster model evaluation
def cluster_summary(df):
    # Crime hotspots per cluster
    crime_hotspots = df.groupby(['cluster', 'Crime_Location_raw']).size().reset_index(name='crime_count')
    crime_hotspots = crime_hotspots.sort_values(by='crime_count', ascending=False)
    
    # Group by the cluster label using crime_hotspots (not df) to compute total and average crime counts
    hotspot_summary = crime_hotspots.groupby('cluster').agg({
        'crime_count': ['sum', 'mean']
    }).reset_index()

    hotspot_summary.columns = ['cluster', 'total_crime_count', 'avg_crime_count']
    print(hotspot_summary.sort_values(by='total_crime_count', ascending=False))
    
    return crime_hotspots

def hybrid_model_summary(model, X_test_lstm, X_test_mlp, y_test):
    results = model.evaluate([X_test_lstm, X_test_mlp], y_test)
    print("Evaluation Results:", results)

# def evaluate_model(model, X_test_lstm, X_test_mlp, y_test_location, y_test_crime, 
#                    class_names_location, class_names_crime, history=None, save_path=None):
#     # Evaluates model performance and generates plots.
    
#     # Args:
#     #     model: Trained Keras model
#     #     X_test_lstm: LSTM test features
#     #     X_test_mlp: MLP test features
#     #     y_test_location: True location labels
#     #     y_test_crime: True crime labels
#     #     class_names_location: List of location class names
#     #     class_names_crime: List of crime class names
#     #     history: Training history object (optional)

#     # Generate predictions
#     location_probs, crime_probs = model.predict([X_test_lstm, X_test_mlp])
#     y_pred_location = np.argmax(location_probs, axis=1)
#     y_pred_crime = np.argmax(crime_probs, axis=1)

#     # 1. Classification Reports
#     print("\n" + "="*40 + " Location Report " + "="*40)
#     print(classification_report(
#         y_test_location,
#         y_pred_location,
#         labels=np.arange(len(class_names_location)),
#         target_names=class_names_location))
    
#     print("\n" + "="*40 + " Crime Report " + "="*40)
#     print(classification_report(
#         y_test_crime,
#         y_pred_crime,
#         labels=np.arange(len(class_names_crime)),
#         target_names=class_names_crime))

#     # 2. AUC-ROC Scores
#     y_test_location_onehot = to_categorical(y_test_location, num_classes=len(class_names_location))
#     y_test_crime_onehot = to_categorical(y_test_crime, num_classes=len(class_names_crime))
    
#     # location_auc = roc_auc_score(y_test_location_onehot, location_probs, multi_class='ovr')
#     # crime_auc = roc_auc_score(y_test_crime_onehot, crime_probs, multi_class='ovr')
#     # For location output, only compute ROC AUC if more than one class is present
#     if np.unique(y_test_location).size > 1:
#         location_auc = roc_auc_score(y_test_location_onehot, location_probs, multi_class='ovr')
#         print(f"\nLocation AUC: {location_auc:.4f}")
#     else:
#         location_auc = None
#         print("\nLocation AUC: Cannot be computed as only one class is present.")
        
#     # Do the same for the crime output
#     if np.unique(y_test_crime).size > 1:
#         crime_auc = roc_auc_score(y_test_crime_onehot, crime_probs, multi_class='ovr')
#         print(f"Crime AUC: {crime_auc:.4f}")
#     else:
#         crime_auc = None
#         print("Crime AUC: Cannot be computed as only one class is present.")

#     print(f"\nLocation AUC: {location_auc:.4f}")
#     print(f"Crime AUC: {crime_auc:.4f}")

#     # Compile metrics into a dictionary
#     metrics = {
#         'location_report': classification_report(y_test_location, y_pred_location, output_dict=True),
#         'crime_report': classification_report(y_test_crime, y_pred_crime, output_dict=True),
#         'location_auc': location_auc,
#         'crime_auc': crime_auc
#     }

#     if save_path:
#         model.save(save_path)

#     # 3. Call plotting function
#     plot_diagnostics(
#         y_test_location, y_pred_location, location_probs,
#         y_test_crime, y_pred_crime, crime_probs,
#         class_names_location, class_names_crime,
#         history
#     )

#     return metrics

def evaluate_model(model, X_test_lstm, X_test_mlp, y_test_location, 
                   y_test_crime, class_names_location, class_names_crime, history=None, save_path=None):
    # Evaluates model performance using classification reports.

    location_probs, crime_probs = model.predict([X_test_lstm, X_test_mlp])
    y_pred_location = np.argmax(location_probs, axis=1)
    y_pred_crime = np.argmax(crime_probs, axis=1)

    # Classification Reports
    print("\n" + "="*40 + " Location Report " + "="*40)
    print(classification_report(y_test_location, y_pred_location, 
                                labels=np.arange(len(class_names_location)), 
                                target_names=class_names_location))

    print("\n" + "="*40 + " Crime Report " + "="*40)
    print(classification_report(y_test_crime, y_pred_crime, 
                                labels=np.arange(len(class_names_crime)), 
                                target_names=class_names_crime))

    # Compile metrics into a dictionary
    metrics = {
        'location_report': classification_report(y_test_location, y_pred_location, output_dict=True),
        'crime_report': classification_report(y_test_crime, y_pred_crime, output_dict=True),
    }

    location_balanced_accuracy = balanced_accuracy_score(y_test_location, y_pred_location)
    crime_balanced_accuracy = balanced_accuracy_score(y_test_crime, y_pred_crime)

    print(f"\nLocation Balanced Accuracy: {location_balanced_accuracy:.4f}")
    print(f"Crime Balanced Accuracy: {crime_balanced_accuracy:.4f}")

    # Add to metrics dictionary:
    metrics['location_balanced_accuracy'] = location_balanced_accuracy
    metrics['crime_balanced_accuracy'] = crime_balanced_accuracy
    if save_path:
        model.save(save_path)

    # 3. Call plotting function
    plot_diagnostics(y_test_location, y_pred_location, location_probs, 
                     y_test_crime, y_pred_crime, crime_probs, 
                     class_names_location, class_names_crime, history)
    #Remove trend prediction to the plot_diagnostics function.

    return metrics