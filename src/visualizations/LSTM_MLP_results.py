import seaborn as sns
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix
)
from tensorflow.keras.utils import to_categorical # type: ignore

# Plot training & validation loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()

def plot_diagnostics(y_true_loc, y_pred_loc, loc_probs,
                     y_true_crime, y_pred_crime, crime_probs,
                     class_names_loc, class_names_crime,
                     history=None):

    # 1. Training History (if available)
    if history is not None:
        # Accuracy Plot
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['location_output_sparse_accuracy'], label='Location Train')
        plt.plot(history.history['val_location_output_sparse_accuracy'], label='Location Val')
        plt.plot(history.history['crime_output_sparse_accuracy'], label='Crime Train')
        plt.plot(history.history['val_crime_output_sparse_accuracy'], label='Crime Val')
        plt.title('Accuracy History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Loss Plot
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.title('Loss History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 2. Confusion Matrices (using matplotlib directly)
    # Location Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm_loc = confusion_matrix(y_true_loc, y_pred_loc)
    plt.imshow(cm_loc, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Location Confusion Matrix')
    plt.colorbar()
    tick_marks_loc = np.arange(len(class_names_loc))
    plt.xticks(tick_marks_loc, class_names_loc, rotation=45, ha='right', fontsize=8)
    plt.yticks(tick_marks_loc, class_names_loc, fontsize=8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add annotations
    thresh_loc = cm_loc.max() / 2.
    for i, j in np.ndindex(cm_loc.shape):
        plt.text(j, i, format(cm_loc[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm_loc[i, j] > thresh_loc else "black", fontsize=6)

    plt.tight_layout()
    plt.show()

    # Crime Confusion Matrix
    plt.figure(figsize=(20, 18))
    cm_crime = confusion_matrix(y_true_crime, y_pred_crime)
    plt.imshow(cm_crime, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title('Crime Confusion Matrix')
    plt.colorbar()
    tick_marks_crime = np.arange(len(class_names_crime))
    plt.xticks(tick_marks_crime, class_names_crime, rotation=45, ha='right', fontsize=6)
    plt.yticks(tick_marks_crime, class_names_crime, fontsize=6)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add annotations
    thresh_crime = cm_crime.max() / 2.
    for i, j in np.ndindex(cm_crime.shape):
        plt.text(j, i, format(cm_crime[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm_crime[i, j] > thresh_crime else "black", fontsize=4)

    plt.tight_layout()
    plt.show()