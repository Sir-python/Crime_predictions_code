import pandas as pd
from pathlib import Path
import os

def load_data(file_path):
    # Load dataset from file path
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    
    file_extension = file.suffix.lower()
    if file_extension == ".csv":
        df = pd.read_csv(file)
    elif file_extension == ".xlsx":
        df = pd.read_excel(file)
    elif file_extension == ".json":
        df = pd.read_json(file)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only CSV, XLSX, and JSON are allowed.")

    print(f"Data loaded successfully from {file}")
    return df

def save_data(df, file_name, save_dir="../data/processed"):
    os.makedirs(save_dir, exist_ok=True)
    try:
        file_path = os.path.join(save_dir, file_name)
        df.to_csv(file_path, index=False)
    except:
        return "File save failed"
    else:
        return "File saved successfullly"