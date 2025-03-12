import os
import requests

from joblib import dump, load
import pandas as pd


def ingest_data():
    csv = requests.get('https://storage.googleapis.com/ubiops/data/Deploying%20with%20popular%20DS%20libraries/sci-kit-deployment/diabetes.csv')
    return csv.content

def write_to_csv(file_name, content):
    with open(file_name, "wb") as f:
        f.write(content)

def load_dataframe(file_name, **kwargs):
    return pd.read_csv(file_name, **kwargs)

def write_df_to_file(df, file_name, index=False, header=True, **kwargs):
    # UbiOps expects JSON serializable output or files, so we convert the dataframes to csv
    df.to_csv(file_name, index = index, header = header, **kwargs)

def write_model_to_file(file_name, model, **kwargs):
    # Persisting the model for use in UbiOps
    with open(file_name, 'wb') as f:
        dump(model, file_name, **kwargs)

def load_model_from_file(base_directory, file_name):
    model = os.path.join(base_directory, file_name)
    return load(model)