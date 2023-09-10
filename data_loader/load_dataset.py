import pandas as pd
from .data_preprocessing import preprocess

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = preprocess(data)
    return (data['review'].to_list(), data['sentiment'].to_list())