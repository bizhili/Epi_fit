import pandas as pd

def read_csv(path= ""):
    data= {}
    df = pd.read_csv(path)
    headers_list = df.columns.tolist()
    numpy_data = df.values
    for i, head in enumerate(headers_list):
        data[head]= numpy_data[:, i]
    return data