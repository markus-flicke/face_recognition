import os

import pandas as pd
from src import Config

def load_A2():
    df = pd.read_csv(os.path.join(Config.ANDREAS_ALBUMS_PATH, 'labels.csv'))
    return [os.path.join(Config.EXTRACTED_FACES_PATH, filename) for filename in df.values[:,0]], list(df.values[:,1])


if __name__=='__main__':
    print(load_A2())