import numpy as np
import pandas as pd
import wget
from tqdm import tqdm

dataset = "https://neurovault.org/media/images/503/"
csv_file = pd.read_csv("../S1_Data.csv")

with tqdm(total = len(csv_file)) as pbar:
    pbar.set_description("[Download PINES Dataset]")
    for filename in csv_file.Files:
        wget.download(dataset+filename)
        pbar.update(1)
