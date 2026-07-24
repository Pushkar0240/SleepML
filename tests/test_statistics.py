import pandas as pd

from src.training.dataset_statistics import DatasetStatistics

df = pd.read_csv("data/dataset/metadata.csv")

stats = DatasetStatistics()

stats.load(df)

stats.report()

stats.export()
