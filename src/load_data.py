import pandas as pd


def load_reports(path):
   df = pd.read_csv(path)
   df = df.dropna(subset=["report"]).reset_index(drop=True)
   return df
