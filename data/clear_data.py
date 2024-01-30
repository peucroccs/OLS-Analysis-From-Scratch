import pandas as pd

WineQualityDf = pd.read_csv("winequality-red.csv", sep=";")
dups = list(WineQualityDf.duplicated())

indexes = [i for i, x in enumerate(dups) if x]
WineQualityDf = WineQualityDf.drop(index=indexes)

WineQualityDf.to_csv("WineQualityDf_cleaned.csv")