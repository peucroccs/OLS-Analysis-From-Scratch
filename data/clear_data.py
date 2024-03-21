import pandas as pd

cancer_df = pd.read_csv(r"C:\Users\pedro\Documents\WineQualityPredictor\data\cancer_reg.csv", encoding='latin-1')
dups = list(cancer_df.duplicated())

indexes = [i for i, x in enumerate(dups) if x]
cancer_df = cancer_df.drop(index=indexes)

cancer_df.to_csv("cancer_reg_cleaned.csv")