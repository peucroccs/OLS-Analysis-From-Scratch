import pandas as pd

cancer_df = pd.read_csv(r"C:\Users\pedro\Documents\WineQualityPredictor\data\cancer_reg.csv", encoding='latin-1')
cancer_df = cancer_df.dropna()

cancer_df.to_csv("cancer_reg_cleaned.csv")