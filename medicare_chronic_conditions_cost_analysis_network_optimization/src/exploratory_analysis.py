import pandas as pd

def race_distribution(df):
    race_mapping = {0:'Unknown', 1:'White', 2:'Black', 3:'Other', 4:'Asian', 5:'Hispanic', 6:'North American Native'}
    df["race_label"] = df["BENE_RACE_CD"].map(race_mapping)
    return df["race_label"].value_counts(normalize=True).to_dict()

def top_chronic_conditions(df, k=10):
    filtered_df = df[df["chronic_condition"] != "None"]
    return filtered_df["chronic_condition"].value_counts().head(k).to_dict()
