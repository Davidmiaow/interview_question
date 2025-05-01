import pandas as pd

def load_data(beneficiary_path, outpatient_path):
    beneficiary_df = pd.read_csv(beneficiary_path)
    outpatient_df = pd.read_csv(outpatient_path)

    return beneficiary_df, outpatient_df

def combine(row, chronic_cols):
    conditions = [col.replace("SP_", "") for col in chronic_cols if row[col] == 1]
    if len(conditions) >= 3:
        return "Multiple"
    elif conditions:
        return "+".join(conditions)
    else:
        return "None"


def merage_chronic_conditions(beneficiary_df, chronic_cols):
    beneficiary_df["chronic_condition"] = beneficiary_df.apply(combine, axis=1)
    return beneficiary_df

def join_data(outpatient_df, beneficiary_df):
    merged_df = outpatient_df.merge(beneficiary_df[['DESYNPUF_ID', 'BENE_RACE_CD', 'chronic_condition']],
                                    on="DESYNPUF_ID", how="left")
    return merged_df

