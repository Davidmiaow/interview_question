from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.data_download import download_file
from src.data_preprocessing import load_data, merage_chronic_conditions, join_data
from src.exploratory_analysis import race_distribution, top_chronic_conditions
from src.provider_segmentation import kmeans_segmentation, isolation_forest, dbscan_outlier_detection

app = FastAPI(title="Medicare Chronic Conditions API", version="1.0")

class SegmentationRequest(BaseModel):
    feature_cols:list
    n_cluster: int=3

class AnomalyDetectionRequest(BaseModel):
    feature_cols: list

# Loading data:
beneficiary_path = "C:\Users\david\OneDrive\Desktop\medicare_chronic_conditions_cost_analysis_network_optimization\data\raw\DE1_0_2009_Beneficiary_Summary_File_Sample_20.csv"
outpatient_path = "C:\Users\david\OneDrive\Desktop\medicare_chronic_conditions_cost_analysis_network_optimization\data\raw\DE1_0_2008_to_2010_Outpatient_Claims_Sample_20.csv"
beneficiary_df, outpatient_df = load_data(beneficiary_path, outpatient_path)

chronic_columns = ["SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR",
                   "SP_COPD", "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT",
                   "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA"]

beneficiary_df = merage_chronic_conditions(beneficiary_df, chronic_columns)
merged_df = join_data(outpatient_df, beneficiary_df)

@app.get("/download_beneficiary_data")
async def download_beneficiary():
    beneficiary_url = "https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_2009_Beneficiary_Summary_File_Sample_20.zip"
    return {"message:": download_file(beneficiary_url,"data/raw/beneficiary_data.zip")}

@app.get("/download_patient_data")
async def download_outpatient_records():
    outpatient_url = "https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_2008_to_2010_Outpatient_Claims_Sample_20.zip"
    return {"message:": download_file(outpatient_url,"data/raw/outpatient_data.zip")}


@app.get("/race-distribution")
async def get_race_distribution():
    return race_distribution(beneficiary_df)

@app.get("/top—chronic-conditions")
async def get_top_chronic_conditions():
    return top_chronic_conditions(beneficiary_df)

@app.get("/kmeans_segmentation")
async def kmeans_segment(req: SegmentationRequest):
    segmented, _  = kmeans_segmentation(merged_df, req.feature_cols, req.n_cluster)
    return segmented.to_dict()

@app.get("/isolation_forest")
async def detect_isolation_forest(req: AnomalyDetectionRequest):
    anomalies, _ = isolation_forest(merged_df, req.feature_cols)
    return anomalies.to_dict()

@app.get("/dbscan_outliers")
async def detect_outliers(req: AnomalyDetectionRequest):
    outliers, _ = dbscan_outlier_detection(merged_df, req.feature_cols)
    return outliers

