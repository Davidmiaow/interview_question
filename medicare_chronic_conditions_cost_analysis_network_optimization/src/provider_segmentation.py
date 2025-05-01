from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from scipy.stats import zscore

def kmeans_segmentation(df, feature_cols, n_cluster=3):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[feature_cols])
    return df[["AT_PHYSN_NPI", "cluster"]], kmeans

def isolation_forest(df, feature_cols):
    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly_score"] = iso.fit_predict(df[feature_cols])
    return df[["AT_PHYSN_NPI", "anomaly_score"]]

def dbscan_outlier_detection(df, feature_cols, eps=0.5, min_samples=5):
    # Todo: change the min samples to optimize
    db = DBSCAN(eps=eps, min_samples=min_samples)
    df["DBSCAN_label"] = db.fit_predict(df[feature_cols])
    return df[["AT_PHYSN_NPI", "DBSCAN_label"]], db

def z_score_detection(series, threshold=3):
    # TODO: optimization point
    z_scores = zscore(series)
    return series[np.abs(z_scores) > threshold]


