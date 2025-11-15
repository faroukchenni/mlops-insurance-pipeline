# model_pipeline.py

"""
Pipeline ML basé sur ton notebook CRISP_DM_Insurance_Prediction :

Étapes principales :
1) Chargement + imputation des valeurs manquantes
2) Encodage des variables catégorielles (get_dummies, drop_first=True)
3) Feature engineering : smoker_age, bmi_children
4) Régression RandomForest pour prédire les charges
5) Clustering K-Means sur les clients (optionnel)
6) Sauvegarde / chargement du modèle de régression avec joblib
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Colonnes utilisées dans ton notebook
NUM_COLS = ["age", "bmi", "children", "charges"]
CAT_COLS = ["sex", "smoker", "region"]
TARGET_COL = "charges"


def _impute_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputer les valeurs manquantes comme dans le notebook :
    - médiane pour les numériques
    - mode (most_frequent) pour les catégorielles
    """
    df = df.copy()

    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    if num_cols:
        imputer_num = SimpleImputer(strategy="median")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    if cat_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    return df


def _add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features d'interaction comme dans ta partie clustering :
    - smoker_age = f(smoker, age)
    - bmi_children = bmi * children
    """
    df = df.copy()

    if {"smoker", "age"}.issubset(df.columns):
        df["smoker_age"] = df["smoker"].map({"yes": 1, "no": 0}).fillna(0) * df["age"]

    if {"bmi", "children"}.issubset(df.columns):
        df["bmi_children"] = df["bmi"] * df["children"]

    return df


def prepare_data(csv_path: str):
    """
    Préparation des données pour la régression (charges) en suivant ton notebook :

    - Charge dataAssurance.csv
    - Impute num + cat
    - Ajoute smoker_age, bmi_children
    - One-hot encode (drop_first=True)
    - Split train/test

    Retourne :
        X_train, X_test, y_train, y_test, df_encoded
    """
    df = pd.read_csv(csv_path)

    # Imputation
    df = _impute_basic(df)

    # Feature engineering
    df = _add_feature_engineering(df)

    # Encodage one-hot des catégorielles
    df_encoded = pd.get_dummies(df, drop_first=True)

    if TARGET_COL not in df_encoded.columns:
        raise ValueError(f"Colonne cible '{TARGET_COL}' introuvable après encodage.")

    X = df_encoded.drop(TARGET_COL, axis=1)
    y = df_encoded[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, df_encoded


def train_model(X_train, y_train):
    """
    Entraîne un RandomForestRegressor (comme dans ton comparatif LinearRegression / RandomForest).
    """
    model = RandomForestRegressor(random_state=42, n_estimators=200)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Calcule les métriques train/test (MAE, RMSE, R2),
    comme ta fonction score_model dans le notebook.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "Train_MAE": mean_absolute_error(y_train, y_pred_train),
        "Test_MAE": mean_absolute_error(y_test, y_pred_test),
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "Train_R2": r2_score(y_train, y_pred_train),
        "Test_R2": r2_score(y_test, y_pred_test),
    }
    return metrics


def save_model(model, path: str = "insurance_model.joblib"):
    """
    Sauvegarde le modèle de régression entraîné.
    """
    joblib.dump(model, path)


def load_model(path: str = "insurance_model.joblib"):
    """
    Charge un modèle de régression sauvegardé.
    """
    return joblib.load(path)


# ===================== CLUSTERING (optionnel) =====================

def prepare_clustering_data(csv_path: str):
    """
    Préparation des données pour le clustering, en suivant ta logique :

    - Charge & impute le DF
    - Drop 'charges' (on cluster sur le profil client, pas sur le coût)
    - Ajoute smoker_age, bmi_children
    - One-hot encode
    - Scale avec StandardScaler

    Retourne :
        X_scaled, scaler, feature_names, df_features
    """
    df = pd.read_csv(csv_path)
    df = _impute_basic(df)

    # On clusterise sans 'charges'
    if TARGET_COL in df.columns:
        base = df.drop(columns=[TARGET_COL])
    else:
        base = df.copy()

    base = _add_feature_engineering(base)

    df_encoded = pd.get_dummies(base, drop_first=True)

    feature_names = df_encoded.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    return X_scaled, scaler, feature_names, df_encoded


def run_kmeans_clustering(X_scaled, best_k: int = 3):
    """
    Exécute KMeans avec k clusters (par défaut 3 comme dans ton exemple) et retourne :
        labels, modèle KMeans, silhouette_score
    """
    km = KMeans(n_clusters=best_k, random_state=42)
    labels = km.fit_predict(X_scaled)

    # Silhouette score, si le cas est valide
    if X_scaled.shape[0] > best_k:
        score = silhouette_score(X_scaled, labels)
    else:
        score = float("nan")

    return labels, km, score

