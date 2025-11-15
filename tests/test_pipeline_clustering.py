from model_pipeline import prepare_clustering_data, run_kmeans_clustering
import numpy as np

def test_clustering_execution():
    X_scaled, scaler, feature_names, df_features = prepare_clustering_data("dataAssurance.csv")

    # Vérifier que les features sont bien scalées
    assert X_scaled.shape[0] > 0
    assert X_scaled.shape[1] > 0

    labels, model, score = run_kmeans_clustering(X_scaled, best_k=3)

    # Vérifier que 3 clusters ont bien été créés
    assert len(np.unique(labels)) == 3
    assert score == score  # score n'est pas NaN

