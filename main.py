# main.py

"""
Script principal CLI pour lancer les étapes de ton projet CRISP-DM Assurance :

Actions possibles :

    python3 main.py --action prepare --csv dataAssurance.csv
    python3 main.py --action train   --csv dataAssurance.csv
    python3 main.py --action evaluate --csv dataAssurance.csv
    python3 main.py --action save    --csv dataAssurance.csv
    python3 main.py --action load    --model-path insurance_model.joblib
    python3 main.py --action cluster --csv dataAssurance.csv --clusters 3
"""

import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    prepare_clustering_data,
    run_kmeans_clustering,
)


def main():
    parser = argparse.ArgumentParser(
        description="CRISP-DM Insurance Prediction - Modular Pipeline"
    )
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        help="Action à exécuter : prepare | train | evaluate | save | load | cluster",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Chemin vers dataAssurance.csv (obligatoire pour prepare/train/evaluate/save/cluster)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="insurance_model.joblib",
        help="Chemin de sauvegarde/chargement du modèle de régression (défaut: insurance_model.joblib)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Nombre de clusters pour l'action 'cluster' (défaut: 3).",
    )

    args = parser.parse_args()
    action = args.action.lower()

    needs_csv = {"prepare", "train", "evaluate", "save", "cluster"}

    if action in needs_csv and not args.csv:
        print("❌ Veuillez fournir --csv chemin/vers/dataAssurance.csv")
        return

    if action == "prepare":
        X_train, X_test, y_train, y_test, df_encoded = prepare_data(args.csv)
        print("✅ Données préparées.")
        print(f"   X_train shape : {X_train.shape}")
        print(f"   X_test  shape : {X_test.shape}")
        print(f"   y_train shape : {y_train.shape}")
        print(f"   y_test  shape : {y_test.shape}")

    elif action == "train":
        X_train, X_test, y_train, y_test, _ = prepare_data(args.csv)
        model = train_model(X_train, y_train)
        print("✅ Modèle entraîné (RandomForestRegressor).")

    elif action == "evaluate":
        X_train, X_test, y_train, y_test, _ = prepare_data(args.csv)
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        print("✅ Résultats de l'évaluation :")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")

    elif action == "save":
        X_train, X_test, y_train, y_test, _ = prepare_data(args.csv)
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)
        print(f"✅ Modèle entraîné et sauvegardé dans '{args.model_path}'.")

    elif action == "load":
        model = load_model(args.model_path)
        print(f"✅ Modèle chargé depuis '{args.model_path}'.")
        print("   Tu peux maintenant l'utiliser dans un autre script pour prédire de nouveaux clients.")

    elif action == "cluster":
        X_scaled, scaler, feature_names, df_features = prepare_clustering_data(
            args.csv
        )
        labels, km, score = run_kmeans_clustering(X_scaled, best_k=args.clusters)

        print("✅ Clustering KMeans terminé.")
        print(f"   Nombre de clusters : {args.clusters}")
        print(f"   Silhouette score   : {score:.4f}")

        import numpy as np

        unique, counts = np.unique(labels, return_counts=True)
        print("   Nombre d'instances par cluster :")
        for u, c in zip(unique, counts):
            print(f"      Cluster {u}: {c} instances")

    else:
        print("❌ Action inconnue. Utilise : prepare | train | evaluate | save | load | cluster")


if __name__ == "__main__":
    main()

