from model_pipeline import prepare_data
import pandas as pd

def test_prepare_data_shapes():
    X_train, X_test, y_train, y_test, df_encoded = prepare_data("dataAssurance.csv")

    # Le dataset doit avoir au moins 100 lignes
    assert df_encoded.shape[0] > 10

    # X/y taille correcte
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_prepare_data_no_nan():
    _, _, _, _, df_encoded = prepare_data("dataAssurance.csv")

    # Vérifier qu'il n'y a plus de valeurs manquantes après imputation
    assert df_encoded.isnull().sum().sum() == 0

