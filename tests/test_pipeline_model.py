from model_pipeline import prepare_data, train_model, evaluate_model

def test_train_model():
    X_train, X_test, y_train, y_test, _ = prepare_data("dataAssurance.csv")
    model = train_model(X_train, y_train)

    # Le modèle doit avoir été entraîné
    assert hasattr(model, "predict")

def test_evaluate_model():
    X_train, X_test, y_train, y_test, _ = prepare_data("dataAssurance.csv")
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Vérifier que toutes les métriques existent
    assert "Test_R2" in metrics
    assert "Test_MAE" in metrics
    assert "Test_RMSE" in metrics

    # R2 ne doit pas être NaN
    assert metrics["Test_R2"] == metrics["Test_R2"]

