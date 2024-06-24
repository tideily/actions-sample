from train import build_model, fit, load_data


def test_load_data_loads_images():
    X_train, y_train, X_test, y_test = load_data()

    assert X_train.shape == (60000, 784)
    assert y_train.shape == (60000,)
    assert X_test.shape == (10000, 784)
    assert y_test.shape == (10000,)


def test_load_data_loads_single_batch():
    X_train, y_train, _, _ = load_data(single_batch=True)

    assert X_train.shape == (32, 784)
    assert y_train.shape == (32,)


def test_load_data_scales_images():
    X_train, _, X_test, _ = load_data()

    assert X_train.min() >= 0 and X_train.max() <= 1
    assert X_test.min() >= 0 and X_test.max() <= 1


def test_build_model():
    model = build_model()
    assert model.input_shape == (None, 784), "Input layer shape is incorrect"
    assert model.output_shape == (None, 10), "Output layer shape is incorrect"
    assert len(model.layers) == 3, "Number of layers is incorrect"


def test_fit():
    model, history = fit(single_batch=True)

    assert model is not None
    assert history is not None
    assert "accuracy" in history.history
    assert len(history.history["accuracy"]) > 0
