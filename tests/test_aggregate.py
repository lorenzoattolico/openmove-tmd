"""Unit test del modulo aggregate (data-free)."""
import numpy as np

from tmd.aggregate import (modal_split, confusion_matrix, quantify, co2,
                           MOVING_MODES, EMISSION_FACTORS)


def test_modal_split_proportions():
    labels = ["Walk"] * 2 + ["Car"] * 6 + ["Bus"] * 2
    ms = modal_split(labels)
    assert abs(sum(ms.values()) - 1.0) < 1e-9
    assert abs(ms["Car"] - 0.6) < 1e-9
    assert abs(ms["Walk"] - 0.2) < 1e-9
    assert ms["Train"] == 0.0


def test_quantify_identity_is_noop():
    M = np.eye(len(MOVING_MODES))
    q = np.array([0.25, 0.25, 0.40, 0.10])
    assert np.allclose(quantify(q, M), q, atol=1e-9)


def test_quantify_debiases():
    # Vero 50/50 Walk/Car; il classificatore manda meta' dei Walk in Car.
    M = np.array([[0.5, 0, 0.5, 0],
                  [0,   1, 0,   0],
                  [0,   0, 1,   0],
                  [0,   0, 0,   1]], dtype=float)
    q_pred = np.array([0.25, 0.0, 0.75, 0.0])     # quote osservate (Walk sotto-contato)
    out = quantify(q_pred, M)
    assert np.allclose(out, [0.5, 0.0, 0.5, 0.0], atol=1e-6)


def test_confusion_matrix_rows_normalized():
    true = ["Walk", "Walk", "Car", "Car", "Car"]
    pred = ["Walk", "Car", "Car", "Car", "Bus"]
    M = confusion_matrix(true, pred)
    for i in range(len(MOVING_MODES)):
        s = M[i].sum()
        assert abs(s - 1.0) < 1e-9          # ogni riga somma 1 (fallback identita' incluso)


def test_co2():
    km = {"Walk": 10.0, "Car": 5.0, "Train": 20.0}
    assert co2(km) == 5.0 * 170 + 20.0 * 40   # Walk = 0
