"""Independent checks for the documented Bell-outcome estimators."""

import itertools

import numpy as np
import pytest

from paulitools import (
    Pauli,
    Pauli_expectation,
    getCentralizer,
    get_pauli_obs,
    get_pauli_pauli_obs,
    toZX,
    toZXArray,
)
from paulitools.util import _weighted_pauli_estimates, _weighted_pauli_estimates_parallel


def _reference(observable, distribution, include_shot_y=False):
    sign = observable.startswith("-")
    observable = observable.lstrip("+-")
    total = 0.0
    for shot, weight in distribution.items():
        shot = shot.lstrip("+-")
        anticommutes = sum(a != "I" and b != "I" and a != b for a, b in zip(observable, shot))
        exponent = sign + observable.count("Y") + anticommutes
        if include_shot_y:
            exponent += shot.count("Y")
        total += weight * (-1) ** exponent
    return total


@pytest.mark.parametrize("include_shot_y", [False, True])
def test_every_one_qubit_sign_and_outcome(include_shot_y):
    estimate = get_pauli_pauli_obs if include_shot_y else get_pauli_obs
    for observable, shot in itertools.product("IXYZ", repeat=2):
        for prefix in ("", "-"):
            distribution = {shot: 1.0}
            expected = _reference(prefix + observable, distribution, include_shot_y)
            np.testing.assert_array_equal(estimate(prefix + observable, distribution), [expected])
            np.testing.assert_array_equal(estimate([prefix + observable], distribution), [expected])
            np.testing.assert_array_equal(estimate(toZX([prefix + observable]), distribution), [expected])


@pytest.mark.parametrize("include_shot_y", [False, True])
def test_random_weighted_estimates_and_parallel_agree_with_string_reference(include_shot_y):
    rng = np.random.default_rng(338)
    estimate = get_pauli_pauli_obs if include_shot_y else get_pauli_obs
    for width in (1, 4, 31):
        observables = ["".join(rng.choice(list("IXYZ"), width)) for _ in range(13)]
        observables[0] = "-" + observables[0]
        outcomes = list(dict.fromkeys("".join(rng.choice(list("IXYZ"), width)) for _ in range(19)))
        weights = rng.random(len(outcomes))
        weights /= weights.sum()
        distribution = dict(zip(outcomes, weights))
        expected = [_reference(p, distribution, include_shot_y) for p in observables]
        np.testing.assert_allclose(estimate(observables, distribution), expected, atol=2e-15)
        np.testing.assert_allclose(estimate(observables, distribution, parallel=True), expected, atol=2e-15)
    assert _weighted_pauli_estimates.nopython_signatures
    assert _weighted_pauli_estimates_parallel.nopython_signatures


def test_binary_outcomes_and_ignored_outcome_phase():
    distribution = {"00": 0.1, "01": 0.2, "11": 0.3, "10": 0.4}
    symbolic = {"I": 0.1, "X": 0.2, "Y": 0.3, "Z": 0.4}
    for estimator in (get_pauli_obs, get_pauli_pauli_obs):
        expected = estimator(["I", "X", "Y", "Z"], symbolic)
        np.testing.assert_allclose(estimator(["I", "X", "Y", "Z"], distribution), expected)
        negative_labels = {"-" + s: w for s, w in symbolic.items()}
        np.testing.assert_allclose(estimator(["I", "X", "Y", "Z"], negative_labels), expected)


def test_python_estimators_accept_wrappers_and_reject_imaginary_observables():
    distribution = {"I": 0.25, "X": 0.75}
    for estimator in (get_pauli_obs, get_pauli_pauli_obs):
        expected = estimator(["X", "-Y"], distribution)
        np.testing.assert_array_equal(estimator(toZXArray(["X", "-Y"]), distribution), expected)
        np.testing.assert_array_equal(estimator(toZXArray(["X", "-Y"], force_large=True), distribution), expected)
        np.testing.assert_array_equal(estimator(Pauli("-Y"), distribution), expected[1:])
        for imaginary in (Pauli("iX"), toZXArray(["iX", "Y"])):
            with pytest.raises(ValueError):
                estimator(imaginary, distribution)
    assert Pauli_expectation([[0, 1]], Pauli("-Y")) == 1.0
    with pytest.raises(ValueError):
        Pauli_expectation([[0, 1]], Pauli("iY"))


def test_scalar_api_uses_first_convention_and_preserves_large_exact_integers():
    distribution = {"I": 0.1, "X": 0.2, "Y": 0.3, "Z": 0.4}
    rows = [[int(toZX(shot)[1]), weight] for shot, weight in distribution.items()]
    for observable in ("I", "X", "Y", "Z", "-Y"):
        assert Pauli_expectation(rows, toZX(observable)) == pytest.approx(_reference(observable, distribution))
    shot, observable = "I" * 30 + "X", "I" * 30 + "Y"
    value = int(toZX(shot)[1])
    assert value > 2 ** 53
    for exact_rows in ([[value, 1.0]], np.array([[value, 1.0]], dtype=object)):
        assert Pauli_expectation(exact_rows, toZX(observable)) == _reference(observable, {shot: 1.0})
    # All 63 packed bits, including the outcome's ignored sign, are valid at k=31.
    assert np.isfinite(Pauli_expectation([[2 ** 63 - 1, 1.0]], toZX("I" * 31)))


@pytest.mark.parametrize("dtype,limit", [(np.float16, 2 ** 11), (np.float32, 2 ** 24), (np.float64, 2 ** 53)])
def test_float_integer_precision_guard(dtype, limit):
    observable = toZX("I" * 31)
    with pytest.raises(ValueError, match="precision"):
        Pauli_expectation(np.array([[limit, 1]], dtype=dtype), observable)
    assert Pauli_expectation(np.array([[limit - 1, 1]], dtype=dtype), observable) == 1.0


@pytest.mark.parametrize("distribution", [
    {}, {"I": 0.0}, {"I": 2.0}, {"I": -1.0, "X": 2.0},
    {"I": np.nan}, {"I": np.inf}, {"I": 1j},
])
def test_invalid_probability_distributions_rejected(distribution):
    for estimator in (get_pauli_obs, get_pauli_pauli_obs):
        with pytest.raises((TypeError, ValueError)):
            estimator("X", distribution)


@pytest.mark.parametrize("shots", [
    [], [0, 1], [[0, 1, 2]], [[0, 0]], [[0, 2]], [[0, -1], [2, 2]],
    [[0.5, 1]], [[np.nan, 1]], [[np.inf, 1]], [[-1, 1]], [[8, 1]],
    [[0, np.nan]], [[0, np.inf]], [[0, 1j]], [[1j, 1]],
])
def test_invalid_scalar_inputs_rejected(shots):
    with pytest.raises((TypeError, ValueError)):
        Pauli_expectation(shots, toZX("X"))


def test_width_shape_and_single_observable_contracts():
    for estimator in (get_pauli_obs, get_pauli_pauli_obs):
        with pytest.raises(ValueError, match="qubit"):
            estimator("XX", {"I": 1.0})
        with pytest.raises(ValueError, match="qubit"):
            estimator("X", {"I": 0.5, "II": 0.5})
        with pytest.raises(TypeError):
            estimator(np.array([1.0, 4.0]), {"I": 1.0})
        with pytest.raises((TypeError, ValueError)):
            estimator("X", {0: 1.0})
        with pytest.raises(TypeError):
            estimator("X", [("I", 1.0)])
        assert estimator(np.array([1], dtype=np.int64), {"I": 1.0}).shape == (0,)
    with pytest.raises(ValueError, match="exactly one"):
        Pauli_expectation([[0, 1]], toZX(["X", "Z"]))


def _span(rows):
    values = {0}
    for row in rows:
        values |= {value ^ int(row) for value in list(values)}
    return values


@pytest.mark.parametrize("outcomes", [["II"], ["II", "XX"], ["II", "XI", "ZI"], ["II", "XX", "ZZ", "YY"]])
def test_getcentralizer_matches_enumerated_center_of_pairwise_difference_span(outcomes):
    center, generators = getCentralizer(dict.fromkeys(outcomes, 1), return_generators=True)
    packed = [int(toZX(outcome)[1]) >> 1 for outcome in outcomes]
    expected_span = _span(a ^ b for a, b in itertools.combinations(packed, 2))
    assert _span(generators[1:] >> 1) == expected_span
    expected_center = {
        a for a in expected_span
        if all(bin(((a & 3) & (b >> 2)) ^ ((a >> 2) & (b & 3))).count("1") % 2 == 0 for b in expected_span)
    }
    center_packed = [sum(int(bit) << j for j, bit in enumerate(row)) for row in center]
    assert _span(center_packed) == expected_center


def test_getcentralizer_rejects_empty_and_mismatched_outcomes():
    with pytest.raises(ValueError):
        getCentralizer({})
    with pytest.raises(ValueError, match="qubit"):
        getCentralizer({"00": 1, "0000": 1})
