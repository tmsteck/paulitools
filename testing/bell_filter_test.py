"""Independent Bell-filter contracts, chunk boundaries, and snapshot behavior."""

from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from paulitools import Pauli, ZXArray
from paulitools.bell_sampling.filters import (
    BellFilterState, BellSamplePool, bell_filter_mask, bell_filtered_purity,
    bell_purity, commuting_mask, y_parities,
)


MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def dense(body):
    matrix = np.ones((1, 1), dtype=complex)
    for char in body:
        matrix = np.kron(matrix, MATRICES[char])
    return matrix


def string_pairing(a, b):
    return sum(x != "I" and y != "I" and x != y for x, y in zip(a, b)) % 2


def reference_mask(samples, generators, shifted):
    return np.array([all(string_pairing(s, g) == (g.count("Y") % 2 if shifted else 0)
                         for g in generators) for s in samples], dtype=bool)


@pytest.mark.parametrize("parallel", [False, True])
def test_y_shift_differs_from_commuting_filter(parallel):
    samples = ZXArray.from_input(["I", "X", "Y", "Z"])
    np.testing.assert_array_equal(y_parities(samples, parallel=parallel), [0, 0, 1, 0])
    np.testing.assert_array_equal(commuting_mask(samples, "Y", parallel=parallel),
                                  [True, False, True, False])
    np.testing.assert_array_equal(bell_filter_mask(samples, "Y", parallel=parallel),
                                  [False, True, False, True])
    assert y_parities(samples).dtype == np.uint8
    assert bell_filter_mask(samples, "Y").dtype == np.bool_


@pytest.mark.parametrize("generators", [["YI", "IY"], ["XX", "ZZ"], ["YY"], ["II"]])
@pytest.mark.parametrize("parallel", [False, True])
def test_dense_matrix_filter_oracle(generators, parallel):
    samples = ["".join(chars) for chars in product("IXYZ", repeat=2)]
    sample_matrices = [dense(s) for s in samples]
    generator_matrices = [dense(g) for g in generators]
    expected_commuting, expected_bell = [], []
    for s in sample_matrices:
        expected_commuting.append(all(np.array_equal(s @ g, g @ s) for g in generator_matrices))
        # Transposition of ordinary Hermitian Pauli products contributes (-1)^Y.
        expected_bell.append(all(np.array_equal(s @ g, g.T @ s) for g in generator_matrices))
    rows = ZXArray.from_input(samples)
    np.testing.assert_array_equal(commuting_mask(rows, generators, parallel=parallel), expected_commuting)
    np.testing.assert_array_equal(bell_filter_mask(rows, generators, parallel=parallel), expected_bell)
    parities = np.array([0 if np.array_equal(m.T, m) else 1 for m in sample_matrices])
    expected_score = np.dot(1 - 2 * parities, expected_bell) / len(samples)
    assert bell_purity(rows) == np.mean(1 - 2 * parities)
    assert bell_filtered_purity(rows, generators, parallel=parallel) == expected_score


@pytest.mark.parametrize("width", [0, 1, 31, 32, 64, 65, 129])
@pytest.mark.parametrize("parallel", [False, True])
def test_chunk_boundaries_against_string_oracle(width, parallel):
    rng = np.random.default_rng(9763 + width)
    samples = ["".join(chars) for chars in rng.choice(list("IXYZ"), size=(23, width))]
    # A fixed local axis at each qubit produces a commuting group with varied Y.
    axes = rng.choice(list("XYZ"), size=width)
    selected = rng.integers(0, 2, size=(7, width))
    generators = ["".join(axis if take else "I" for axis, take in zip(axes, row)) for row in selected]
    phased_samples = [prefix + body for prefix, body in zip(np.resize(["+", "-", "+i", "-i"], 23), samples)]
    phased_generators = [prefix + body for prefix, body in zip(np.resize(["-i", "-", "+i"], 7), generators)]
    rows = ZXArray.from_input(phased_samples)
    gens = ZXArray.from_input(phased_generators)
    parity = np.array([s.count("Y") % 2 for s in samples], dtype=np.int64)
    np.testing.assert_array_equal(y_parities(rows, parallel=parallel), parity)
    expected = reference_mask(samples, generators, shifted=True)
    np.testing.assert_array_equal(bell_filter_mask(rows, gens, parallel=parallel), expected)
    np.testing.assert_array_equal(commuting_mask(rows, gens, parallel=parallel),
                                  reference_mask(samples, generators, shifted=False))
    assert bell_filtered_purity(rows, gens, parallel=parallel) == np.dot(1 - 2 * parity, expected) / len(samples)


def test_score_uses_original_denominator_and_negative_y_weights():
    assert bell_filtered_purity(["I", "Y"], "Z") == 0.5
    assert bell_filtered_purity(["Y", "Y", "X"], "I") == pytest.approx(-1 / 3)
    assert bell_purity(["Y", "Y", "X"]) == pytest.approx(-1 / 3)
    # Selection leaves no survivor: the statistic is zero, not an undefined mean.
    assert bell_filtered_purity(["Y"], "Z") == 0.0


@pytest.mark.parametrize("width", [0, 1, 32, 65])
def test_empty_generators_and_samples(width):
    empty = ZXArray.empty(width)
    samples = ZXArray.identities(width, 3)
    np.testing.assert_array_equal(commuting_mask(samples, []), np.ones(3, dtype=bool))
    np.testing.assert_array_equal(bell_filter_mask(samples, empty), np.ones(3, dtype=bool))
    assert bell_filtered_purity(samples, []) == bell_purity(samples) == 1
    assert y_parities(empty).shape == (0,)
    assert commuting_mask(empty, samples).shape == (0,)
    assert bell_filter_mask(empty, samples).shape == (0,)
    for operation in (lambda: bell_purity(empty), lambda: bell_filtered_purity(empty, [])):
        with pytest.raises(ValueError, match="At least one sample"):
            operation()
    pool = BellSamplePool(empty)
    state = pool.filter([])
    assert state.mask.shape == (0,)
    with pytest.raises(ValueError, match="At least one sample"):
        _ = pool.purity
    with pytest.raises(ValueError, match="At least one sample"):
        _ = state.score


@pytest.mark.parametrize("parallel", [False, True])
def test_incremental_filters_match_full_recomputation(parallel):
    samples = ZXArray.from_input(["".join(chars) for chars in product("IXYZ", repeat=3)])
    pool = BellSamplePool(samples)
    current = pool.filter([], parallel=parallel)
    generators = []
    # Last addition is dependent; its Y parity must still agree with the span.
    for addition in ["YII", "IYI", "IIY", "YYY", "-iYYY"]:
        old_mask, old_score = current.mask, current.score
        proposed = current.extend(addition, parallel=parallel)
        generators.append(addition)
        np.testing.assert_array_equal(proposed.mask, bell_filter_mask(samples, generators, parallel=parallel))
        assert proposed.score == bell_filtered_purity(samples, generators, parallel=parallel)
        np.testing.assert_array_equal(current.mask, old_mask)
        assert current.score == old_score
        current = proposed
    same = current.extend([], parallel=parallel)
    np.testing.assert_array_equal(same.mask, current.mask)
    assert same.score == current.score
    assert same.generators == current.generators


def test_snapshot_and_returned_values_are_independent():
    samples = ZXArray.from_input(["I", "X", "Y", "Z"])
    generators = ZXArray.from_input("Y")
    pool = BellSamplePool(samples)
    state = pool.filter(generators)
    expected_mask = np.array([False, True, False, True])
    samples[:] = "Y"
    generators[:] = "I"
    state.mask[:] = False
    returned_generators = state.generators
    returned_generators[:] = "I"
    np.testing.assert_array_equal(state.mask, expected_mask)
    assert state.generators.to_strings() == ["+Y"]
    assert state.score == 0.5
    assert pool.purity == 0.5
    assert pool.n_qubits == 1
    assert pool.n_samples == 4
    np.testing.assert_array_equal(pool.filter("Y").mask, expected_mask)
    with pytest.raises(FrozenInstanceError):
        pool._n_qubits = 2
    with pytest.raises(FrozenInstanceError):
        state._total = 99
    with pytest.raises(TypeError, match="BellSamplePool.filter"):
        BellFilterState()


def test_noncommuting_generators_and_bad_extensions_rejected():
    samples = ZXArray.from_input(["I", "X", "Y", "Z"])
    # Commutation postselection is defined even for a noncommuting constraint set.
    np.testing.assert_array_equal(commuting_mask(samples, ["X", "Z"]), [True, False, False, False])
    for operation in (bell_filter_mask, bell_filtered_purity):
        with pytest.raises(ValueError, match="mutually commute"):
            operation(samples, ["X", "Z"])
    pool = BellSamplePool(samples)
    with pytest.raises(ValueError, match="mutually commute"):
        pool.filter(["X", "Z"])
    state = pool.filter("X")
    old_mask = state.mask
    with pytest.raises(ValueError, match="current generators"):
        state.extend("Z")
    with pytest.raises(ValueError, match="mutually commute"):
        state.extend(["X", "Z"])
    np.testing.assert_array_equal(state.mask, old_mask)


def test_width_mismatch_and_explicit_raw_boundary():
    samples = ZXArray.from_input(["I", "Y"])
    pool = BellSamplePool(samples)
    state = pool.filter("Y")
    for operation in (lambda: commuting_mask(samples, "II"),
                      lambda: bell_filter_mask(samples, "II"),
                      lambda: bell_filtered_purity(samples, "II"),
                      lambda: pool.filter("II"), lambda: state.extend("II")):
        with pytest.raises(ValueError, match="width"):
            operation()
    # A Pauli object is accepted; packed arrays are explicitly wrapped by caller.
    np.testing.assert_array_equal(y_parities(Pauli("+iY")), [1])
    raw = samples.legacy_array(copy=True)
    assert bell_purity(ZXArray.from_raw(raw)) == 0


def test_shifted_mask_depends_on_commuting_support_span():
    samples = ZXArray.from_input(["".join(chars) for chars in product("IXYZ", repeat=2)])
    independent = ["YI", "IY"]
    alternative = ["YI", "YY"]
    redundant_phased = ["-YI", "+iIY", "-iYY", "II"]
    expected = bell_filter_mask(samples, independent)
    np.testing.assert_array_equal(bell_filter_mask(samples, alternative), expected)
    np.testing.assert_array_equal(bell_filter_mask(samples, redundant_phased), expected)
