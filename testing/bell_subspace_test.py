"""Independent GF(2) checks of prepared Bell-sampling support subspaces."""

from itertools import product

import numpy as np
import pytest

from paulitools import Pauli, ZXArray
from paulitools.bell_sampling.subspace import SupportBasis, SupportSampler
from paulitools.bell_sampling import subspace


def brute_span(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    span = {tuple(np.zeros(bits.shape[1], dtype=np.uint8))}
    for row in bits:
        span |= {tuple(int(a) ^ int(b) for a, b in zip(value, row)) for value in tuple(span)}
    return span


def bit_rows(values, width):
    return np.array(sorted(values), dtype=np.uint8).reshape(len(values), 2 * width)


def row_set(collection):
    return {tuple(row) for row in collection.binary()}


@pytest.mark.parametrize("width", [0, 1, 2, 3])
def test_membership_and_cosets_against_exhaustive_binary_oracle(width):
    rng = np.random.default_rng(129 + width)
    all_bits = np.array(list(product((0, 1), repeat=2 * width)), dtype=np.uint8).reshape(4 ** width, 2 * width)
    for count in [0, 1, 2, 5]:
        bits = rng.integers(0, 2, size=(count, 2 * width), dtype=np.uint8)
        original = ZXArray.from_bits(bits[:, :width], bits[:, width:])
        basis = SupportBasis(original)
        expected = brute_span(bits)
        np.testing.assert_array_equal(basis.contains(all_bits), [tuple(row) in expected for row in all_bits])
        representatives = basis.coset_reduce(all_bits)
        np.testing.assert_array_equal(representatives.phases(), np.zeros(len(all_bits)))
        assert brute_span(basis.binary()) == expected
        assert basis.rank == (len(expected).bit_length() - 1)
        for row, representative in zip(all_bits, representatives.binary()):
            assert tuple(row ^ representative) in expected
            coset = bit_rows({tuple(int(a) ^ int(b) for a, b in zip(row, member)) for member in expected}, width)
            expected_rows = np.repeat(representative[None, :], len(coset), axis=0)
            np.testing.assert_array_equal(basis.coset_reduce(coset).binary(), expected_rows)
        shuffled = SupportBasis(original[::-1])
        np.testing.assert_array_equal(shuffled.binary(), basis.binary())


def test_robels_pivot_order_regression_and_phase_blind_contract():
    # These are commuting strings. Robels' old inconsistent pivot order reduced
    # ZZ to XY rather than identity; preparation and reduction must agree.
    basis = SupportBasis(["ZZ", "YX"])
    assert basis.coset_reduce(["ZZ", "YX", "-iXY", "+iII"]).to_strings() == ["+II"] * 4
    assert basis.contains(["-ZZ", "+iYX", "-iXY", "II"]).all()
    # Opposite signs impose no contradiction in phase-free support algebra.
    signed = SupportBasis(["X", "-X", "+iX", "-iX"])
    assert signed.rank == 1
    assert signed.basis.to_strings() == ["+X"]
    np.testing.assert_array_equal(signed.linear_combinations([[0], [1]]).phases(), [0, 0])


@pytest.mark.parametrize("width", [0, 1, 2, 3])
def test_union_intersection_and_quotient_against_sets(width):
    rng = np.random.default_rng(123 + width)
    for size_a, size_b in [(0, 0), (0, 2), (2, 0), (2, 3), (4, 2)]:
        a = rng.integers(0, 2, size=(size_a, 2 * width), dtype=np.uint8)
        b = rng.integers(0, 2, size=(size_b, 2 * width), dtype=np.uint8)
        left, right = SupportBasis(a), SupportBasis(b)
        sa, sb = brute_span(a), brute_span(b)
        expected_union = {tuple(int(a) ^ int(b) for a, b in zip(x, y)) for x in sa for y in sb}
        assert brute_span(left.union(right).binary()) == expected_union
        assert brute_span(left.intersection(right).binary()) == sa & sb
        assert left.quotient_dimension(right) == (len(expected_union) // len(sb)).bit_length() - 1
        assert left.rank + right.rank == left.union(right).rank + left.intersection(right).rank


@pytest.mark.parametrize("width", [0, 31, 32, 64, 65, 129])
def test_chunk_boundaries_combinations_membership_and_phases(width):
    rng = np.random.default_rng(width + 813)
    bits = rng.integers(0, 2, size=(5, 2 * width), dtype=np.uint8)
    phases = rng.integers(0, 4, size=5, dtype=np.uint8)
    collection = ZXArray.from_bits(bits[:, :width], bits[:, width:], phases=phases)
    basis = SupportBasis(collection)
    coefficients = np.array(list(product((0, 1), repeat=basis.rank)), dtype=np.uint8).reshape(2 ** basis.rank, basis.rank)
    combinations = basis.linear_combinations(coefficients)
    assert row_set(combinations) == brute_span(bits)
    assert basis.contains(combinations).all()
    assert not basis.coset_reduce(combinations).binary().any()
    assert not combinations.phases().any()
    other = SupportBasis(collection[:2])
    assert basis.intersection(other).rank == other.rank
    sampler = basis.sampler(exclude_span=other)
    if basis.rank > other.rank:
        samples = sampler.sample(127, rng=rng)
        assert basis.contains(samples).all()
        assert not other.contains(samples).any()


def test_prepared_sampling_uniform_exclusion_not_contained_in_basis():
    # The excluded span also contains elements outside A; only A intersection B
    # is removed. A has eight elements and exactly six are eligible.
    a = SupportBasis(["XI", "ZI", "IX"])
    b = SupportBasis(["XI", "ZZ"])
    expected = brute_span(a.binary()) - brute_span(b.binary())
    assert len(expected) == 6
    sampler = a.sampler(exclude_span=b)
    samples = sampler.sample(12000, rng=np.random.default_rng(71))
    unique, counts = np.unique(samples.binary(), axis=0, return_counts=True)
    assert {tuple(row) for row in unique} == expected
    # A fixed seed and a deliberately loose frequency bound avoid fragile tests.
    assert np.all(np.abs(counts - 2000) < 180)
    repeat = sampler.sample(12000, rng=np.random.default_rng(71))
    np.testing.assert_array_equal(repeat.binary(), samples.binary())


def test_prepared_sampler_reuses_exclusion_decomposition(monkeypatch):
    a = SupportBasis(["XI", "ZI", "IX"])
    excluded_input = ZXArray.from_input(["XI", "ZZ"])
    sampler = a.sampler(exclude_span=excluded_input)
    excluded_input[0] = "II"

    def unexpected_reduction(*args, **kwargs):
        raise AssertionError("Prepared draws must not repeat null-space preparation")

    monkeypatch.setattr(subspace, "null_space", unexpected_reduction)
    rng = np.random.default_rng(91)
    for _ in range(5):
        samples = sampler.sample(1, rng=rng)
        assert a.contains(samples).all()
        assert not SupportBasis(["XI", "ZZ"]).contains(samples).any()


def test_rank_one_complement_and_identity_exclusion():
    # Excluding a codimension-one subspace accepts one half of the ambient
    # coefficients. Every one of the four eligible elements must be sampled.
    basis = SupportBasis(["XI", "ZI", "IX"])
    excluded = SupportBasis(["XI", "ZI"])
    draws = basis.sample(600, rng=np.random.default_rng(72), exclude_span=excluded)
    assert row_set(draws) == brute_span(basis.binary()) - brute_span(excluded.binary())
    nonidentity = basis.sample(600, rng=np.random.default_rng(73), exclude_identity=True)
    assert row_set(nonidentity) == brute_span(basis.binary()) - {(0, 0, 0, 0)}
    empty_span = SupportBasis([], n_qubits=2)
    assert row_set(basis.sample(600, rng=np.random.default_rng(73), exclude_span=empty_span)) == row_set(nonidentity)


@pytest.mark.parametrize("width", [0, 1, 65])
def test_empty_span_empty_draws_and_impossible_exclusion(width):
    basis = SupportBasis([], n_qubits=width)
    assert basis.rank == 0
    assert basis.contains(ZXArray.identities(width, count=2)).all()
    assert basis.sample(3, rng=np.random.default_rng(1)).to_strings() == ["+" + "I" * width] * 3
    assert basis.linear_combinations(np.empty((4, 0))).n_paulis == 4
    for kwargs in ({"exclude_identity": True}, {"exclude_span": basis}):
        sampler = basis.sampler(**kwargs)
        empty = sampler.sample(0, rng=np.random.default_rng(2))
        assert empty.n_qubits == width and empty.n_paulis == 0
        with pytest.raises(ValueError, match="No support elements"):
            sampler.sample(1, rng=np.random.default_rng(2))
    assert basis.coset_reduce([]).n_paulis == 0
    assert len(basis.contains([])) == 0


def test_input_and_output_ownership_and_explicit_numeric_contract():
    original = ZXArray.from_input(["ZZ", "YX"])
    expected = SupportBasis(original)
    basis = SupportBasis(original)
    original[0] = "II"
    raw = basis.binary()
    raw[:] = 0
    view = basis.basis
    view[0] = "II"
    view2 = basis.to_zxarray()
    view2[0] = "II"
    np.testing.assert_array_equal(basis.binary(), expected.binary())
    copy = SupportBasis(basis)
    np.testing.assert_array_equal(copy.binary(), basis.binary())
    assert SupportBasis(Pauli("+iZZ")).rank == 1
    assert SupportBasis([[1, 0, 0, 1]]).basis.to_strings() == ["+ZX"]
    with pytest.raises(ValueError):
        SupportBasis([[-1, 1]])


def test_invalid_arguments_and_widths():
    basis = SupportBasis(["X"])
    rng = np.random.default_rng(1)
    with pytest.raises(TypeError, match="bool"):
        SupportBasis(ZXArray.from_input("X"), n_qubits=True)
    with pytest.raises(ValueError, match="nonnegative"):
        SupportBasis([], n_qubits=-1)
    for method in (basis.contains, basis.coset_reduce, basis.union, basis.intersection, basis.quotient_dimension):
        with pytest.raises(ValueError, match="width"):
            method(["XX"])
    with pytest.raises(ValueError, match="width"):
        basis.sampler(exclude_span=SupportBasis(["XX"]))
    with pytest.raises(ValueError, match="nonnegative"):
        basis.sample(-1, rng=rng)
    for invalid in (1.5, True):
        with pytest.raises(TypeError):
            basis.sample(invalid, rng=rng)
    with pytest.raises(TypeError, match="Generator"):
        basis.sample(1, rng=1)
    with pytest.raises(TypeError, match="bool"):
        basis.sampler(exclude_identity=1)
    with pytest.raises(TypeError, match="SupportBasis"):
        SupportSampler(["X"])
    for coefficients in ([[2]], [[-1]], [[np.nan]], [0, 1], np.zeros((1, 1, 1))):
        with pytest.raises(ValueError):
            basis.linear_combinations(coefficients)
    assert basis.linear_combinations([1]).to_strings() == ["+X"]
    with pytest.raises(AttributeError):
        basis.rank = 5
    with pytest.raises(AttributeError):
        basis.n_qubits = 5


def test_numeric_hot_paths_compile_in_nopython_mode():
    basis = SupportBasis(["XI", "ZI", "IX"])
    basis.contains(["XX"])
    basis.coset_reduce(["XX"])
    basis.linear_combinations([[1, 1, 1]])
    basis.sample(2, rng=np.random.default_rng(4), exclude_span=["XI"])
    for kernel in (subspace._reduce_chunks, subspace._contains_chunks, subspace._combine_chunks, subspace._combine_bits):
        assert kernel.nopython_signatures
        assert kernel.targetoptions["nogil"] is True


def test_large_exclusion_complement_without_ambient_rejection():
    width = 65
    bits = np.concatenate((np.eye(width, dtype=np.uint8), np.zeros((width, width), dtype=np.uint8)), axis=1)
    basis = SupportBasis(bits)
    excluded = SupportBasis(bits[:-1])
    sampler = basis.sampler(exclude_span=excluded)
    samples = sampler.sample(31, rng=np.random.default_rng(45))
    assert basis.contains(samples).all()
    assert not excluded.contains(samples).any()
    assert samples.binary()[:, width - 1].all()
    assert basis.quotient_dimension(excluded) == 1
