"""Optional independent Galois checks of the Bell support-algebra pipeline.

Install ``paulitools[verification]`` to run these. Galois supplies every oracle
row space and null space; no PauliTools reduction is used to form expected
results. These are binary support checks, not full-phase or sampling-independence
certifications. The mandatory tests separately check phases and exact operators.
"""

import numpy as np
import pytest

try:
    galois = pytest.importorskip("galois", reason="Optional Galois verification dependency is not installed")
except RuntimeError as error:
    # Some Galois versions enable Numba caching during import. A read-only
    # dependency installation can leave Numba without a usable cache locator.
    # Skip only that environment failure; unrelated import/runtime defects
    # must remain visible, and oracle execution errors are never caught here.
    message = str(error)
    if (message.startswith("cannot cache function ")
            and "no locator available for file " in message):
        pytest.skip(
            "Galois import cannot locate a writable Numba cache; set "
            "NUMBA_CACHE_DIR to a writable directory for optional verification",
            allow_module_level=True,
        )
    raise
GF2 = galois.GF2

from paulitools import ZXArray
from paulitools.bell_sampling import SupportBasis, bell_differences


def row_space(bits):
    return GF2(np.asarray(bits, dtype=np.uint8)).row_space()


def assert_same_span(actual, expected):
    np.testing.assert_array_equal(np.asarray(row_space(actual)), np.asarray(row_space(expected)))


def membership(candidates, generators):
    # Ordinary GF(2) annihilators characterize a row space, independently of
    # PauliTools' packed pivot representation and its coset implementation.
    annihilator = GF2(generators).null_space()
    return np.all(GF2(candidates) @ annihilator.T == 0, axis=1)


def coset_oracle(candidates, generators):
    reduced = GF2(candidates).copy()
    for row in row_space(generators):
        pivot = np.flatnonzero(row)[0]
        selected = np.flatnonzero(reduced[:, pivot])
        reduced[selected] += row
    return np.asarray(reduced, dtype=np.uint8)


def center_oracle(generators, width):
    basis = row_space(generators)
    swapped = GF2(np.concatenate((basis[:, width:], basis[:, :width]), axis=1))
    gram = basis @ swapped.T
    return (gram.null_space() @ basis).row_space()


def ambient_oracle(generators, width):
    bits = GF2(generators)
    swapped = GF2(np.concatenate((bits[:, width:], bits[:, :width]), axis=1))
    return swapped.null_space()


def phased_collection(bits, width, rng):
    phases = rng.integers(0, 4, len(bits), dtype=np.uint8)
    return ZXArray.from_bits(bits[:, :width], bits[:, width:], phases=phases)


@pytest.mark.parametrize("width", [1, 3, 10, 31, 32, 65])
@pytest.mark.parametrize("case", [0, 1, 2])
def test_prepared_support_and_two_stream_pipeline_against_galois(width, case):
    rng = np.random.default_rng(7500 + 11 * width + case)
    rows_a, rows_b = [(0, 3), (4, 5), (6, 4)][case]
    a = rng.integers(0, 2, size=(rows_a, 2 * width), dtype=np.uint8)
    b = rng.integers(0, 2, size=(rows_b, 2 * width), dtype=np.uint8)
    if case == 1:
        a[:, width:] = 0  # An isotropic space has its whole span as center.
    if rows_a:
        b[0] = a[0]  # Guarantee a common direction when this row is nonzero.
        a[-1] = a[0] ^ a[1]  # Include a dependent generator.
    a_object = phased_collection(a, width, rng)
    b_object = phased_collection(b, width, rng)
    left, right = SupportBasis(a_object), SupportBasis(b_object)
    a_basis = row_space(a)
    union_basis = row_space(np.concatenate((a, b), axis=0))

    np.testing.assert_array_equal(left.binary(), np.asarray(a_basis))
    assert left.rank == len(a_basis)
    assert_same_span(left.union(right).binary(), union_basis)
    dependencies = GF2(np.concatenate((a, b), axis=0)).T.null_space()
    expected_intersection = (dependencies[:, :rows_a] @ GF2(a)).row_space()
    assert_same_span(left.intersection(right).binary(), expected_intersection)
    assert left.quotient_dimension(right) == len(union_basis) - len(row_space(b))

    candidates = np.concatenate((a, b, rng.integers(0, 2, size=(9, 2 * width), dtype=np.uint8)), axis=0)
    np.testing.assert_array_equal(left.contains(candidates), membership(candidates, a))
    reduced = left.coset_reduce(candidates)
    np.testing.assert_array_equal(reduced.binary(), coset_oracle(candidates, a))
    assert membership(candidates ^ reduced.binary(), a).all()
    assert not reduced.phases().any()
    coefficients = rng.integers(0, 2, size=(9, left.rank), dtype=np.uint8)
    np.testing.assert_array_equal(left.linear_combinations(coefficients).binary(),
                                  np.asarray(GF2(coefficients) @ a_basis))

    assert_same_span(a_object.center().binary(), center_oracle(a, width))
    assert_same_span(a_object.centralizer().binary(), ambient_oracle(a, width))
    assert len(a_object.centralizer()) == 2 * width - len(a_basis)
    sampler = left.sampler(exclude_span=right)
    if left.quotient_dimension(right):
        draws = sampler.sample(17, rng=rng).binary()
        assert membership(draws, a).all()
        assert not membership(draws, b).any()
    else:
        assert len(sampler.sample(0, rng=rng)) == 0
        with pytest.raises(ValueError, match="No support elements"):
            sampler.sample(1, rng=rng)

    # Independent seeded streams exercise the complete numerical chain:
    # two Bell-label collections -> support XOR -> basis -> center. Uniform
    # labels here are an algebra test input, not a model of a particular state.
    seeds = np.random.SeedSequence(9000 + 11 * width + case).spawn(2)
    shots = [0, 7, 8][case]
    streams = [np.random.default_rng(seed).integers(0, 2, size=(shots, 2 * width), dtype=np.uint8)
               for seed in seeds]
    difference_oracle = GF2(streams[0]) + GF2(streams[1])
    differences = bell_differences(phased_collection(streams[0], width, rng),
                                   phased_collection(streams[1], width, rng),
                                   parallel=bool(case == 2))
    np.testing.assert_array_equal(differences.binary(), np.asarray(difference_oracle))
    assert not differences.phases().any()
    prepared = SupportBasis(differences)
    assert_same_span(prepared.binary(), difference_oracle)
    assert_same_span(prepared.basis.center().binary(), center_oracle(difference_oracle, width))
    assert_same_span(prepared.basis.centralizer().binary(), ambient_oracle(difference_oracle, width))


@pytest.mark.parametrize("labels, center_rank, ambient_rank", [
    (["XI", "ZI"], 0, 2),
    (["ZZ", "YX"], 2, 2),
])
def test_structured_center_and_coset_regressions_against_galois(labels, center_rank, ambient_rank):
    collection = ZXArray.from_input(labels)
    bits, width = collection.binary(), collection.n_qubits
    center = collection.center()
    ambient = collection.centralizer()
    assert len(center) == center_rank
    assert len(ambient) == ambient_rank
    assert_same_span(center.binary(), center_oracle(bits, width))
    assert_same_span(ambient.binary(), ambient_oracle(bits, width))
    reduced = SupportBasis(collection).coset_reduce(collection)
    np.testing.assert_array_equal(reduced.binary(), coset_oracle(bits, bits))
    assert not reduced.binary().any()
