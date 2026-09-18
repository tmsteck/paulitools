"""Bell-label XOR contracts, independently checked in the character alphabet."""

import numpy as np
import pytest

from paulitools import ZXArray
from paulitools.bell_sampling.differences import bell_differences, cyclic_bell_differences


def support_product(a, b):
    if a == b:
        return "I"
    if a == "I":
        return b
    if b == "I":
        return a
    return ({"X", "Y", "Z"} - {a, b}).pop()


@pytest.mark.parametrize("width", [0, 1, 31, 32, 64, 65, 129])
@pytest.mark.parametrize("parallel", [False, True])
def test_paired_against_character_oracle(width, parallel):
    rng = np.random.default_rng(314 + width)
    a = ["".join(row) for row in rng.choice(list("IXYZ"), size=(17, width))]
    b = ["".join(row) for row in rng.choice(list("IXYZ"), size=(17, width))]
    left = ZXArray.from_input([p + s for p, s in zip(np.resize(["+", "-", "+i", "-i"], 17), a)])
    right = ZXArray.from_input(["-i" + s for s in b])
    expected = ZXArray.from_input(["".join(support_product(x, y) for x, y in zip(u, v))
                                   for u, v in zip(a, b)])
    result = bell_differences(left, right, parallel=parallel)
    np.testing.assert_array_equal(result.binary(), expected.binary())
    np.testing.assert_array_equal(result.phases(), 0)
    assert result.n_qubits == width
    assert len(result) == 17
    if width:
        before = left.binary()
        result.set_bits(0, 0, z_bit=1 - int(result.z_bits()[0, 0]))
        np.testing.assert_array_equal(left.binary(), before)


@pytest.mark.parametrize("parallel", [False, True])
def test_difference_is_support_xor_not_operator_product(parallel):
    left, right = ZXArray.from_input("X"), ZXArray.from_input("Y")
    result = bell_differences(left, right, parallel=parallel)
    np.testing.assert_array_equal(result.binary(), ZXArray.from_input("Z").binary())
    assert result.phases().tolist() == [0]
    assert (left @ right).phases().tolist() == [1]


@pytest.mark.parametrize("width", [0, 1, 32, 65])
@pytest.mark.parametrize("parallel", [False, True])
def test_empty_and_cyclic_contract(width, parallel):
    empty = ZXArray.empty(width)
    assert len(bell_differences(empty, empty, parallel=parallel)) == 0
    assert cyclic_bell_differences(empty, parallel=parallel).n_qubits == width
    single = ZXArray.from_input("Y" * width)
    np.testing.assert_array_equal(cyclic_bell_differences(single, parallel=parallel).binary(), 0)
    samples = ZXArray.from_input(["X" * width, "Y" * width, "Z" * width])
    result = cyclic_bell_differences(samples, parallel=parallel)
    np.testing.assert_array_equal(result.binary(), ZXArray.from_input(["Z" * width, "X" * width, "Y" * width]).binary())
    np.testing.assert_array_equal(np.bitwise_xor.reduce(result.binary(), axis=0), 0)


def test_width_and_count_must_match_without_broadcast():
    with pytest.raises(ValueError, match="widths"):
        bell_differences("X", "XX")
    with pytest.raises(ValueError, match="same number"):
        bell_differences(["X", "Y"], "X")
    raw = ZXArray.from_input("X").legacy_array()
    np.testing.assert_array_equal(bell_differences(ZXArray.from_raw(raw), "X").binary(), 0)


def test_kernels_compile_in_nopython_mode():
    from paulitools.bell_sampling.differences import _paired_xor, _paired_xor_parallel
    for parallel in [False, True]:
        bell_differences("X", "Z", parallel=parallel)
    assert _paired_xor.nopython_signatures
    assert _paired_xor_parallel.nopython_signatures
