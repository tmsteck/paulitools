"""Malformed packed arrays must fail before compiled indexed reads or shifts."""

import numpy as np
import pytest

from paulitools import (
    commutes, differences, ingroup, inner_product, left_pad, right_pad, row_reduce,
    symplectic_inner_product, symplectic_inner_product_int, toZX,
    unpack_sym_forms_to_matrices,
)
from paulitools.core import append
from paulitools.group import _packed_rows_to_binary


BAD_PACKED = [
    np.array([], dtype=np.int64),
    np.array([[1, 4]], dtype=np.int64),
    np.array([-1, 0], dtype=np.int64),
    np.array([32, 0], dtype=np.int64),
    np.array([1, -1], dtype=np.int64),
    np.array([1, 8], dtype=np.int64),
]


@pytest.mark.parametrize("bad", BAD_PACKED)
@pytest.mark.parametrize("kernel", [row_reduce, inner_product, differences, _packed_rows_to_binary])
def test_single_input_boundaries_reject_malformed_packed_arrays(kernel, bad):
    with pytest.raises(ValueError):
        kernel(bad)


@pytest.mark.parametrize("bad", BAD_PACKED)
def test_paired_difference_and_membership_validate_both_arrays(bad):
    valid = toZX("X")
    for left, right in ((bad, valid), (valid, bad)):
        with pytest.raises(ValueError):
            differences(left, right)
        for reduced in (False, True):
            with pytest.raises(ValueError):
                ingroup(left, right, reduced=reduced)
    # An empty candidate batch does not bypass validation of its reference span.
    with pytest.raises(ValueError):
        ingroup(np.array([1], dtype=np.int64), bad)


@pytest.mark.parametrize("bad", BAD_PACKED)
def test_single_operator_product_and_padding_validate_before_indexing(bad):
    valid = toZX("X")
    for left, right in ((bad, valid), (valid, bad)):
        with pytest.raises(ValueError):
            symplectic_inner_product(left, right)
        with pytest.raises(ValueError):
            commutes(left, right)
    for pad in (right_pad, left_pad):
        with pytest.raises(ValueError):
            pad(bad, 2)


@pytest.mark.parametrize("bad", [np.array([1], dtype=np.int64), toZX(["X", "Z"])])
def test_product_requires_exactly_one_operator_per_input(bad):
    with pytest.raises(ValueError, match="exactly one"):
        symplectic_inner_product(bad, toZX("X"))
    with pytest.raises(ValueError, match="exactly one"):
        symplectic_inner_product(toZX("X"), bad)


def test_valid_legacy_padding_and_scalar_product_behavior_is_preserved():
    x, zz = toZX("X"), toZX("ZZ")
    assert symplectic_inner_product(x, zz) == 1  # Legacy implicit right padding.
    assert symplectic_inner_product_int(toZX("XI")[1], zz[1], 2) == 1
    assert symplectic_inner_product(toZX("XI")[1], zz[1], 2) == 1
    np.testing.assert_array_equal(right_pad(zz, 1), zz)  # Historical shrink no-op.
    np.testing.assert_array_equal(right_pad(toZX("-X"), 2), toZX("-XI"))
    np.testing.assert_array_equal(left_pad(toZX("-X"), 2), toZX("-IX"))
    assert commutes(toZX(["X", "X"]), toZX(["X", "Z"]))  # Legacy first-row comparison.
    with pytest.raises(ValueError, match="at least one"):
        commutes(np.array([1], dtype=np.int64), x)
    with pytest.raises(ValueError, match="at least one"):
        commutes(x, np.array([1], dtype=np.int64))


def test_empty_batches_and_signed_int64_limit_remain_valid():
    for k in (0, 1, 31):
        empty = np.array([k], dtype=np.int64)
        np.testing.assert_array_equal(row_reduce(empty), empty)
        np.testing.assert_array_equal(differences(empty), empty)
        assert inner_product(empty).shape == (0, 0)
        assert _packed_rows_to_binary(empty).shape == (0, 2 * k)
        assert ingroup(empty, empty).shape == (0,)
    highest = np.array([31, np.iinfo(np.int64).max], dtype=np.int64)
    assert row_reduce(highest)[1] == np.iinfo(np.int64).max - 1
    assert symplectic_inner_product(highest, highest) == 0
    with pytest.raises(ValueError, match="same k"):
        ingroup(np.array([1], dtype=np.int64), np.array([2], dtype=np.int64))


@pytest.mark.parametrize("rows", [
    [(1, np.array([4], dtype=np.int64)), (2, np.array([4], dtype=np.int64))],
    [(1, np.array([], dtype=np.int64))],
    [(1, np.array([[4]], dtype=np.int64))],
    [(-1, np.array([0], dtype=np.int64))],
    [(32, np.array([0], dtype=np.int64))],
    [(1, np.array([-1], dtype=np.int64))],
    [(1, np.array([8], dtype=np.int64))],
])
def test_legacy_tuple_unpacking_rejects_inconsistent_or_short_rows(rows):
    with pytest.raises(ValueError):
        unpack_sym_forms_to_matrices(rows)


def test_legacy_tuple_unpacking_preserves_bit_order():
    x, z = unpack_sym_forms_to_matrices([(2, toZX("XY")[1:]), (2, toZX("ZI")[1:])])
    np.testing.assert_array_equal(x, [[1, 1], [0, 0]])
    np.testing.assert_array_equal(z, [[0, 1], [1, 0]])


def test_append_validates_headers_before_access_and_single_array_fast_path():
    for bad in (np.array([], dtype=np.int64), np.array([1, 8], dtype=np.int64)):
        with pytest.raises(ValueError):
            append([bad])
        with pytest.raises(ValueError):
            append([toZX("X"), bad])
