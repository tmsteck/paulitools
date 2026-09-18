"""Independent operator-matrix and phase regressions for the object API."""

from itertools import product

import numpy as np
import pytest

from paulitools import Pauli, ZXArray, toZX, toZXArray


MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}
PREFIX = ("+", "+i", "-", "-i")


def label(q, body):
    return PREFIX[q] + body


def dense_label(text):
    if text.startswith(("+i", "-i")):
        coefficient, body = (1j if text[0] == "+" else -1j), text[2:]
    elif text.startswith(("+", "-")):
        coefficient, body = (1 if text[0] == "+" else -1), text[1:]
    else:
        coefficient, body = 1, text
    result = np.array([[coefficient]], dtype=complex)
    for char in body:
        result = np.kron(result, MATRICES[char])
    return result


def local_product_table():
    table = {}
    for a, b in product(MATRICES, repeat=2):
        matrix = MATRICES[a] @ MATRICES[b]
        matches = [(q, c) for q, c in product(range(4), MATRICES)
                   if np.array_equal(matrix, (1j ** q) * MATRICES[c])]
        assert len(matches) == 1
        table[a, b] = matches[0]
    return table


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("force_large", [False, True])
def test_every_phased_dense_product(width, force_large):
    bodies = ["".join(chars) for chars in product("IXYZ", repeat=width)]
    labels = [label(q, body) for q, body in product(range(4), bodies)]
    collection = ZXArray.from_input(labels, force_large=force_large)
    dense = [dense_label(text) for text in labels]
    # Both broadcast directions exercise ordered multiplication for all pairs.
    for index, text in enumerate(labels):
        operator = Pauli(text, force_large=force_large)
        forward = collection @ operator
        backward = operator @ collection
        for row, (a, b) in enumerate(zip(forward.to_strings(), backward.to_strings())):
            np.testing.assert_array_equal(dense_label(a), dense[row] @ dense[index])
            np.testing.assert_array_equal(dense_label(b), dense[index] @ dense[row])
        scalar = operator @ Pauli(text)
        np.testing.assert_array_equal(dense_label(str(scalar)), dense[index] @ dense[index])
        np.testing.assert_array_equal(dense_label(str(operator.adjoint())), dense[index].conj().T)


@pytest.mark.parametrize("width", [31, 32, 64, 65, 129])
def test_chunk_boundary_products_against_local_matrix_oracle(width):
    rng = np.random.default_rng(9341 + width)
    alphabet = np.array(list("IXYZ"))
    table = local_product_table()
    a = ["".join(row) for row in rng.choice(alphabet, size=(17, width))]
    b = ["".join(row) for row in rng.choice(alphabet, size=(17, width))]
    a[0], b[0] = "I" * (width - 1) + "X", "I" * (width - 1) + "Z"
    qa, qb = rng.integers(0, 4, size=(2, 17))
    expected = []
    for row in range(17):
        phase, chars = int(qa[row] + qb[row]), []
        for left, right in zip(a[row], b[row]):
            q, char = table[left, right]
            phase += q
            chars.append(char)
        expected.append(label(phase % 4, "".join(chars)))
    left = ZXArray.from_input([label(int(q), s) for q, s in zip(qa, a)])
    right = ZXArray.from_input([label(int(q), s) for q, s in zip(qb, b)])
    assert (left @ right).to_strings() == expected
    assert (left @ ZXArray.from_input(right, force_large=True)).to_strings() == expected


@pytest.mark.parametrize("width", [1, 31, 32, 65])
def test_numeric_object_inputs_have_explicit_encoding(width):
    row = np.ones(2 * width, dtype=np.int8)
    assert str(Pauli(row)) == "+" + "Y" * width
    assert str(Pauli(row, encoding="eigenvalues")) == "+" + "I" * width
    batch = np.array([row, -row])
    with pytest.raises(ValueError):
        toZXArray(batch)
    explicit = toZXArray(batch, encoding="eigenvalues")
    assert explicit.to_strings() == ["+" + "I" * width, "+" + "Y" * width]
    split = ZXArray.from_eigenvalues(batch[:, :width], batch[:, width:])
    assert split == explicit


def test_width_mismatch_requires_explicit_padding():
    with pytest.raises(ValueError, match="width"):
        ZXArray.from_input(["X", "YY"])
    with pytest.raises(ValueError, match="width"):
        Pauli("X") @ Pauli("YY")
    with pytest.raises(ValueError, match="width"):
        Pauli("X").commutes(Pauli("YY"))
    assert ZXArray.from_input(["-iX", "YY"], n_qubits=3).to_strings() == ["-iXII", "+YYI"]
    assert str(Pauli("-iY").pad(3)) == "-iYII"
    assert str(Pauli("-iY").pad(3, side="left")) == "-iIIY"
    with pytest.raises(ValueError):
        Pauli("XX").pad(1)
    with pytest.raises(ValueError):
        Pauli("XX", n_qubits=1)


@pytest.mark.parametrize("source", ["X", np.array([0, 1]), Pauli("X"), toZXArray("X")])
def test_explicit_width_type_is_consistent_for_all_input_paths(source):
    for invalid in (1.0, 1.5, "1"):
        with pytest.raises(TypeError):
            Pauli(source, n_qubits=invalid)
    assert str(Pauli(source, n_qubits=np.int64(1))) == "+X"


@pytest.mark.parametrize("force_large", [False, True])
def test_indexing_constructor_and_kernel_buffers_copy(force_large):
    source = ZXArray.from_input(["X", "-iY", "Z"], force_large=force_large)
    chosen = source[1]
    chosen.phase = 0
    chosen.set_bits(0, z_bit=0)
    assert str(chosen) == "+X"
    assert source.to_strings() == ["+X", "-iY", "+Z"]
    source[1] = chosen
    assert source.to_strings() == ["+X", "+X", "+Z"]
    chosen.phase = 2
    assert str(source[1]) == "+X"
    subset = source[[2, 0]]
    subset.set_bits(0, 0, x_bit=1)
    assert source.to_strings() == ["+X", "+X", "+Z"]
    copied = Pauli(source[0])
    copied.phase = 3
    assert str(source[0]) == "+X"
    _, z, x, q = source.kernel_args()
    z[:] = 0
    x[:] = 0
    q[:] = 3
    assert source.to_strings() == ["+X", "+X", "+Z"]
    source[::-1] = source
    assert source.to_strings() == ["+Z", "+X", "+X"]
    source[:2] = Pauli("-iY")
    assert source.to_strings() == ["-iY", "-iY", "+X"]


def test_explicit_raw_aliasing_and_copy_contract():
    raw = toZX("X")
    alias = ZXArray.from_raw(raw, copy=False)
    scalar = Pauli.from_raw(raw)
    raw[1] = toZX("-Z")[1]
    assert alias.to_strings() == ["-Z"]
    assert str(scalar) == "+X"
    alias.legacy_array(copy=False)[1] ^= 1
    assert alias.to_strings() == ["+Z"]
    assert raw[1] == toZX("Z")[1]


@pytest.mark.parametrize("force_large", [False, True])
def test_imaginary_phases_cannot_leak_into_sign_only_exports(force_large):
    operators = ZXArray.from_input(["+iX", "-iY"], force_large=force_large)
    for export in (operators.legacy_array, operators.pauliint_collection, operators.signs,
                   lambda: operators.data, lambda: operators.packed_values):
        with pytest.raises(ValueError, match="phase"):
            export()
    assert operators.phases().tolist() == [1, 3]
    assert operators.binary().tolist() == [[0, 1], [1, 1]]
    assert Pauli("+iX").equiv(Pauli("-X"))
    assert Pauli("+iX") != Pauli("-X")
    assert Pauli("+iX") == Pauli("iX")
    assert str(Pauli("ix")) == "+IX"


def stabilizer_projector(labels):
    width = dense_label(labels[0]).shape[0]
    out = np.eye(width, dtype=complex)
    for text in labels:
        out = out @ ((np.eye(width) + dense_label(text)) / 2)
    return out


@pytest.mark.parametrize("force_large", [False, True])
def test_group_methods_distinguish_support_and_stabilizer_phases(force_large):
    supports = ZXArray.from_input(["+iXI", "-ZI"], force_large=force_large)
    assert len(supports.center()) == 0
    ambient = supports.centralizer()
    assert len(ambient) == 2
    assert np.all(ambient.commutation_matrix(supports))
    assert not np.any(ambient.z_bits()[:, 0])
    assert not np.any(ambient.x_bits()[:, 0])
    one = ZXArray.from_input(["XI"], force_large=force_large)
    assert len(one.center()) == 1
    assert len(one.centralizer()) == 3
    assert len(ZXArray.from_input(["+X", "-X"]).support_basis()) == 1
    for invalid in (["X", "-X"], ["XX", "ZZ", "YY"], ["X", "Z"], ["+iX"]):
        with pytest.raises(ValueError):
            ZXArray.from_input(invalid, force_large=force_large).stabilizer_basis()
    consistent = ["XX", "ZZ", "-YY"]
    reduced = ZXArray.from_input(consistent, force_large=force_large).stabilizer_basis()
    assert len(reduced) == 2
    np.testing.assert_array_equal(stabilizer_projector(reduced.to_strings()), stabilizer_projector(consistent))


@pytest.mark.parametrize("force_large", [False, True])
def test_empty_and_zero_width_object_algebra(force_large):
    empty = ZXArray.empty(2, force_large=force_large)
    assert empty.to_strings() == []
    assert empty.binary().shape == (0, 4)
    assert empty.symplectic_matrix().shape == (0, 0)
    assert len(empty.support_basis()) == 0
    assert len(empty.center()) == 0
    assert len(empty.centralizer()) == 4
    assert len(empty.stabilizer_basis()) == 0
    assert len(empty @ Pauli("XX")) == 0
    scalar = Pauli("-i", force_large=force_large)
    assert scalar.n_qubits == 0
    assert str(scalar @ scalar) == "-"
    assert str(scalar.pad(2)) == "-iII"
    assert str(Pauli.identity(0, force_large=force_large)) == "+"
    with pytest.raises(ValueError):
        ZXArray.from_input(["-"], force_large=force_large).stabilizer_basis()
    zero = ZXArray.empty(0, force_large=force_large)
    assert len(zero.centralizer()) == 0
    assert len(zero.center()) == 0
