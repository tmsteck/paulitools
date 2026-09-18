# Import core functions for Pauli string manipulation
from .core import (
    toZX, toString, right_pad, left_pad, append, concatenate_ZX,
    symplectic_inner_product, symplectic_inner_product_int,
    commutes, bsip_array, commute_array_fast,
    unpack_sym_forms_to_matrices, GLOBAL_INTEGER,
    toZX_extended, toString_extended, symplectic_inner_product_extended,
    commutes_extended, to_standard_if_possible
)

# Import group theory functions
from .group import (
    row_reduce, generators, null_space, inner_product, radical, centralizer,
    differences, ingroup, row_space, center, ambient_centralizer,
    stabilizer_reduce, stabilizer_reduce_bits,
)

# Import utility functions
from .util import (
    toBinary, convert_array_type, popcount, getParity,
    get_pauli_obs, get_pauli_pauli_obs, getCentralizer,
    Pauli_expectation, filtered_purity, filtered_purity_reference, get_purity
)

from .large_pauli import (
    MAX_STANDARD_QUBITS,
    PauliInt,
    PauliIntCollection,
    create_pauli_struct,
    pauli_struct_set_bits,
    pauli_struct_get_bits,
    pauli_struct_to_binary,
    pauli_struct_copy,
    toZX_large,
    symplectic_inner_product_struct,
    symplectic_inner_product_pauliint,
    commutes_struct,
    commutes_pauliint,
    commutation_matrix
)

from .zx_array import (
    ZXArray,
    toZXArray,
)

from .pauli import Pauli

from .storage import (
    SerializationError,
    save_pauli_data,
    load_pauli_data,
    load_legacy_payload,
    append_pauli_data,
    iter_pauli_records,
)

from .bell_sampling import (
    BellFilterState, BellSamplePool, SupportBasis, SupportSampler,
    bell_differences, cyclic_bell_differences, y_parities,
    commuting_mask, bell_filter_mask, bell_purity, bell_filtered_purity,
    symplectic_fwht,
)

# Define what gets imported with "from paulitools import *"
__all__ = [
    # Core functions
    'toZX', 'toString', 'right_pad', 'left_pad', 'append', 'concatenate_ZX',
    'symplectic_inner_product', 'symplectic_inner_product_int',
    'commutes', 'bsip_array', 'commute_array_fast',
    'unpack_sym_forms_to_matrices', 'GLOBAL_INTEGER',
    'toZX_extended', 'toString_extended',
    'symplectic_inner_product_extended', 'commutes_extended',
    'to_standard_if_possible',
    
    # Group functions
    'row_reduce', 'generators', 'null_space', 'inner_product', 'radical',
    'centralizer', 'center', 'ambient_centralizer', 'stabilizer_reduce',
    'stabilizer_reduce_bits', 'differences', 'ingroup', 'row_space',
    
    # Utility functions
    'toBinary', 'convert_array_type', 'popcount', 'getParity',
    'get_pauli_obs', 'get_pauli_pauli_obs', 'getCentralizer',
    'Pauli_expectation', 'filtered_purity', 'filtered_purity_reference', 'get_purity',

    # Large Pauli helpers
    'MAX_STANDARD_QUBITS', 'PauliInt', 'PauliIntCollection',
    'create_pauli_struct', 'pauli_struct_set_bits', 'pauli_struct_get_bits',
    'pauli_struct_to_binary', 'pauli_struct_copy',
    'toZX_large', 'symplectic_inner_product_struct', 'symplectic_inner_product_pauliint',
    'commutes_struct', 'commutes_pauliint', 'commutation_matrix',

    # Ergonomic ZX wrapper
    'Pauli', 'ZXArray', 'toZXArray',

    # Serialization helpers
    'SerializationError', 'save_pauli_data', 'append_pauli_data', 'load_pauli_data',
    'load_legacy_payload', 'iter_pauli_records',

    # Bell-label numerics and prepared support spaces
    'BellFilterState', 'BellSamplePool', 'SupportBasis', 'SupportSampler',
    'bell_differences', 'cyclic_bell_differences', 'y_parities',
    'commuting_mask', 'bell_filter_mask', 'bell_purity', 'bell_filtered_purity',
    'symplectic_fwht',
]
