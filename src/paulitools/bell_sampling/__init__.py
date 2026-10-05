"""Compiled Bell-label and symplectic-subspace primitives.

This layer owns numerical operations, not sampling independence, verifier
thresholds, stopping rules, or experimental resource accounting. See the
public API guide in docs/BELL_SAMPLING.md.
"""

from .differences import bell_differences, cyclic_bell_differences
from .filters import (
    BellFilterState, BellSamplePool, bell_filter_mask, bell_filtered_purity,
    bell_purity, commuting_mask, y_parities,
)
from .subspace import SupportBasis, SupportSampler
from .transforms import symplectic_fwht

__all__ = [
    "bell_differences", "cyclic_bell_differences", "y_parities",
    "commuting_mask", "bell_filter_mask", "bell_purity", "bell_filtered_purity",
    "BellSamplePool", "BellFilterState", "SupportBasis", "SupportSampler",
    "symplectic_fwht",
]

from .prefix_geometry import prefix_center_intersection_ranks
__all__.append("prefix_center_intersection_ranks")
