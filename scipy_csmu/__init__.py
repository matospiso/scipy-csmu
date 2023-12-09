__version__ = "0.1.0"

from .handling import (
    split_compressed_data,
    split_compressed_indices,
    sparse_vector_from_indices_and_values,
    sparse_vector_to_indices_and_values,
)
from .matmat import matmat
from .norms import get_norms_along_compressed_axis, get_squared_norms_along_compressed_axis
from .scaling import inplace_scale_along_compressed_axis, inplace_scale_along_uncompressed_axis
from .sparsify import inplace_sparsify, inplace_sparsify_vector
