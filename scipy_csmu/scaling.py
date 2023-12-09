from typing import Union

import numpy as np
from scipy import sparse as sp


def inplace_scale_along_compressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix], scale: np.ndarray) -> None:
    """
    Scale rows (columns) of a CSR (CSC) matrix by vector scale.
    :param A: CSR (or CSC) matrix to be scaled
    :param scale: vector of scaling factors
    :return: None
    """
    A.data *= np.repeat(scale, np.diff(A.indptr))


def inplace_scale_along_uncompressed_axis(A: Union[sp.csr_matrix, sp.csc_matrix], scale: np.ndarray) -> None:
    """
    Scale rows (columns) of a CSC (CSR) matrix by vector scale.
    :param A: CSR (or CSC) matrix to be scaled
    :param scale: vector of scaling factors
    :return: None
    """
    A.data *= scale[A.indices]
