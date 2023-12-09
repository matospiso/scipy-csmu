from typing import Union

import numpy as np
from scipy import sparse as sp


def inplace_scale_along_compressed_axis(x: Union[sp.csr_matrix, sp.csc_matrix], scale: np.ndarray) -> None:
    """
    Scale rows (columns) of a CSR (CSC) matrix by vector scale.
    :param x: CSR (or CSC) matrix to be scaled
    :param scale: vector of scaling factors
    :return: None
    """
    x.data *= np.repeat(scale, np.diff(x.indptr))


def inplace_scale_along_uncompressed_axis(x: Union[sp.csr_matrix, sp.csc_matrix], scale: np.ndarray) -> None:
    """
    Scale rows (columns) of a CSC (CSR) matrix by vector scale.
    :param x: CSR (or CSC) matrix to be scaled
    :param scale: vector of scaling factors
    :return: None
    """
    x.data *= scale[x.indices]
