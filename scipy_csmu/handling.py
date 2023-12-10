from typing import Union

import numpy as np
import scipy.sparse as sp


def split_compressed_data(A: Union[sp.csr_matrix, sp.csc_matrix]) -> list[np.ndarray]:
    """
    Returns a list of arrays of values of nonzero entries in each row (column) of a CSR (CSC) matrix.
    :param A: CSR (or CSC) matrix
    :return: list of 1D numpy arrays containing values of nonzero elements in each row (column) of input matrix
    """
    list_of_data = []
    for ptr1, ptr2 in zip(A.indptr[:-1], A.indptr[1:]):
        list_of_data.append(A.data[ptr1:ptr2])
    return list_of_data


def split_compressed_indices(A: Union[sp.csr_matrix, sp.csc_matrix]) -> list[np.ndarray]:
    """
    Returns a list of arrays of indices of nonzero entries in each row (column) of a CSR (CSC) matrix.
    :param A: CSR (or CSC) matrix
    :return: list of 1D numpy arrays containing indices of nonzero elements in each row (column) of input matrix
    """
    list_of_indices = []
    for ptr1, ptr2 in zip(A.indptr[:-1], A.indptr[1:]):
        list_of_indices.append(A.indices[ptr1:ptr2])
    return list_of_indices


def sparse_vector_from_indices_and_values(
    indices: np.ndarray,
    values: np.ndarray,
    dimension: int,
    column: bool = False,
) -> Union[sp.csr_matrix, sp.csc_matrix]:
    """
    Transforms a tuple of indices, values and to a sparse vector.
    :param indices: integer 1D numpy array of indices, min(indices) >= 0, max(indices) < dimension
    :param values: 1D numpy array of values
    :param dimension: dimension of the vector
    :param column: whether to return a column vector
    :return: CSR (or CSC) matrix with 1 row (column)
    """
    if column:
        return sp.csc_matrix((values, indices, np.array([0, len(values)])), shape=(dimension, 1))
    else:
        return sp.csr_matrix((values, indices, np.array([0, len(values)])), shape=(1, dimension))


def sparse_vector_to_indices_and_values(vec: Union[sp.csr_matrix, sp.csc_matrix]) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Transforms a sparse vector to a tuple of indices and values.
    :param vec: CSR (or CSC) matrix with 1 row (column)
    :return: tuple of indices, values and dimension of the vector
    """
    if isinstance(vec, sp.csr_matrix):
        return vec.indices, vec.data, vec.shape[1]
    else:
        return vec.indices, vec.data, vec.shape[0]
