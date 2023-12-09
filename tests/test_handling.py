import unittest

import numpy as np
import scipy.sparse as sp

from scipy_csmu.handling import (
    sparse_vector_from_indices_and_values,
    sparse_vector_to_indices_and_values,
    split_compressed_data,
    split_compressed_indices,
)

INDICES = np.array([[0, 2, 3, 5, 6], [0, 1, 3, 4, 6], [1, 2, 4, 5, 6]])
VALUES = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
CSR = sp.csr_matrix((VALUES.ravel(), INDICES.ravel(), np.array([0, 5, 10, 15])), shape=(3, 7))


class TestSparseVectorFromIndicesAndValues(unittest.TestCase):
    def test_csr(self):
        csr = sparse_vector_from_indices_and_values(INDICES[0], VALUES[0])
        np.testing.assert_array_equal(csr.indptr, CSR[0, :].indptr)
        np.testing.assert_array_equal(csr.indices, CSR[0, :].indices)
        np.testing.assert_array_equal(csr.data, CSR[0, :].data)

    def test_csc(self):
        csc = sparse_vector_from_indices_and_values(INDICES[0], VALUES[0], column=True)
        np.testing.assert_array_equal(csc.indptr, CSR[0, :].T.indptr)
        np.testing.assert_array_equal(csc.indices, CSR[0, :].T.indices)
        np.testing.assert_array_equal(csc.data, CSR[0, :].T.data)


class TestSparseVectorToIndicesAndValues(unittest.TestCase):
    def test_csr(self):
        indices, values = sparse_vector_to_indices_and_values(CSR[0, :])
        np.testing.assert_array_equal(indices, INDICES[0])
        np.testing.assert_array_equal(values, VALUES[0])

    def test_csc(self):
        indices, values = sparse_vector_to_indices_and_values(CSR[0, :].T)
        np.testing.assert_array_equal(indices, INDICES[0])
        np.testing.assert_array_equal(values, VALUES[0])


class TestSplitCompressedData(unittest.TestCase):
    def test_csr(self):
        actual = split_compressed_data(CSR)
        expected = [VALUES[0], VALUES[1], VALUES[2]]
        np.testing.assert_array_equal(actual, expected)

    def test_csc(self):
        actual = split_compressed_data(CSR.T)
        expected = [VALUES[0], VALUES[1], VALUES[2]]
        np.testing.assert_array_equal(actual, expected)


class TestSplitCompressedIndices(unittest.TestCase):
    def test_csr(self):
        actual = split_compressed_indices(CSR)
        expected = [INDICES[0], INDICES[1], INDICES[2]]
        np.testing.assert_array_equal(actual, expected)

    def test_csc(self):
        actual = split_compressed_indices(CSR.T)
        expected = [INDICES[0], INDICES[1], INDICES[2]]
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
