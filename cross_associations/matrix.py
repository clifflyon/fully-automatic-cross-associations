"""
sparse binary matrix
"""
# pylint: disable=invalid-name

from collections import Counter, defaultdict

import numpy as np
from numpy import ceil, exp, isfinite, log2


def entropy_bits(ary):
    """ -log2 of ary, defining -log2(0) ~approx Inf """
    ary += exp(-700)
    return -log2(ary)


def logstar(num):
    """ log-star (universal integer code length) of num """
    ell = 0
    while num > 1:
        ell += 1
        num = log2(num)
    return ell


def int_bits(ary):
    """ log2 of A, defining log2(0) = 0 and rounding up """
    ary[ary == 0] = 1
    return ceil(log2(ary))


def _rc_cluster_cost(ary):
    """ cluster cost for row and column groupings helper """
    dim = len(ary)
    cst = np.cumsum(ary)
    cst = cst[::-1] - dim + range(1, dim + 1)
    return sum(int_bits(cst))


class Matrix:
    """ spare matrix and all the parameters """

    def __init__(self, matrix_object):
        # the _matrix object is assumed to be a csc matrix
        self._matrix = matrix_object
        num_rows, num_cols = self._matrix.shape
        self._row_clusters = [0] * num_rows
        self._col_clusters = [0] * num_cols
        self._D_non_zero = None
        self._cached_max_row_cluster = max(self._row_clusters)
        self._cached_max_col_cluster = max(self._col_clusters)
        self._dok_copy = matrix_object.asformat("dok")
        self._csr_copy = matrix_object.asformat("csr")

    @staticmethod
    def _cluster_sizes(ary, max_idx):
        c = Counter(ary)
        zeros = [0] * (max_idx + 1)
        for cidx in c:
            zeros[cidx] += c[cidx]
        return zeros

    @property
    def matrix(self):
        """
        Return the sparse matrix object
        """
        return self._matrix

    @property
    def col_clusters(self):
        """
        Return array of column cluster assignments
        """
        return self._col_clusters

    @property
    def row_clusters(self):
        """
        Return array of row cluster assignments
        """
        return self._row_clusters

    @staticmethod
    def _resequence(cluster_labels):
        """
        Yield a squashed version of the cluster indexes.
        :param cluster_labels: the cluster index values
        """
        mp = {b: a for a, b in enumerate(sorted(np.unique(cluster_labels)))}
        for elem in cluster_labels:
            yield mp[elem]

    def defrag_clusters(self):
        """
        Remove empty clusters
        """
        self._row_clusters = [_ for _ in self._resequence(self._row_clusters)]
        self._col_clusters = [_ for _ in self._resequence(self._col_clusters)]
        self._D_non_zero = None
        self._cached_max_col_cluster = max(self._col_clusters)
        self._cached_max_row_cluster = max(self._row_clusters)

    @property
    def row_cluster_sizes(self):
        """
        Return array of row cluster sizes, including empty clusters
        """
        return self._cluster_sizes(self._row_clusters,
                                   self._cached_max_row_cluster)

    @property
    def col_cluster_sizes(self):
        """
        Return array of col cluster sizes, including empty clusters
        """
        return self._cluster_sizes(self._col_clusters,
                                   self._cached_max_col_cluster)

    @property
    def num_row_clusters(self):
        """
        Return the number of row clusters
        """
        return len(self.row_cluster_sizes)

    @property
    def num_col_clusters(self):
        """
        Return the number of column clusters
        """
        return len(self.col_cluster_sizes)

    @property
    def D_non_zero(self):
        """
        Return the non-zero counts of the cross associations.

        This is a heavy computation, so we cache and update as much as
        possible
        """
        if self._D_non_zero is None:
            self._D_non_zero = np.zeros(
                (self.num_row_clusters, self.num_col_clusters))
            for i in range(self.num_row_clusters):
                rX = np.ravel(np.nonzero(np.array(self._row_clusters) == i))
                for j in range(self.num_col_clusters):
                    cX = np.ravel(np.nonzero(np.array(self._col_clusters) == j))
                    self._D_non_zero[i, j] = np.sum(self._dok_copy[rX, :][:, cX])       
        return self._D_non_zero

    @property
    def co_cluster_sizes(self):
        """
        Return the co-cluster sizes (Nxy)
        """
        return np.outer(
            self.row_cluster_sizes,
            self.col_cluster_sizes
        )

    def _entropy_terms(self):
        Dnz = self.D_non_zero
        Nxy = self.co_cluster_sizes
        if Dnz.shape != Nxy.shape:
            self._D_non_zero = None
            Dnz = self.D_non_zero
        Dz = Nxy - Dnz

        assert np.all(Dnz >= 0)
        assert np.all(Dz >= 0)

        Pz = Dz / Nxy
        Pz[~isfinite(Pz)] = 0
        Pnz = Dnz / Nxy
        Pnz[~isfinite(Pnz)] = 0
        entropy_terms = np.multiply(Dz, entropy_bits(Pz)) + \
            np.multiply(Dnz, entropy_bits(Pnz))
        return entropy_terms

    @property
    def cost(self):
        """
        returns:(total encoding cost, per-block 0/1s only)
        """
        # col and row
        cost = logstar(self.num_col_clusters) + \
            logstar(self.num_row_clusters)
        # cluster sizes
        cost += self._cluster_size_cost
        cost += int_bits(self.co_cluster_sizes + 1).sum()
        entropy_terms = self._entropy_terms()
        # block cost
        per_block_cost = ceil(entropy_terms.sum())
        cost += per_block_cost
        return cost, per_block_cost

    @property
    def _cluster_size_cost(self):
        """
        Compute the cost of the cluster sizes
        """
        return _rc_cluster_cost(self.row_cluster_sizes) + \
            _rc_cluster_cost(self.col_cluster_sizes)

    def _cluster_entropies(self, orientation="row"):
        """
        Compute the row or column block entropies
        :param orientation: row or col
        """
        entropy_terms = self._entropy_terms()
        if orientation == 'col':
            return np.sum(entropy_terms, axis=0) / self.col_cluster_sizes
        elif orientation == 'row':
            return np.sum(entropy_terms, axis=1) / self.row_cluster_sizes

    def max_entropy_cluster(self, orientation="row"):
        """
        Return the value and index of the worst cluster for the given
        orientation

        :param orientation: row or column
        """
        entropies = self._cluster_entropies(orientation=orientation)
        idx = np.argmax(entropies)
        val = entropies[idx]
        return idx, val

    def entropy_of_cluster(self, index, orientation="row"):
        """
        :param index: index of the cluster to check
        :param orientation: row or col
        """
        return self._cluster_entropies(orientation=orientation)[index]

    def cluster_members(self, index, orientation="row"):
        """
        return the index values that belong to the cluster

        :param index: the index value
        :param orientation: row or column
        """
        if orientation == 'row':
            Q = self._row_clusters
        elif orientation == 'col':
            Q = self._col_clusters
        return [_ for _, i in enumerate(Q) if i == index]

    def _get_row(self, index):
        row_values = self._csr_copy[index]
        # if this is scipy, not numpy, get the proper values
        try:
            row_values = row_values.toarray()[0]
        except AttributeError:
            pass
        return row_values

    def _get_col(self, index):
        col_values = self._matrix[:, index]
        # if this is scipy, not numpy, get the proper values
        try:
            col_values = col_values.toarray()
        except AttributeError:
            pass
        return col_values

    def hstack_dnz(self):
        """
        Add a column for new counts
        """
        extra_cluster = [[0.] for _ in range(self._D_non_zero.shape[0])]
        self._D_non_zero = np.hstack((self.D_non_zero, extra_cluster))
        self._cached_max_col_cluster += 1

    def vstack_dnz(self):
        """
        Add a row for new counts
        """
        extra_cluster = [0. for _ in range(self._D_non_zero.shape[1])]
        self._D_non_zero = np.vstack((self.D_non_zero, extra_cluster))
        self._cached_max_row_cluster += 1

    @property
    def transformed_matrix(self):
        """
        return seriated matrix
        """
        cx = Counter(self.col_clusters)
        idx_x = []
        for cluster_id, _ in cx.most_common():
            idx_x += self.cluster_members(cluster_id, orientation="col")
        cy = Counter(self.row_clusters)
        idx_y = []
        for cluster_id, _ in cy.most_common():
            idx_y += self.cluster_members(cluster_id, orientation="row")
        # this index trick may depend on the type
        return self._matrix[:, idx_x][idx_y]

    def update_col_cluster(self, index, new_col_cluster):
        """
        Update a cluster index
        :param index: the index to update
        :param new_col_cluster: the new col cluster
        """
        old_col_cluster = self._col_clusters[index]
        if old_col_cluster == new_col_cluster:
            return
        col_values = self._get_col(index)
        dnz_deltas = defaultdict(int)
        for i, v in enumerate(col_values):
            dnz_deltas[self._row_clusters[i]] += v
        for k in dnz_deltas:
            self._D_non_zero[k, old_col_cluster] -= dnz_deltas[k]
            self._D_non_zero[k, new_col_cluster] += dnz_deltas[k]

        self._col_clusters[index] = new_col_cluster

    def update_row_cluster(self, index, new_row_cluster):
        """
        Update a cluster index
        :param index: the index to update
        :param new_row_cluster: the new row cluster
        """
        old_row_cluster = self._row_clusters[index]
        if old_row_cluster == new_row_cluster:
            return
        row_values = self._get_row(index)
        dnz_deltas = defaultdict(int)
        for i, v in enumerate(row_values):
            dnz_deltas[self._col_clusters[i]] += v
        for k in dnz_deltas:
            self._D_non_zero[old_row_cluster, k] -= dnz_deltas[k]
            self._D_non_zero[new_row_cluster, k] += dnz_deltas[k]
        self._row_clusters[index] = new_row_cluster
