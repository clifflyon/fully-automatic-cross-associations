"""
perform a cluster search
"""
import logging
import random
import sys
from collections import namedtuple
from math import inf

import numpy as np
import scipy.io

from cross_associations.matrix import Matrix
from cross_associations.plots import plot_shaded

EPSILON = 1e-05


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CostTracker:
    """
    Convenience cost tracking
    """

    def __init__(self):
        self._cost_history = [(0, 0), (0, 0)]

    def track(self, cost):
        """
        Add an observation to the history
        :param cost: tuple of total and block costs
        """
        self._cost_history.append(cost)

    @property
    def current_cost_str(self):
        """
        Construct a human-readable string
        """
        current_cost = self._cost_history[-1]
        previous_cost = self._cost_history[-2]
        return "Global: {} ({}) Block-encoding: {} ({})".format(
            current_cost[0],
            current_cost[0] - previous_cost[0],
            current_cost[1],
            current_cost[1] - previous_cost[1]
        )

    @property
    def matlab_style_cost(self):
        """
        cost 8233 (C2: 8205)
        """
        return "cost {:.0f} (C2: {:.0f})".format(*self._cost_history[-1])

    @property
    def current_block_cost(self):
        """
        Get the current block encoding cost
        """
        return self._cost_history[-1][1]

    @property
    def current_total_cost(self):
        """
        Get the current total cost
        """
        return self._cost_history[-1][0]

    def global_improved(self):
        """
        Evaluate whether the global cost went down
        """
        return self._cost_history[-1][0] < self._cost_history[-2][0]

    def block_improved(self):
        """
        Evaluate whether the block cost went down
        """
        return self._cost_history[-1][1] < self._cost_history[-2][1]


class ClusterSearch:
    """
    Search for a good clustering: add a new cluster, reshape old clusters
    """

    def __init__(self, matrix_object, random_seed=42):
        self._matrix = matrix_object
        self._num_row_clusters = 1
        self._num_col_clusters = 1
        self._random_seed = random_seed
        self._tracker = CostTracker()
        np.random.seed(self._random_seed)
        random.seed(self._random_seed)
        self.best_cost = inf
        self.best_so_far = None

    def track(self):
        """
        Add a tracking observation to the history
        """
        self._tracker.track(self._matrix.cost)

    def get_cost_string(self, style="col"):
        if style == 'col':
            fstring = '## (k={:d}) l={:d} Current {}'
        elif style == 'row':
            fstring = '## k={:d} (l={:d}) Current {}'
        elif style == 'final':
            fstring = '## k={:d} l={:d} Final {}'
        return fstring.format(
            self._matrix.num_row_clusters,
            self._matrix.num_col_clusters,
            self._tracker.matlab_style_cost
        )

    def update_best(self):
        if self._tracker.current_total_cost < self.best_cost:
            self.best_cost = self._tracker.current_total_cost
            self.best_so_far = (
                self._matrix.row_clusters[:],
                self._matrix.col_clusters[:])

    def run(self):
        """
        Run the clustering algorithm
        """
        _axis = namedtuple("Axis", ("add_cluster", "reshape", "style"))
        col_axis = _axis(self.add_col_cluster, self.reshape_col, "col")
        row_axis = _axis(self.add_row_cluster, self.reshape_row, "row")
        self.track()
        logger.debug("## Starting {}".format(self._tracker.matlab_style_cost))
        while True:
            for axis in (col_axis, row_axis):
                axis.add_cluster()
                axis.reshape()
                self._matrix.defrag_clusters()
                self.track()
                logger.debug(self.get_cost_string(style=axis.style))
                self.update_best()
            if self._tracker.current_block_cost == 0:
                break
            if not self._tracker.global_improved():
                break
        (self._matrix._row_clusters,
         self._matrix._col_clusters) = self.best_so_far
        self._matrix.defrag_clusters()
        self.track()
        logger.debug(self.get_cost_string(style="final"))
        plot_shaded(self._matrix)

    def _reshape(self, orientation="row"):
        """
        Given a clustering, see if it can be improved.
        """
        if orientation == "row":
            cluster_labels = self._matrix.row_clusters
            update_function = self._matrix.update_row_cluster
        elif orientation == "col":
            cluster_labels = self._matrix.col_clusters
            update_function = self._matrix.update_col_cluster
        cluster_ids = np.unique(cluster_labels)

        while True:
            start_cost = self._matrix.cost[1]
            for idx, _ in enumerate(cluster_labels):
                best_alternative = -1
                current_cost = float('inf')
                for alternative in cluster_ids:
                    update_function(idx, alternative)
                    new_cost = self._matrix.cost[1]
                    if new_cost < current_cost:
                        current_cost = new_cost
                        best_alternative = alternative
                update_function(idx, best_alternative)

            # in practice, reshape won't ever increase start cost
            if self._matrix.cost[1] >= start_cost:
                break

            logger.debug("Intermediate cost {:.0f} (C2: {:.0f})".format(
                *self._matrix.cost))

    def reshape_col(self):
        """
        reshape the column clusters
        """
        self._reshape(orientation="col")

    def reshape_row(self):
        """
        reshape the row clusters
        """
        self._reshape(orientation="row")

    def add_col_cluster(self):
        """
        Add a new column cluster
        """
        self._add_cluster(orientation="col")

    def add_row_cluster(self):
        """
        Add a new row cluster
        """
        self._add_cluster(orientation="row")

    def _add_cluster(self, orientation="row"):
        """
        Add a new row or column cluster
        """
        # do this while Dnz and Nxy will agree
        cluster_index, cluster_entropy = self._matrix.max_entropy_cluster(
            orientation=orientation)

        if orientation == "col":
            self._matrix.hstack_dnz()
            cluster_labels = self._matrix.col_clusters
            update_function = self._matrix.update_col_cluster
        elif orientation == "row":
            self._matrix.vstack_dnz()
            cluster_labels = self._matrix.row_clusters
            update_function = self._matrix.update_row_cluster

        new_cluster_index = max(cluster_labels) + 1
        for idx in self._matrix.cluster_members(
                cluster_index, orientation=orientation):
            original = cluster_labels[idx]
            update_function(idx, new_cluster_index)
            new_entropy = self._matrix.entropy_of_cluster(
                cluster_index, orientation=orientation)
            if new_entropy <= cluster_entropy - EPSILON:
                cluster_entropy = new_entropy
            else:
                update_function(idx, original)


def main():
    """
    Demo run
    """
    mat_file = 'data/NCP.mat'
    loaded_matrix = scipy.io.loadmat(mat_file)
    matrix = Matrix(loaded_matrix['NCP'])
    cluster_search = ClusterSearch(matrix)
    cluster_search.run()
    return 0

if __name__ == "__main__":

    sys.exit(main())
