# clustering.py
# Hang Su 2016
# Usage: see python clustering.py -h


import argparse
import time
import numpy as np


class DisjointSet(object):
    """
    Disjoint-set data structure that supports union/find operations and do-not-merge constraints.
    """
    def __init__(self, values, neg_constraints=None):
        if isinstance(values, int) or (isinstance(values, float) and values % 1 == 0):
            values = range(values)
        self.values = set(values)
        self.parent = dict(zip(values, values))
        self.rank = dict(zip(values, [0] * len(values)))
        self.neg_constraints = neg_constraints

        # a dict with constraints indexed by each cluster
        if neg_constraints is not None:
            neg_constraints_dict = {}
            for v in self.values:
                neg_constraints_dict[v] = set()
            for (f, t) in self.neg_constraints:
                neg_constraints_dict[f].add(t)
                neg_constraints_dict[t].add(f)
            self.neg_constraints_dict = neg_constraints_dict

    def find(self, v):
        """Find the root of v.

        :param v: a node
        :return: the root node of v
        """
        p = self.parent[v]
        if p != v:
            self.parent[v] = self.find(p)
        return self.parent[v]

    def union(self, v1, v2):
        """Union the set that contains v1 and the set that contains v2.

        :param v1: node 1
        :param v2: node 2
        :return: bool, True if a union happens (the number of disjoint sets is reduced by 1)
        """
        rt1, rt2 = self.find(v1), self.find(v2)
        # v1 and v2 are already in the same cluster
        if rt1 == rt2:
            return False

        # v1 and v2 are in different clusters, but there is do-not-merge constraint
        if self.neg_constraints is not None and (min(rt1, rt2), max(rt1, rt2)) in self.neg_constraints:
            return False

        # v1 and v2 are in different clusters, merge them
        rk1, rk2 = self.rank[rt1], self.rank[rt2]
        if rk1 < rk2:
            self.parent[rt1] = rt2
            parent, child = rt2, rt1
        elif rk1 > rk2:
            self.parent[rt2] = rt1
            parent, child = rt1, rt2
        else:
            self.parent[rt2] = rt1
            self.rank[rt1] += 1
            parent, child = rt1, rt2

        # update constraints
        if self.neg_constraints is None:
            return True
        self.neg_constraints_dict[parent].update(self.neg_constraints_dict[child])
        for c in self.neg_constraints_dict[child]:
            self.neg_constraints_dict[c].remove(child)
            self.neg_constraints_dict[c].add(parent)
            self.neg_constraints.remove((min(c, child), max(c, child)))
            self.neg_constraints.add((min(c, parent), max(c, parent)))
        del self.neg_constraints_dict[child]

        return True

    def get_clusters(self):
        """List representation of the clusters.

        :return: a list of clusters, each contains its members in a list
        """
        clusters = {}
        for v in self.values:
            p = self.find(v)
            if p in clusters:
                clusters[p].append(v)
            else:
                clusters[p] = [v]
        return list(clusters.values())


class TimedBlock:
    """
    Context manager that times the execution of a block of code.
    """
    def __init__(self, msg='', verbose=False):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        if self.verbose and self.msg:
            print('{} ...'.format(self.msg), end='', flush=True)
        self.tic = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        toc = time.time()
        if self.verbose and self.msg:
            print(' done! ({:.2f} secs)'.format(toc - self.tic))


def load_ndarray(file_path, dtype=np.float32):
    """Load nd-array from a file."""
    if file_path.endswith('.mat'):
        from scipy.io import loadmat
        mat = loadmat(file_path)
        var_names = list(filter(lambda k: not k.startswith('__'), mat.keys()))
        if len(var_names) != 1:
            raise ValueError('There are {} variables in {}. 1 is expected.'.format(len(var_names), file_path))
        mat = mat[var_names[0]].astype(dtype=dtype)
    elif file_path.endswith('.npy'):
        mat = np.load(file_path).astype(dtype=dtype)
    else:
        raise ValueError('Cannot load data from this file type: {}.'.format(file_path))
    return mat


def save_ndarray(file_path, mat, var_name='A', oned_as='row'):
    """Save nd-array to a file."""
    if file_path.endswith('.mat'):
        from scipy.io import savemat
        savemat(file_path, {var_name: mat}, oned_as=oned_as)
    elif file_path.endswith('.npy'):
        np.save(file_path, mat)
    else:
        raise ValueError('Cannot write data to this file type: {}.'.format(file_path))


def constraints_from_matrix(mat, target_fn=(lambda v: v)):
    """Convert constrain format from matrix representation to set of pairs."""
    constraint_pairs = set()
    if mat is None:
        return constraint_pairs
    for r in range(mat.shape[0]):
        constraint_pairs.update(filter(lambda p: target_fn(mat[p[0], p[1]]), ((r, c) for c in range(r+1, mat.shape[0]))))
    return constraint_pairs


def vector_cluster_repr(clusters, num_samples, start_idx=0, check_completeness=True, dtype=np.int32):
    """Convert cluster format from set of clusters to a vector containing cluster id for each data point."""
    vec = np.zeros(num_samples, dtype=dtype)
    seen = set()
    for i, c in zip(range(start_idx, len(clusters) + start_idx), list(clusters)):
        vec[list(c)] = i
        if check_completeness:
            seen.update(c)
    if check_completeness and len(seen) != num_samples:
        print('Warning: clusters do not cover all the nodes.')
    return vec


def constraint_similarity_clustering(similarity, neg_constraints, threshold=None, num_clusters=None):
    """Clustering w/ do-not-connect constraints.

    :param similarity: an np matrix (float32) storing the pairwise similarity between nodes
    :param neg_constraints: a set of negative constraints as 2-tuples
    :param threshold: pairs with similarity lower than threshold will not be connected
    :param num_clusters: the wanted number of clusters; not that one of threshold and num_clusters need to be None
    :return: a list of clusters, each contains its members in a list; or a list of such lists if num_clusters=0
    """
    if (threshold is None) + (num_clusters is None) != 1:
        raise ValueError('One and only one of threshold and num_clusters should be specified.')

    num_samples = similarity.shape[0]

    # use disjoint-set data structure to maintain the clusters
    clusters = DisjointSet(num_samples, neg_constraints)

    if threshold is not None:
        # construct tuples (i, j, sim) for the pairs in the upper right triangle that meet similarity threshold
        similarity_of_pairs = [(r, c, similarity[r, c])
                               for r in range(num_samples-1)
                               for c in range(r+1, num_samples)
                               if similarity[r, c] >= threshold]

        # sort the pairs according to similarity (from high to low)
        similarity_of_pairs.sort(key=lambda v: v[2], reverse=True)

        # iterate through pairs, update clusters (when not violating any do-not-connect constraints)
        for p in similarity_of_pairs:
            clusters.union(p[0], p[1])

        return clusters.get_clusters()
    elif num_clusters > 0:
        if num_clusters > num_samples:
            print('Warning: {} clusters can not be found on {} samples.'.format(num_clusters, num_samples))

        # construct tuples (i, j, sim) for all pairs in the upper right triangle
        similarity_of_pairs = [(r, c, similarity[r, c])
                               for r in range(num_samples-1)
                               for c in range(r+1, num_samples)]

        # sort the pairs according to similarity (from high to low)
        similarity_of_pairs.sort(key=lambda v: v[2], reverse=True)

        # iterate through pairs, update clusters (when not violating any do-not-connect constraints)
        cur_num_clusters = num_samples
        for p in similarity_of_pairs:
            if cur_num_clusters <= num_clusters:
                break
            cur_num_clusters -= clusters.union(p[0], p[1])
        if cur_num_clusters > num_clusters:
            print('Warning: cannot get {} clusters, got {} instead.'.format(num_clusters, cur_num_clusters))

        return clusters.get_clusters()
    else:
        # construct tuples (i, j, sim) for all pairs in the upper right triangle
        similarity_of_pairs = [(r, c, similarity[r, c])
                               for r in range(num_samples-1)
                               for c in range(r+1, num_samples)]

        # sort the pairs according to similarity (from high to low)
        similarity_of_pairs.sort(key=lambda v: v[2], reverse=True)

        # iterate through pairs, update clusters (when not violating any do-not-connect constraints)
        clusters_hierarchy = [clusters.get_clusters()]
        for p in similarity_of_pairs:
            if clusters.union(p[0], p[1]):
                clusters_hierarchy.append(clusters.get_clusters())

        return clusters_hierarchy


def main():
    parser = argparse.ArgumentParser(description='Constraint Similarity-Based Clustering.')
    parser.add_argument('similarity', type=str, help='Path to a file containing similarity matrix')
    parser.add_argument('output', type=str, help='Output file path')
    parser.add_argument('threshold', type=float, help='Threshold, or the number of clusters when -n is turned on')
    parser.add_argument('-c', '--constraints', type=str, help='Path to a file containing constraints')
    parser.add_argument('-z', '--zero-based', action='store_true', help='Cluster ID will start from 0 if turned on')
    parser.add_argument('-d', '--distance', action='store_true', help='Distance is given rather than similarity')
    parser.add_argument('-n', '--num-clusters', action='store_true', help='Number of clusters should be given if True')
    parser.add_argument('-v', '--verbose', action='store_true', help='Output runtime information when True')
    args = parser.parse_args()

    # load similarity matrix
    with TimedBlock('Loading similarity matrix from {}'.format(args.similarity), verbose=args.verbose):
        similarity = load_ndarray(args.similarity)
    if args.distance:
        similarity *= -1
        if not args.num_clusters:
            args.threshold *= -1

    # load negative constraints (do-not-link constraints)
    if args.constraints:
        with TimedBlock('Loading constraints from {}'.format(args.constraints), verbose=args.verbose):
            constraints = load_ndarray(args.constraints, dtype=np.bool)
        with TimedBlock('Converting constraints', verbose=args.verbose):
            neg_constraints = constraints_from_matrix(constraints)
    else:
        neg_constraints = None

    # compute clusters
    (threshold, num_clusters) = (None, int(args.threshold)) if args.num_clusters else (args.threshold, None)
    with TimedBlock('Computing clusters', verbose=args.verbose):
        clusters = constraint_similarity_clustering(similarity, neg_constraints, threshold, num_clusters)

    # save results
    with TimedBlock('Converting clusters', verbose=args.verbose):
        if isinstance(clusters[0][0], list):
            cluster_hierarchy = clusters
            clusters = np.zeros((len(clusters), similarity.shape[0]), dtype=np.int32)
            for i, c in enumerate(cluster_hierarchy):
                clusters[i] = vector_cluster_repr(c, similarity.shape[0], start_idx=(0 if args.zero_based else 1))
        else:
            clusters = vector_cluster_repr(clusters, similarity.shape[0], start_idx=(0 if args.zero_based else 1))
    with TimedBlock('Saving results to {}'.format(args.output), verbose=args.verbose):
        save_ndarray(args.output, clusters, var_name='clusters')


if __name__ == '__main__':
    main()
