# clustering_legacy.py
# Hang Su 2017
# Usage: see python clustering_legacy.py -h


import argparse
import numpy as np
from utils import TimedBlock, load_ndarray, save_ndarray, DisjointSet


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
