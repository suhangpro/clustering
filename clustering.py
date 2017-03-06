# clustering_legacy.py
# Hang Su 2017
# Usage: see python clustering.py -h


import argparse
import os
import numpy as np
import sklearn.cluster
from sklearn.metrics.pairwise import pairwise_distances
from utils import TimedBlock, load_ndarray, save_ndarray

methods_available = ('AffinityPropagation', 'AgglomerativeClustering', 'Birch', 'DBSCAN', 'KMeans', 'MiniBatchKMeans',
                     'MeanShift', 'SpectralClustering')
methods_can_take_dist = ('AffinityPropagation', 'DBSCAN', 'SpectralClustering')
methods_require_n_clusters = ('AgglomerativeClustering', 'Birch', 'KMeans', 'MiniBatchKMeans', 'SpectralClustering')

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type=str, required=True, help='output path')
parser.add_argument('-f', '--feature', metavar='F', default='', help='path to feature matrix')
parser.add_argument('-d', '--distance', metavar='D', default='', help='path to distance matrix, only compatible with '
                                                                      'methods: ' + '/'.join(methods_can_take_dist))
parser.add_argument('--do-not-link', default='', help='path to do-not-link constraint matrix, only works with methods: '
                                                      + '/'.join(methods_can_take_dist))
parser.add_argument('--must-link', default='', help='path to must-link constraint matrix, only works with methods: '
                                                    + '/'.join(methods_can_take_dist))
parser.add_argument('-z', '--zero-based', action='store_true', help='output with zero-based indexing if enabled')
parser.add_argument('-v', '--verbose', default=1, type=int, help='verbose level 0|1[default]|2')
parser.add_argument('-j', '--n-jobs', default=-1, type=int, help='number of parallel jobs to run (default: -1)')
parser.add_argument('--metric', default='euclidean', help='metric for computing distances (default: euclidean)')
parser.add_argument('--nc', '--n-clusters', default=-1, type=int, help='number of clusters, required for '
                                                                       + '/'.join(methods_require_n_clusters))
parser.add_argument('--max-iter', default=-1, type=int, help='maximum number of iterations (default: '
                                                             'KMeans:300/MiniBatchKMeans:100/AffinityPropagation:200)')
parser.add_argument('--linkage', default='ward', help='which linkage criterion to use (default: ward)')
parser.add_argument('-m', '--method', default='KMeans', choices=methods_available,
                    help='specifies which clustering algorithm to use (default: KMeans)')
args = parser.parse_args()


def cluster(feat, dist, opts):
    if (feat is not None) + (dist is not None) != 1:
        raise ValueError('Only one of {feat|dist} should be given')
    if feat is None and opts.method not in methods_can_take_dist:
        raise ValueError('{} expects feature matrix, but got distance matrix/constraints'.format(opts.method))
    if opts.nc == -1 and opts.method in methods_require_n_clusters:
        raise ValueError('n_clusters is required for method {}'.format(opts.method))
    x = feat if feat is not None else dist

    if opts.method == 'AffinityPropagation':
        if opts.max_iter == -1:
            opts.max_iter = 200
        if dist is not None:
            model = sklearn.cluster.AffinityPropagation(max_iter=opts.max_iter, affinity='precomputed',
                                                        verbose=opts.verbose)
            x *= -1
        else:
            model = sklearn.cluster.AffinityPropagation(max_iter=opts.max_iter, affinity='euclidean',
                                                        verbose=opts.verbose)
    elif opts.method == 'AgglomerativeClustering':
        model = sklearn.cluster.AgglomerativeClustering(n_clusters=opts.nc, affinity=opts.metric)
    elif opts.method == 'Birch':
        model = sklearn.cluster.Birch(n_clusters=opts.nc)
    elif opts.method == 'DBSCAN':
        if dist is not None:
            model = sklearn.cluster.DBSCAN(metric='precomputed', n_jobs=opts.n_jobs)
        else:
            model = sklearn.cluster.DBSCAN(metric=opts.metric, n_jobs=opts.n_jobs)
    elif opts.method == 'KMeans':
        if opts.max_iter == -1:
            opts.max_iter = 300
        model = sklearn.cluster.KMeans(n_clusters=opts.nc, max_iter=opts.max_iter, n_jobs=opts.n_jobs,
                                       verbose=opts.verbose)
    elif opts.method == 'MiniBatchKMeans':
        if opts.max_iter == -1:
            opts.max_iter = 100
        model = sklearn.cluster.MiniBatchKMeans(n_clusters=opts.nc, max_iter=opts.max_iter,
                                                verbose=opts.verbose)
    elif opts.method == 'MeanShift':
        model = sklearn.cluster.MeanShift(n_jobs=opts.n_jobs)
    elif opts.method == 'SpectralClustering':
        if dist is not None:
            delta = 1.
            x = np.exp(- dist ** 2 / (2. * delta ** 2))
            model = sklearn.cluster.SpectralClustering(n_clusters=opts.nc, affinity='precomputed',
                                                       n_jobs=opts.n_jobs)
        else:
            model = sklearn.cluster.SpectralClustering(n_clusters=opts.nc, affinity='rbf', n_jobs=opts.n_jobs)
    else:
        raise ValueError('Unsupported algorithm: {}'.format(opts.method))
    return model.fit(x)


def main():
    global args
    verbose = args.verbose > 0
    if verbose:
        print(args)
    dtype = np.float64
    inf_float = np.finfo(np.float32).max / 10  # TODO make sure that this somewhat arbitrary choice is not a problem

    # load data
    if args.feature != '':
        with TimedBlock(msg='Loading feature matrix from '+args.feature, verbose=verbose):
            feat = load_ndarray(args.feature, dtype)
        _, task_name = os.path.split(args.feature)
    else:
        feat = None
    if args.distance != '':
        with TimedBlock(msg='Loading distance matrix from '+args.distance, verbose=verbose):
            dist = load_ndarray(args.distance, dtype)
            dist = np.triu(dist, 1).T + np.triu(dist, 0)
        _, task_name = os.path.split(args.distance)
    else:
        dist = None
    if args.do_not_link != '':
        with TimedBlock(msg='Loading do-not-link constraint matrix from '+args.do_not_link, verbose=verbose):
            do_not_link = load_ndarray(args.do_not_link, dtype=np.bool)
            do_not_link = np.triu(do_not_link, 1).T + np.triu(do_not_link, 0)
    else:
        do_not_link = None
    if args.must_link != '':
        with TimedBlock(msg='Loading must-link constraint matrix from '+args.must_link, verbose=verbose):
            must_link = load_ndarray(args.must_link, dtype=np.bool)
            must_link = np.triu(must_link, 1).T + np.triu(must_link, 0)
    else:
        must_link = None

    if not (args.output.endswith('.mat') or args.output.endswith('.npy')):
        try:
            os.makedirs(args.output)
        except OSError:
            pass
        args.output = os.path.join(args.output, task_name[:-4]+'_{}'.format(args.method)+task_name[-4:])

    # some consolidation
    if (feat is not None) + (dist is not None) != 1:
        raise ValueError('Only one of {--feature|--distance} should be given')
    if must_link is not None or do_not_link is not None:
        if dist is None:
            with TimedBlock(msg='Computing {} distance'.format(args.metric), verbose=verbose):
                dist = pairwise_distances(feat, metric=args.metric)
            feat = None
        if must_link is not None:
            dist[must_link] = 0.0
        if do_not_link is not None:
            assert dist.max() < inf_float
            dist[do_not_link] = inf_float

    # run clustering
    args.verbose = args.verbose >= 2
    with TimedBlock(msg='Running {}'.format(args.method), verbose=verbose):
        model = cluster(feat, dist, args)
    labels = model.labels_
    if not args.zero_based:
        labels += 1
    with TimedBlock(msg='Saving results to {}'.format(args.output), verbose=verbose):
        save_ndarray(args.output, labels, 'labels')


if __name__ == '__main__':
    main()
