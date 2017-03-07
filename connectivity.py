import argparse
import numpy as np
from utils import DisjointSet


def num_edges_make_connected(n, trial_and_error=False):
    graph = DisjointSet(n)
    num_edges, num_components = 0, n
    if trial_and_error:
        edges = set()
    else:
        edges = [(i, j) for i in range(n -1) for j in range(i + 1, n)]
        edges = [edges[i] for i in np.random.permutation(int(0.5 * n * (n - 1)))]
    while num_components > 1:
        if trial_and_error:
            p, q = np.random.randint(n, size=2)
            p, q = int(p), int(q)
            if p == q or (p, q) in edges or (q, p) in edges:
                continue
            edges.add((p, q))
        else:
            p, q = edges[num_edges]
        num_components -= graph.union(p, q)
        num_edges += 1
    return num_edges


def connectivity_simulation(n, num_runs, sample_scheme='trial_and_error', verbose=True):
    num_edges = []
    if sample_scheme == 'trial_and_error':
        trial_and_error = True
    elif sample_scheme == 'pre_shuffle':
        trial_and_error = False
    else:
        raise ValueError('Unknown sampling scheme: {}'.format(sample_scheme))
    for i in range(num_runs):
        num_edges.append(num_edges_make_connected(n, trial_and_error))
        if verbose and (i + 1) % 10 == 0:
            print('.', flush=True, end='')
        if verbose and (i + 1) % 200 == 0:
            print(' {}/{}'.format(i + 1, num_runs), flush=True)
    num_edges.sort()
    idx, total_edges = 0, 0.5 * n * (n - 1)
    connected_ratio = []
    while idx < num_runs:
        cur = num_edges[idx]
        idx += 1
        while idx < num_runs and num_edges[idx] == cur:
            idx += 1
        connected_ratio.append((cur / total_edges, idx / num_runs))
    if verbose:
        print('\ndone!')
    return connected_ratio


def main():
    parser = argparse.ArgumentParser(description='Graph connectivity simulation.')
    parser.add_argument('nodes', type=int, help='Number of nodes in the graph')
    parser.add_argument('runs',type=int, help='Number of runs to simulate')
    parser.add_argument('-v', '--verbose', action='store_true', help='Output runtime information when True')
    parser.add_argument('-d', '--delimiter', type=str, default=', ', help='Delimiter in output')
    parser.add_argument('-s', '--sample-scheme', default='trial_and_error', choices=('pre_shuffle', 'trial_and_error'),
                        help='Sampling scheme [default: \'trial_and_error\']')
    args = parser.parse_args()

    connected_ratio = connectivity_simulation(args.nodes, args.runs, args.sample_scheme, args.verbose)
    for e, r in connected_ratio:
        print('{:.6f}{}{}'.format(e, args.delimiter, r))


if __name__ == '__main__':
    main()
